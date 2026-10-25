"""Step 7b (CAT 4): the same encoder through eager PyTorch, ONNX Runtime and OpenVINO.

    uv run python scripts/benchmark_runtimes.py --out reports/runtimes.md

What it does, in order:
  1. loads the encoder in each runtime (eager PyTorch, ONNX Runtime with graph optimisations off
     and on, OpenVINO FP32 converted from the ONNX file, OpenVINO INT8 quantised with NNCF);
  2. times single questions and batches of 4, with the same CPU thread count everywhere
     (`--threads`, default 2 = the CPU limit of the load tests);
  3. measures what each runtime costs in accuracy, two ways: how close its question vectors are to
     the eager ones (cosine), and the retrieval quality on the dev questions (hit@3, MRR) over the
     cached passage vectors, so the numbers say what changes for the service.

The OpenVINO IR files are written to models/e5-small-openvino/ and -int8/ (track them with DVC).
Run it with nothing else heavy open: a shared CPU makes the times noisy.
"""

import argparse
import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

Runner = Callable[[np.ndarray, np.ndarray], np.ndarray]  # (input_ids, attention_mask) -> hidden
Tokenizer = Callable[[list[str]], dict[str, np.ndarray]]
REPEATS = 5
BATCH = 4


# ---- runtimes: each returns a Runner ---------------------------------------------------------


def eager_runner(model_name: str, threads: int) -> Runner:
    import torch
    from transformers import AutoModel

    torch.set_num_threads(threads)
    model = AutoModel.from_pretrained(model_name).eval()

    def run(input_ids: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            out = model(
                input_ids=torch.from_numpy(input_ids),
                attention_mask=torch.from_numpy(attention_mask),
            )
        return out.last_hidden_state.numpy()

    return run


def ort_runner(onnx_path: Path, graph_optimisation: bool, threads: int) -> Runner:
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if graph_optimisation
        else ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    )
    options.intra_op_num_threads = threads
    session = ort.InferenceSession(str(onnx_path), options, providers=["CPUExecutionProvider"])

    def run(input_ids: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        feed = {"input_ids": input_ids, "attention_mask": attention_mask}
        return session.run(None, feed)[0]

    return run


def convert_to_openvino(onnx_path: Path, ir_path: Path) -> Path:
    """ONNX -> OpenVINO IR (what the `ovc` command does), written as model.xml + model.bin.

    The weights stay FP32 (`compress_to_fp16=False`): the default would store them as FP16 and the
    row called "FP32" would not be.
    """
    import openvino as ov

    ir_path.parent.mkdir(parents=True, exist_ok=True)
    ov.save_model(ov.convert_model(onnx_path), ir_path, compress_to_fp16=False)
    return ir_path


def openvino_runner(ir_path: Path, threads: int) -> Runner:
    import openvino as ov

    compiled = ov.Core().compile_model(
        ir_path,
        "CPU",
        {"PERFORMANCE_HINT": "LATENCY", "INFERENCE_NUM_THREADS": threads},
    )

    def run(input_ids: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        result = compiled({"input_ids": input_ids, "attention_mask": attention_mask})
        return result[compiled.output(0)]

    return run


def quantize_to_int8(
    ir_path: Path, calibration: Sequence[dict[str, np.ndarray]], out_path: Path
) -> Path:
    """Post-training INT8 quantisation with NNCF, calibrated on real inputs (one item per call)."""
    import nncf
    import openvino as ov

    model = ov.Core().read_model(ir_path)
    quantized = nncf.quantize(
        model,
        nncf.Dataset(list(calibration)),
        model_type=nncf.ModelType.TRANSFORMER,
        subset_size=len(calibration),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ov.save_model(quantized, out_path)
    return out_path


# ---- encoding: tokenise, run, pool, normalise (as OnnxEmbedder does) ---------------------------


class Encoder:
    def __init__(self, tokenizer: Tokenizer, runner: Runner) -> None:
        self._tokenizer, self._runner = tokenizer, runner

    def __call__(self, texts: list[str]) -> np.ndarray:
        enc = self._tokenizer(texts)
        ids = enc["input_ids"].astype(np.int64)
        mask = enc["attention_mask"].astype(np.int64)
        hidden = self._runner(ids, mask)
        weights = mask[..., None].astype(hidden.dtype)
        pooled = (hidden * weights).sum(axis=1) / np.clip(weights.sum(axis=1), 1e-9, None)
        return pooled / np.linalg.norm(pooled, axis=1, keepdims=True)

    def embed_query(self, text: str) -> list[float]:
        return self([f"query: {text}"])[0].tolist()

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return self([f"passage: {t}" for t in texts]).tolist()


# ---- measurements ----------------------------------------------------------------------------


@dataclass
class Timing:
    p50_ms: float
    p95_ms: float
    batch_ms_per_question: float


def percentile(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def time_encoder(
    encode: Callable[[list[str]], object],
    questions: Sequence[str],
    repeats: int = REPEATS,
) -> Timing:
    """Per-call latency of single questions, and per-question time when 4 go in one call."""
    texts = [f"query: {q}" for q in questions]
    for text in texts[:10]:  # warm-up: first calls pay one-off initialisation
        encode([text])
    single = []
    for _ in range(repeats):
        for text in texts:
            start = time.perf_counter()
            encode([text])
            single.append((time.perf_counter() - start) * 1000)
    batched = []
    for _ in range(repeats):
        start = time.perf_counter()
        for i in range(0, len(texts), BATCH):
            encode(texts[i : i + BATCH])
        batched.append((time.perf_counter() - start) * 1000 / len(texts))
    return Timing(percentile(single, 0.5), percentile(single, 0.95), statistics.median(batched))


def cosine_to_reference(vectors: np.ndarray, reference: np.ndarray) -> tuple[float, float]:
    """Mean and minimum cosine between each vector and the reference runtime's vector."""
    sims = (vectors * reference).sum(axis=1)  # rows are L2-normalised
    return float(sims.mean()), float(sims.min())


def table(rows: list[dict], n: int, threads: int) -> str:
    base = next((r for r in rows if r["name"] == "eager PyTorch"), rows[0])
    header = (
        f"{n} questions, {threads} CPU threads, single-question latency (p50 / p95 of each call) "
        f"and 4 questions per call; accuracy against {base['name']}"
    )
    columns = (
        "| Runtime | p50 (ms) | p95 (ms) | per question, 4 per call (ms) "
        f"| speed-up vs {base['name']} (p50) | cosine to reference (mean / min) | hit@3 | MRR |"
    )
    lines = [header, "", columns, "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        t = r["timing"]
        lines.append(
            f"| {r['name']} | {t.p50_ms:.1f} | {t.p95_ms:.1f} | {t.batch_ms_per_question:.1f} "
            f"| {base['timing'].p50_ms / t.p50_ms:.2f}x | {r['cos_mean']:.4f} / {r['cos_min']:.4f} "
            f"| {r['hit3']:.3f} | {r['mrr']:.3f} |"
        )
    return "\n".join(lines) + "\n"


# ---- main ------------------------------------------------------------------------------------


def main() -> None:
    from transformers import AutoTokenizer

    from legal_rag.config import settings
    from legal_rag.data import load_corpus
    from legal_rag.evaluate import evaluate, load_questions
    from legal_rag.experiments import DEFAULT_CACHE, MatrixRetriever, embedded_corpus

    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--n", type=int, default=100, help="questions timed")
    ap.add_argument("--calibration", type=int, default=200, help="inputs for INT8 calibration")
    ap.add_argument("--skip-eager", action="store_true")
    ap.add_argument("--skip-int8", action="store_true")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    onnx_dir = settings.onnx_dir
    onnx_path = onnx_dir / "model.onnx"
    tokenizer_hf = AutoTokenizer.from_pretrained(onnx_dir)

    def tokenizer(texts: list[str]) -> dict[str, np.ndarray]:
        return tokenizer_hf(
            texts, padding=True, truncation=True, max_length=512, return_tensors="np"
        )

    dev = load_questions()
    articles = load_corpus()
    questions = [q.question for q in dev]
    questions += ["ما حكم: " + " ".join(a.text_ar.split()[:15]) for a in articles if a.text_ar]
    questions = questions[: args.n]

    ir_fp32 = onnx_dir.parent / "e5-small-openvino" / "model.xml"
    ir_int8 = onnx_dir.parent / "e5-small-openvino-int8" / "model.xml"
    convert_to_openvino(onnx_path, ir_fp32)

    runtimes: list[tuple[str, Callable[[], Runner]]] = []
    if not args.skip_eager:
        runtimes.append(
            (
                "eager PyTorch",
                lambda: eager_runner(settings.embedding_model, args.threads),
            )
        )
    runtimes += [
        (
            "ONNX Runtime, graph optimisation off",
            lambda: ort_runner(onnx_path, False, args.threads),
        ),
        (
            "ONNX Runtime, graph optimisation on",
            lambda: ort_runner(onnx_path, True, args.threads),
        ),
        ("OpenVINO FP32", lambda: openvino_runner(ir_fp32, args.threads)),
    ]
    if not args.skip_int8:
        calibration_texts = [f"query: {q}" for q in questions[: args.calibration // 2]]
        calibration_texts += [
            f"passage: {a.text_ar}" for a in articles[: args.calibration // 2] if a.text_ar
        ]
        items = [
            {
                k: v.astype(np.int64)
                for k, v in tokenizer([t]).items()
                if k in {"input_ids", "attention_mask"}
            }
            for t in calibration_texts
        ]
        quantize_to_int8(ir_fp32, items, ir_int8)
        runtimes.append(("OpenVINO INT8", lambda: openvino_runner(ir_int8, args.threads)))

    # passage vectors come from the cache the experiments built; no runtime is timed on them
    model_name = settings.embedding_model
    matrices = [
        embedded_corpus(
            _PassageEncoder(onnx_path, tokenizer),
            model_name,
            articles,
            v,
            DEFAULT_CACHE,
        )
        for v in ("ar", "en")
    ]

    rows, reference = [], None
    for name, build in runtimes:
        encoder = Encoder(tokenizer, build())
        vectors = encoder([f"query: {q}" for q in questions])
        reference = vectors if reference is None else reference
        mean, low = cosine_to_reference(vectors, reference)
        retriever = MatrixRetriever(encoder, articles, matrices, fusion="max")
        result = evaluate(retriever, dev)["overall"]
        rows.append(
            {
                "name": name,
                "timing": time_encoder(encoder, questions),
                "cos_mean": mean,
                "cos_min": low,
                "hit3": result["hit@3"],
                "mrr": result["mrr"],
            }
        )
        print(f"done: {name}", flush=True)

    text = table(rows, len(questions), args.threads)
    print(text)
    if args.out:
        args.out.write_text(text, encoding="utf-8")


class _PassageEncoder:
    """Only used when the passage-vector cache is missing: encodes the corpus with ONNX Runtime."""

    def __init__(self, onnx_path: Path, tokenizer: Tokenizer) -> None:
        self._encoder = Encoder(tokenizer, ort_runner(onnx_path, True, 2))

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), 32):
            out.extend(self._encoder.embed_passages(texts[i : i + 32]))
        return out


if __name__ == "__main__":
    main()
