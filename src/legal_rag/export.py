"""Export the multilingual-e5 encoder to ONNX with dynamic batch and sequence axes.

Only the transformer is exported (input_ids + attention_mask -> last_hidden_state).
Mean pooling and L2 normalisation stay outside the graph.
"""

import logging
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

from legal_rag.config import settings
from legal_rag.logging_conf import configure_logging, correlation_context

log = logging.getLogger(__name__)

OPSET = 17


class _Encoder(torch.nn.Module):
    """Wrap the HF model so the exported graph has a plain tensor output."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state


def export_onnx(model_name: str | None = None, out_dir: Path | None = None) -> Path:
    name = model_name or settings.embedding_model
    out_dir = out_dir or settings.onnx_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "model.onnx"

    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModel.from_pretrained(name).eval()
    sample = tokenizer(
        ["query: العقد شريعة المتعاقدين", "passage: The contract makes the law of the parties."],
        padding=True,
        return_tensors="pt",
    )
    axes = {0: "batch", 1: "seq"}
    log.info("exporting to ONNX", extra={"model": name, "path": str(out_path)})
    with torch.no_grad():
        torch.onnx.export(
            _Encoder(model),
            (sample["input_ids"], sample["attention_mask"]),
            str(out_path),
            input_names=["input_ids", "attention_mask"],
            output_names=["last_hidden_state"],
            dynamic_axes={"input_ids": axes, "attention_mask": axes, "last_hidden_state": axes},
            opset_version=OPSET,
        )
    tokenizer.save_pretrained(out_dir)

    import onnx  # export-only dependency (uv group "export")

    onnx.checker.check_model(str(out_path))
    total = sum(p.stat().st_size for p in out_dir.glob("model.onnx*"))
    log.info("onnx export ok", extra={"size_mb": round(total / 1e6, 1)})
    return out_path


def main() -> None:
    configure_logging(settings.log_level)
    with correlation_context():
        export_onnx()


if __name__ == "__main__":
    main()
