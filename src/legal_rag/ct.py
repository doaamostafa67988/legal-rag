"""Glue between the continuous-training workflow and the model registry.

    python -m legal_rag.ct run-id --experiment legal-rag-ct --run-name "<run name>"
    python -m legal_rag.ct gate --run-id <id> --metric mrr --margin 0.0

`gate` is the promotion gate: the candidate takes the `staging` alias only if it beats the version
behind `production` on the metric by more than the margin. A candidate that does not win is
*not* a failure: the attempt is logged and the command exits 0, so the pipeline stays green and
simply stops (nothing is built, nothing moves). Real errors (the candidate is not registered, the
registry is down) do raise. Production is never touched here; a person moves it.

When run by GitHub Actions the result goes to `$GITHUB_OUTPUT` (promoted, version) and to the job
summary (`$GITHUB_STEP_SUMMARY`).
"""

import argparse
import os
import sys

from legal_rag import registry
from legal_rag.registry import PRODUCTION, STAGING, Decision


def latest_run_id(client, experiment: str, run_name: str) -> str:
    """The newest run with this name in the experiment (the candidate the pipeline just made)."""
    exp = client.get_experiment_by_name(experiment)
    if exp is None:
        raise LookupError(f"no experiment named {experiment!r}")
    runs = client.search_runs(
        [exp.experiment_id],
        filter_string=f"tags.mlflow.runName = '{run_name}'",
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise LookupError(f"no run named {run_name!r} in {experiment!r}")
    return runs[0].info.run_id


def run_gate(run_id: str, metric: str, margin: float, client=None) -> tuple[Decision, str | None]:
    """Challenge production with the candidate; returns the decision and the candidate's version."""
    client = client or registry._client()
    decision = registry.promote_if_better(
        run_id, metric, target=STAGING, against=PRODUCTION, margin=margin, client=client
    )
    version = registry.version_for_run(client, run_id)
    return decision, (version.version if version else None)


def summary_markdown(decision: Decision, version: str | None, metric: str, margin: float) -> str:
    incumbent = "none" if decision.incumbent is None else f"{decision.incumbent:.4f}"
    verdict = (
        f"candidate (version {version}) promoted to **{STAGING}**"
        if decision.promote
        else f"candidate (version {version}) **not** promoted; {PRODUCTION} is unchanged"
    )
    return (
        "### Continuous training: promotion gate\n\n"
        f"- {verdict}\n"
        f"- metric `{metric}`: candidate {decision.candidate:.4f}, "
        f"production {incumbent}, margin {margin}\n"
        f"- reason: {decision.reason}\n"
    )


def _append(env_var: str, text: str) -> None:
    path = os.environ.get(env_var)
    if path:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(text)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m legal_rag.ct")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_id = sub.add_parser("run-id", help="print the id of the newest run with this name")
    p_id.add_argument("--experiment", required=True)
    p_id.add_argument("--run-name", required=True)
    p_gate = sub.add_parser("gate", help="promote to staging only if it beats production")
    p_gate.add_argument("--run-id", required=True)
    p_gate.add_argument("--metric", default="mrr")
    p_gate.add_argument("--margin", type=float, default=0.0)
    args = ap.parse_args(argv)

    if args.cmd == "run-id":
        sys.stdout.write(latest_run_id(registry._client(), args.experiment, args.run_name) + "\n")
        return 0

    decision, version = run_gate(args.run_id, args.metric, args.margin)
    verdict = "PROMOTED to staging" if decision.promote else "NOT promoted"
    sys.stdout.write(
        f"{verdict}: {decision.reason} "
        f"(candidate {decision.candidate}, production {decision.incumbent})\n"
    )
    _append("GITHUB_OUTPUT", f"promoted={str(decision.promote).lower()}\nversion={version or ''}\n")
    _append("GITHUB_STEP_SUMMARY", summary_markdown(decision, version, args.metric, args.margin))
    return 0  # a lost challenge is not a build failure


if __name__ == "__main__":
    sys.exit(main())
