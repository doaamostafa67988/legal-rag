"""The model registry of the retriever: versions, aliases and the promotion rule.

MLflow 3 deprecates registry *stages* (None / Staging / Production) in favour of *aliases*, so the
handbook's lifecycle is expressed as aliases: a version with no alias is "None", `staging` and
`production` are the two stages. Serving loads `models:/legal-rag-retriever@production`, so
moving the `production` alias changes what is served without touching code or rebuilding.

    uv run python -m legal_rag.registry show
    uv run python -m legal_rag.registry alias --version 1 --alias staging
    uv run python -m legal_rag.registry alias --version 1 --alias production   # a human decision
    uv run python -m legal_rag.registry challenge --run-id <run id> --metric mrr --target staging

`challenge` is `promote_if_better`: the candidate takes the alias only if it beats the version
currently behind `against` (default `production`) on the metric.
"""

import argparse
import sys
from dataclasses import dataclass

NAME = "legal-rag-retriever"
STAGING = "staging"
PRODUCTION = "production"
EPSILON = 1e-12  # float noise must not turn a tie into a win


@dataclass(frozen=True)
class Decision:
    promote: bool
    candidate: float
    incumbent: float | None
    reason: str


def decide(
    candidate: float,
    incumbent: float | None,
    margin: float = 0.0,
    higher_is_better: bool = True,
) -> Decision:
    """Does the candidate beat the incumbent by more than `margin`? A tie never wins.

    No incumbent means nothing is being served yet, so the candidate is promoted.
    """
    if incumbent is None:
        return Decision(True, candidate, None, "nothing is behind the alias yet")
    gain = (candidate - incumbent) if higher_is_better else (incumbent - candidate)
    if gain > margin + EPSILON:
        return Decision(True, candidate, incumbent, f"better by {gain:.4f} (margin {margin})")
    return Decision(
        False, candidate, incumbent, f"gain {gain:.4f} does not exceed the margin {margin}"
    )


def source_run_id(version) -> str:
    """The run a version was registered from (the tag is ours; run_id is MLflow's own)."""
    return (getattr(version, "tags", None) or {}).get("source_run_id") or version.run_id


def version_for_run(client, run_id: str, name: str = NAME):
    """The newest registered version that came from this run, or None."""
    versions = client.search_model_versions(f"name='{name}'")
    matches = [v for v in versions if source_run_id(v) == run_id]
    return max(matches, key=lambda v: int(v.version)) if matches else None


def alias_version(client, alias: str, name: str = NAME):
    """The version behind an alias, or None when the alias does not exist.

    Only "does not exist" means None. Any other failure (server down, bad credentials) must
    raise: a gate that treats an outage as "no incumbent" would promote anything.
    """
    from mlflow.exceptions import MlflowException

    try:
        return client.get_model_version_by_alias(name, alias)
    except MlflowException as exc:
        if exc.error_code != "RESOURCE_DOES_NOT_EXIST":
            raise
        return None


def metric_of(client, version, metric: str) -> float:
    run = client.get_run(source_run_id(version))
    if metric not in run.data.metrics:
        have = sorted(run.data.metrics)
        raise KeyError(f"run {run.info.run_id} has no metric {metric!r}; it has {have}")
    return float(run.data.metrics[metric])


def promote_if_better(
    candidate_run_id: str,
    metric: str = "hit_at_3",
    *,
    target: str = PRODUCTION,
    against: str = PRODUCTION,
    margin: float = 0.0,
    higher_is_better: bool = True,
    name: str = NAME,
    client=None,
) -> Decision:
    """Move `target` to the candidate's version if it beats the version behind `against`.

    The handbook's metric is MAE (lower is better); retrieval quality is hit@3 / MRR / faithfulness,
    so `higher_is_better` defaults to True. Continuous training passes target="staging": the
    pipeline may fill staging, but production is moved by a person.
    """
    client = client or _client()
    candidate = version_for_run(client, candidate_run_id, name)
    if candidate is None:
        raise LookupError(f"run {candidate_run_id} has no version of {name!r}: register it first")
    incumbent = alias_version(client, against, name)
    if incumbent is not None and incumbent.version == candidate.version:
        value = metric_of(client, candidate, metric)
        return Decision(False, value, None, f"already behind {against!r}")
    decision = decide(
        metric_of(client, candidate, metric),
        None if incumbent is None else metric_of(client, incumbent, metric),
        margin,
        higher_is_better,
    )
    if decision.promote:
        client.set_registered_model_alias(name, target, candidate.version)
    return decision


def _client():
    from mlflow import MlflowClient, set_tracking_uri

    from legal_rag.config import settings

    set_tracking_uri(settings.mlflow_tracking_uri)
    return MlflowClient()


def show(client, name: str = NAME) -> str:
    versions = sorted(client.search_model_versions(f"name='{name}'"), key=lambda v: int(v.version))
    if not versions:
        return f"{name}: no versions yet"
    # aliases are stored on the registered model ({alias: version}); search results omit them
    by_alias = client.get_registered_model(name).aliases
    lines = [name]
    for v in versions:
        mine = sorted(a for a, version in by_alias.items() if str(version) == str(v.version))
        lines.append(f"  v{v.version}  [{', '.join(mine) or 'None'}]  run {source_run_id(v)[:8]}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m legal_rag.registry")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("show", help="versions and their aliases")
    p_alias = sub.add_parser("alias", help="point an alias at a version (a manual promotion)")
    p_alias.add_argument("--version", required=True)
    p_alias.add_argument("--alias", required=True, choices=[STAGING, PRODUCTION])
    p_ch = sub.add_parser("challenge", help="promote_if_better")
    p_ch.add_argument("--run-id", required=True)
    p_ch.add_argument("--metric", default="hit_at_3")
    p_ch.add_argument("--target", default=PRODUCTION, choices=[STAGING, PRODUCTION])
    p_ch.add_argument("--against", default=PRODUCTION, choices=[STAGING, PRODUCTION])
    p_ch.add_argument("--margin", type=float, default=0.0)
    p_ch.add_argument("--lower-is-better", action="store_true")
    args = ap.parse_args(argv)

    client = _client()
    if args.cmd == "show":
        sys.stdout.write(show(client) + "\n")
    elif args.cmd == "alias":
        client.set_registered_model_alias(NAME, args.alias, args.version)
        sys.stdout.write(show(client) + "\n")
    else:
        d = promote_if_better(
            args.run_id,
            args.metric,
            target=args.target,
            against=args.against,
            margin=args.margin,
            higher_is_better=not args.lower_is_better,
            client=client,
        )
        verdict = f"PROMOTED to {args.target}" if d.promote else "NOT promoted"
        sys.stdout.write(
            f"{verdict}: {d.reason} (candidate {d.candidate}, incumbent {d.incumbent})\n"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
