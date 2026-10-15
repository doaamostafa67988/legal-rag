from types import SimpleNamespace as NS

import pytest

from legal_rag import registry
from legal_rag.registry import Decision, decide, promote_if_better, show


def test_decide_without_incumbent_promotes():
    assert decide(0.5, None).promote is True


def test_decide_tie_never_wins():
    assert decide(0.9, 0.9).promote is False


def test_decide_needs_to_beat_the_margin():
    assert decide(0.93, 0.9, margin=0.05).promote is False
    assert decide(0.96, 0.9, margin=0.05).promote is True


def test_decide_lower_is_better():
    assert decide(1.0, 2.0, higher_is_better=False).promote is True
    assert decide(3.0, 2.0, higher_is_better=False).promote is False


class FakeClient:
    """Just enough of MlflowClient: versions, aliases and run metrics."""

    def __init__(self, versions, metrics, aliases=None):
        self.versions = versions  # {version: run_id}
        self.metrics = metrics  # {run_id: {metric: value}}
        self.aliases = dict(aliases or {})  # {alias: version}
        self.set_calls = []

    def _mv(self, version):
        return NS(version=str(version), run_id=self.versions[version], tags={}, aliases=[])

    def search_model_versions(self, _filter):
        return [self._mv(v) for v in self.versions]

    def get_model_version_by_alias(self, name, alias):
        from mlflow.exceptions import MlflowException
        from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST

        if alias not in self.aliases:
            raise MlflowException("no such alias", error_code=RESOURCE_DOES_NOT_EXIST)
        return self._mv(self.aliases[alias])

    def get_run(self, run_id):
        return NS(info=NS(run_id=run_id), data=NS(metrics=self.metrics[run_id]))

    def set_registered_model_alias(self, name, alias, version):
        self.aliases[alias] = int(version)
        self.set_calls.append((alias, str(version)))


def make(aliases=None):
    return FakeClient(
        versions={1: "run-a", 2: "run-b"},
        metrics={"run-a": {"hit_at_3": 0.9, "mrr": 0.8}, "run-b": {"hit_at_3": 0.95, "mrr": 0.7}},
        aliases=aliases,
    )


def test_promotes_when_nothing_is_in_production():
    client = make()
    d = promote_if_better("run-b", client=client)
    assert d.promote and client.set_calls == [("production", "2")]


def test_promotes_a_better_candidate():
    client = make({"production": 1})
    assert promote_if_better("run-b", client=client).promote is True
    assert client.aliases["production"] == 2


def test_keeps_production_when_candidate_is_worse():
    client = make({"production": 2})
    d = promote_if_better("run-a", client=client)
    assert d.promote is False and client.set_calls == []


def test_staging_target_leaves_production_alone():
    client = make({"production": 1})
    promote_if_better("run-b", target="staging", client=client)
    assert client.aliases == {"production": 1, "staging": 2}


def test_candidate_already_in_production_is_a_noop():
    client = make({"production": 2})
    d = promote_if_better("run-b", client=client)
    assert d.promote is False and "already" in d.reason and client.set_calls == []


def test_unregistered_run_is_an_error():
    with pytest.raises(LookupError):
        promote_if_better("run-z", client=make())


def test_missing_metric_is_an_error():
    with pytest.raises(KeyError):
        promote_if_better("run-b", metric="faithfulness", client=make())


def test_other_registry_errors_are_not_read_as_no_incumbent():
    from mlflow.exceptions import MlflowException
    from mlflow.protos.databricks_pb2 import TEMPORARILY_UNAVAILABLE

    client = make()

    def boom(name, alias):
        raise MlflowException("server down", error_code=TEMPORARILY_UNAVAILABLE)

    client.get_model_version_by_alias = boom
    with pytest.raises(MlflowException):
        promote_if_better("run-b", client=client)


def test_show_lists_versions_and_aliases():
    client = make()
    client.search_model_versions = lambda _f: [
        NS(version="1", run_id="abcdef123456", tags={}, aliases=[]),  # search omits aliases
        NS(version="2", run_id="fedcba654321", tags={}, aliases=[]),
    ]
    client.get_registered_model = lambda _n: NS(aliases={"production": "1", "staging": "1"})
    out = show(client)
    assert "v1  [production, staging]" in out and "v2  [None]" in out
    assert isinstance(registry.decide(1, 0), Decision)
