from types import SimpleNamespace as NS

import pytest

from legal_rag import ct


class FakeClient:
    """Versions, aliases, runs and metrics: enough of MlflowClient for the gate."""

    def __init__(self, metrics, aliases=None):
        self.versions = {1: "run-prod", 2: "run-cand"}
        self.metrics = metrics
        self.aliases = dict(aliases or {"production": 1})
        self.runs = []

    def search_model_versions(self, _filter):
        return [NS(version=str(v), run_id=r, tags={}) for v, r in self.versions.items()]

    def get_model_version_by_alias(self, name, alias):
        from mlflow.exceptions import MlflowException
        from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST

        if alias not in self.aliases:
            raise MlflowException("no alias", error_code=RESOURCE_DOES_NOT_EXIST)
        v = self.aliases[alias]
        return NS(version=str(v), run_id=self.versions[v], tags={})

    def get_run(self, run_id):
        return NS(info=NS(run_id=run_id), data=NS(metrics=self.metrics[run_id]))

    def set_registered_model_alias(self, name, alias, version):
        self.aliases[alias] = int(version)

    def get_experiment_by_name(self, name):
        return NS(experiment_id="7") if name == "legal-rag-ct" else None

    def search_runs(self, ids, filter_string="", order_by=None, max_results=1):
        return self.runs[:max_results]


def make(prod=0.8, cand=0.9):
    return FakeClient({"run-prod": {"mrr": prod}, "run-cand": {"mrr": cand}})


def test_a_better_candidate_goes_to_staging_not_production():
    client = make(0.8, 0.9)
    decision, version = ct.run_gate("run-cand", "mrr", 0.0, client)
    assert decision.promote and version == "2"
    assert client.aliases == {"production": 1, "staging": 2}


def test_a_worse_candidate_changes_nothing():
    client = make(0.9, 0.8)
    decision, version = ct.run_gate("run-cand", "mrr", 0.0, client)
    assert not decision.promote and version == "2" and client.aliases == {"production": 1}


def test_a_tie_does_not_win_but_a_negative_margin_lets_it_through():
    assert not ct.run_gate("run-cand", "mrr", 0.0, make(0.8, 0.8))[0].promote
    assert ct.run_gate("run-cand", "mrr", -0.01, make(0.8, 0.8))[0].promote


def test_the_margin_must_be_beaten():
    assert not ct.run_gate("run-cand", "mrr", 0.05, make(0.80, 0.83))[0].promote
    assert ct.run_gate("run-cand", "mrr", 0.05, make(0.80, 0.90))[0].promote


def test_an_unregistered_candidate_is_an_error():
    with pytest.raises(LookupError):
        ct.run_gate("run-unknown", "mrr", 0.0, make())


def test_latest_run_id():
    client = make()
    client.runs = [NS(info=NS(run_id="newest")), NS(info=NS(run_id="older"))]
    assert ct.latest_run_id(client, "legal-rag-ct", "any") == "newest"
    with pytest.raises(LookupError):
        ct.latest_run_id(client, "no-such-experiment", "any")
    client.runs = []
    with pytest.raises(LookupError):
        ct.latest_run_id(client, "legal-rag-ct", "any")


def test_summary_says_what_happened():
    won, version = ct.run_gate("run-cand", "mrr", 0.0, make(0.8, 0.9))
    text = ct.summary_markdown(won, version, "mrr", 0.0)
    assert "promoted to **staging**" in text and "0.9000" in text and "0.8000" in text
    lost, version = ct.run_gate("run-cand", "mrr", 0.0, make(0.9, 0.8))
    assert "**not** promoted" in ct.summary_markdown(lost, version, "mrr", 0.0)


def test_main_exits_zero_when_the_challenge_is_lost(monkeypatch, tmp_path, capsys):
    out, summary = tmp_path / "out", tmp_path / "summary"
    monkeypatch.setenv("GITHUB_OUTPUT", str(out))
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    monkeypatch.setattr(ct.registry, "_client", lambda: make(0.9, 0.8))
    assert ct.main(["gate", "--run-id", "run-cand", "--metric", "mrr"]) == 0
    assert "promoted=false" in out.read_text() and "NOT promoted" in capsys.readouterr().out
    assert "not** promoted" in summary.read_text()


def test_main_writes_the_outputs_for_a_win(monkeypatch, tmp_path):
    out = tmp_path / "out"
    monkeypatch.setenv("GITHUB_OUTPUT", str(out))
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    monkeypatch.setattr(ct.registry, "_client", lambda: make(0.8, 0.9))
    assert ct.main(["gate", "--run-id", "run-cand"]) == 0
    assert out.read_text() == "promoted=true\nversion=2\n"
