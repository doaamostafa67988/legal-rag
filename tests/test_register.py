import pytest

from legal_rag.register import spec_from_params

PARAMS = {
    "embedding_model": "intfloat/multilingual-e5-base",
    "variants": "ar,en",
    "fusion": "weighted",
    "weights": "0.5,0.5",
    "corpus_sha256": "abc",
}


def test_spec_reads_a_weighted_run():
    s = spec_from_params(PARAMS, "rid", "e5-base / alpha=0.5")
    assert s.variants == ("ar", "en") and s.weights == (0.5, 0.5) and s.source_run_id == "rid"


def test_spec_reads_a_run_without_weights():
    s = spec_from_params({**PARAMS, "fusion": "max", "weights": ""}, "rid", "n")
    assert s.weights == ()


def test_spec_keeps_the_plus_in_a_single_variant():
    params = {**PARAMS, "fusion": "single", "variants": "ar+en", "weights": ""}
    s = spec_from_params(params, "r", "n")
    assert s.variants == ("ar+en",)


def test_spec_needs_the_params_a_finalist_logs():
    with pytest.raises(KeyError):
        spec_from_params({"fusion": "max"}, "rid", "n")
