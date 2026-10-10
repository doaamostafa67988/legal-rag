import logging

import pytest

from legal_rag.timing import timed


def test_timed_keeps_the_result_and_the_name_and_logs_duration(caplog):
    @timed
    def add(a, b):
        return a + b

    with caplog.at_level(logging.DEBUG):
        assert add(1, 2) == 3
    assert add.__name__ == "add"
    assert caplog.records[-1].duration_ms >= 0


def test_timed_logs_even_when_the_function_raises(caplog):
    @timed
    def boom():
        raise ValueError("x")

    with caplog.at_level(logging.DEBUG), pytest.raises(ValueError):
        boom()
    assert caplog.records[-1].func.endswith("boom")
