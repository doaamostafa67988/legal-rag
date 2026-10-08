"""A small decorator that logs how long a function took (DEBUG, so INFO stays quiet)."""

import functools
import logging
import time
from collections.abc import Callable
from typing import ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")


def timed(fn: Callable[P, R]) -> Callable[P, R]:
    log = logging.getLogger(fn.__module__)

    @functools.wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        start = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:  # logged even when fn raises
            log.debug(
                "timed",
                extra={
                    "func": fn.__qualname__,
                    "duration_ms": round((time.perf_counter() - start) * 1000, 1),
                },
            )

    return wrapper
