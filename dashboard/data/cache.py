"""In-memory caching for parquet/JSON reads.

Recorded runs are immutable once written, so caching raw reads keyed on
their arguments (runs dir, run id, epoch) is always safe. This is what keeps
epoch/rollout switching fast (requirements.md section 38: filter changes
<200ms, map updates <500ms) without hand-rolling an LRU.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Callable, TypeVar

F = TypeVar("F", bound=Callable)

_cached_funcs: list = []


def cached(maxsize: int = 32) -> Callable[[F], F]:
    def decorator(func: F) -> F:
        wrapped = lru_cache(maxsize=maxsize)(func)
        _cached_funcs.append(wrapped)
        return wrapped  # type: ignore[return-value]

    return decorator


def clear_all() -> None:
    """Drop every cached read. Used by tests that regenerate fixtures."""

    for func in _cached_funcs:
        func.cache_clear()
