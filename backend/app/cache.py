
from __future__ import annotations

import functools
import threading
import time
from typing import Any, Callable, TypeVar

from .config import settings

F = TypeVar("F", bound=Callable[..., Any])

_store: dict[tuple, tuple[float, Any]] = {}
_lock = threading.Lock()


def ttl_cache(ttl: int | None = None) -> Callable[[F], F]:
    seconds = ttl if ttl is not None else settings.cache_ttl_seconds

    def decorator(fn: F) -> F:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            key = (fn.__module__, fn.__qualname__, args,
                   tuple(sorted(kwargs.items())))
            now = time.monotonic()
            with _lock:
                hit = _store.get(key)
                if hit is not None and hit[0] > now:
                    return hit[1]
            value = fn(*args, **kwargs)
            with _lock:
                _store[key] = (now + seconds, value)
            return value

        wrapper.cache_clear = lambda: clear()       # type: ignore[attr-defined]
        return wrapper                              # type: ignore[return-value]

    return decorator


def clear() -> None:
    with _lock:
        _store.clear()


def stats() -> dict[str, int]:
    with _lock:
        now = time.monotonic()
        return {"entries": len(_store),
                "live": sum(1 for exp, _ in _store.values() if exp > now)}
