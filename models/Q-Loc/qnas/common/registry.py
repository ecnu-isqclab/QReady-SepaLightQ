from __future__ import annotations

from typing import Callable


class Registry(dict[str, Callable]):
    def register(self, name: str, fn: Callable | None = None):
        def decorator(item: Callable) -> Callable:
            self[name] = item
            return item

        return decorator(fn) if fn is not None else decorator


SAMPLERS = Registry()
SCORERS = Registry()
