"""Task queue facade.

Uses RQ when Redis/RQ is configured, otherwise falls back to immediate in-process
execution while still persisting task status in the database.
"""

from __future__ import annotations

import os
from typing import Any, Callable


class TaskQueue:
    def __init__(self) -> None:
        self.redis_url = os.getenv("REDIS_URL", "")
        self._queue = None
        if self.redis_url:
            try:
                from redis import Redis
                from rq import Queue

                self._queue = Queue("shuxi", connection=Redis.from_url(self.redis_url))
            except Exception:
                self._queue = None

    @property
    def mode(self) -> str:
        return "rq" if self._queue is not None else "inline"

    def enqueue(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> str:
        if self._queue is not None:
            job = self._queue.enqueue(fn, *args, **kwargs)
            return str(job.id)
        fn(*args, **kwargs)
        return "inline"
