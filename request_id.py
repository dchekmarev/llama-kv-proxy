# request_id.py

"""Per-request correlation id, propagated to log records.

A short id is taken from the incoming X-Request-ID header (or generated) and
stored in a ContextVar. The RequestIdFilter copies it onto every log record so
all lines from one request can be grepped together. Background tasks spawned
during a request inherit the id (asyncio copies the context); loop tasks
started at startup carry an empty id.
"""

import contextvars
import logging
import uuid

# Empty outside a request (background loops, startup).
request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "request_id", default=""
)


def new_request_id() -> str:
    """A short, unique-enough id for a request that did not send one."""
    return uuid.uuid4().hex[:12]


class RequestIdFilter(logging.Filter):
    """Attach the current request id to every log record (empty if none)."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        return True
