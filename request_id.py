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
import re
import uuid

# Empty outside a request (background loops, startup).
request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "request_id", default=""
)

# A correlation id reaches log lines, the X-Request-ID response header and the
# request-log file names, so only a short plain token is accepted from a client.
_SAFE_ID = re.compile(r"[A-Za-z0-9._:-]{1,64}")


def new_request_id() -> str:
    """A short, unique-enough id for a request that did not send one."""
    return uuid.uuid4().hex[:12]


def sanitize_request_id(rid: str) -> str:
    """Return a correlation id that is safe to log, echo and put in a filename.

    A client id outside the safe token charset (or longer than the id can be)
    is replaced with a generated one instead of being repaired, so a forged id
    can never end up in a path or a log line.
    """
    if rid and len(rid) <= 64 and _SAFE_ID.fullmatch(rid):
        return rid
    return new_request_id()


class RequestIdFilter(logging.Filter):
    """Attach the current request id to every log record (empty if none)."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        return True
