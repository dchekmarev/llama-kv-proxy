# chat_flow/_retry.py

"""Proxy-side retry of a retryable backend failure before a response commits.

A stale pooled connection, a backend restart or a transient upstream 5xx must
not be bounced to the client as 502 + Retry-After while a retry could still
succeed: the proxy holds the client connection open and re-runs the attempt.
The failure is signalled with :class:`BackendRetry`, which unwinds the attempt
through its own cleanup (releasing the slot) exactly like
:class:`chat_flow._pause.PausedRetry`, and the pipeline's retry loop re-dispatches
with the same correlation id and request group, up to ``BACKEND_RETRY_ATTEMPTS``
attempts. A failure after the response has committed (the first byte reached the
client) is a real stream truncation and is never retried.
"""


class BackendRetry(Exception):
    """Raised inside an attempt on a retryable backend failure.

    The attempt owns no response bytes yet, so it can unwind and be re-run
    from scratch with the same correlation id and request group.
    """


def backoff_seconds(attempt: int, base: float) -> float:
    """Seconds to wait before retry number ``attempt + 1``.

    Exponential: ``base * 2**attempt`` (``attempt`` 0 is the first retry).
    A ``base`` of 0 waits nothing.
    """
    return base * (2**attempt)
