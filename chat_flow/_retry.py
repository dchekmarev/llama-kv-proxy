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

import httpx


class BackendRetry(Exception):
    """Raised inside an attempt on a retryable backend failure.

    The attempt owns no response bytes yet, so it can unwind and be re-run
    from scratch with the same correlation id and request group.
    """


# The transport failures the proxy retries. A retry re-dispatches the attempt,
# so it must never be used when the backend may have received the request (and
# started or finished generating): a pool timeout, a refused or timed-out
# connect, and a stale pooled connection all mean the request never reached a
# serving backend, while a read/write timeout or a connection reset after the
# request was sent is ambiguous.
#
# RemoteProtocolError is the one ambiguous entry, kept because it is the stale
# pooled connection ("Server disconnected without sending a response", the
# common case the retry exists for). The same type also fires on a mid-response
# close ("... incomplete chunked read"), which may follow real generation; that
# path is retried too, the accepted cost of keeping a racing stale connection
# transparent.
_RETRYABLE_TRANSPORT_ERRORS: tuple[type[httpx.RequestError], ...] = (
    httpx.ConnectError,  # connection refused or failed before the request
    httpx.ConnectTimeout,  # the connection never established
    httpx.PoolTimeout,  # no pooled connection became available
    httpx.RemoteProtocolError,  # a stale pooled connection / mid-body close
)


def is_retryable_transport_error(exc: httpx.HTTPError) -> bool:
    """Whether a transport failure happened before the backend acted.

    A read/write timeout or a connection reset after the request was sent is
    deliberately excluded: the backend may have started generating, and a retry
    would recompute the answer for nothing. ``RemoteProtocolError`` is the one
    exception -- it is the stale-pooled-connection signal the retry exists for,
    even though the same exception type can also surface on a mid-response
    close.
    """
    return isinstance(exc, _RETRYABLE_TRANSPORT_ERRORS)


def backoff_seconds(attempt: int, base: float) -> float:
    """Seconds to wait before retry number ``attempt + 1``.

    Exponential: ``base * 2**attempt`` (``attempt`` 0 is the first retry).
    A ``base`` of 0 waits nothing.
    """
    return base * (2**attempt)
