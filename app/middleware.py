# app/middleware.py

"""The X-Request-ID middleware."""

from collections.abc import Awaitable, Callable

from fastapi import Request, Response

import app as app_pkg
from core.request_id import request_id_var, sanitize_request_id


@app_pkg.app.middleware("http")
async def request_id_middleware(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """Bind the correlation id (X-Request-ID or a fresh one) to the request.

    The id lives in a ContextVar, so every log line of the request (and of the
    background tasks it spawns) carries it. The context is reset afterwards so
    the next request (or a loop task) starts from an empty id. A client id that
    is not a plain short token is replaced: it is echoed in a header, logged and
    used in request-log file names.
    """
    rid = sanitize_request_id(request.headers.get("x-request-id", ""))
    token = request_id_var.set(rid)
    try:
        response = await call_next(request)
    finally:
        request_id_var.reset(token)
    response.headers["X-Request-ID"] = rid
    return response
