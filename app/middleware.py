# app/middleware.py

"""The X-Request-ID middleware."""

from collections.abc import Awaitable, Callable

from fastapi import Request, Response

import app as app_pkg
from request_id import new_request_id, request_id_var


@app_pkg.app.middleware("http")
async def request_id_middleware(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """Bind the correlation id (X-Request-ID or a fresh one) to the request.

    The id lives in a ContextVar, so every log line of the request (and of the
    background tasks it spawns) carries it. The context is reset afterwards so
    the next request (or a loop task) starts from an empty id.
    """
    rid = request.headers.get("x-request-id") or new_request_id()
    token = request_id_var.set(rid)
    try:
        response = await call_next(request)
    finally:
        request_id_var.reset(token)
    response.headers["X-Request-ID"] = rid
    return response
