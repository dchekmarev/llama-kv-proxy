# tests/test_request_id.py

"""Request correlation id: a ContextVar propagated to log records and echoed
in the X-Request-ID response header by the middleware."""

import logging
from types import SimpleNamespace

import app as app_module
from request_id import RequestIdFilter, new_request_id, request_id_var


def test_new_request_id_is_short_and_unique():
    a, b = new_request_id(), new_request_id()
    assert len(a) == 12
    assert a != b


def test_filter_sets_request_id_from_context():
    record = logging.LogRecord(
        name="t",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="hi",
        args=(),
        exc_info=None,
    )
    # Outside a request: empty id, filter still passes the record through.
    assert RequestIdFilter().filter(record) is True
    assert record.request_id == ""

    token = request_id_var.set("abc123")
    try:
        assert RequestIdFilter().filter(record) is True
        assert record.request_id == "abc123"
    finally:
        request_id_var.reset(token)


class _FakeResponse:
    def __init__(self) -> None:
        self.headers: dict[str, str] = {}


async def test_middleware_uses_incoming_header():
    req = SimpleNamespace(headers={"x-request-id": "incoming-123"})
    captured: dict[str, str] = {}

    async def call_next(_req):
        captured["rid"] = request_id_var.get()
        return _FakeResponse()

    resp = await app_module.request_id_middleware(req, call_next)
    assert captured["rid"] == "incoming-123"
    assert resp.headers["X-Request-ID"] == "incoming-123"


async def test_middleware_generates_id_when_missing():
    req = SimpleNamespace(headers={})
    captured: dict[str, str] = {}

    async def call_next(_req):
        captured["rid"] = request_id_var.get()
        return _FakeResponse()

    resp = await app_module.request_id_middleware(req, call_next)
    assert len(captured["rid"]) == 12
    assert resp.headers["X-Request-ID"] == captured["rid"]


async def test_middleware_resets_context_after_request():
    """The contextvar is reset after the handler runs (no leak to the next req)."""
    req = SimpleNamespace(headers={})

    async def call_next(_req):
        return _FakeResponse()

    await app_module.request_id_middleware(req, call_next)
    assert request_id_var.get() == ""
