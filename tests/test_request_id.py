# tests/test_request_id.py

"""Request correlation id: a ContextVar propagated to log records and echoed
in the X-Request-ID response header by the middleware."""

import logging
import re
from types import SimpleNamespace

import pytest

import app as app_module
from core.request_id import (
    RequestIdFilter,
    new_request_id,
    request_id_var,
    sanitize_request_id,
)

_SAFE_ID = re.compile(r"[A-Za-z0-9._:-]{1,64}")


def test_new_request_id_is_short_and_unique():
    a, b = new_request_id(), new_request_id()
    assert len(a) == 12
    assert a != b


def test_sanitize_request_id_keeps_plain_token():
    assert sanitize_request_id("incoming-123") == "incoming-123"
    assert sanitize_request_id("trace.abc:9") == "trace.abc:9"


@pytest.mark.parametrize(
    "raw",
    [
        "../../etc/passwd",
        "a/b",
        "a b",
        "a\nb",
        "a\rb",
        "id;rm -rf /",
        "x" * 200,
        "",
        " ",
    ],
)
def test_sanitize_request_id_replaces_unsafe(raw):
    """The id reaches log lines, response headers and request-log file names,
    so anything outside a short plain token is replaced, not repaired."""
    rid = sanitize_request_id(raw)
    assert rid != raw
    assert _SAFE_ID.fullmatch(rid), f"unsafe id {rid!r} survived"


async def test_middleware_replaces_unsafe_header():
    req = SimpleNamespace(headers={"x-request-id": "../../evil\nX-Injected: 1"})
    captured: dict[str, str] = {}

    async def call_next(_req):
        captured["rid"] = request_id_var.get()
        return _FakeResponse()

    resp = await app_module.request_id_middleware(req, call_next)
    assert _SAFE_ID.fullmatch(captured["rid"]), f"unsafe id {captured['rid']!r} bound"
    assert resp.headers["X-Request-ID"] == captured["rid"]


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
