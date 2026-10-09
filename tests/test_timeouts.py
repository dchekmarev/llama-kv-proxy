# tests/test_timeouts.py

"""P2-2: control-plane timeouts must be configurable via env, not hardcoded
in app/routes.py. Retry-After / connect-timeout: the dead-backend retry
contract (RETRY_AFTER_S header value, CONNECT_TIMEOUT bound on the connect
phase)."""

import importlib
import time

import httpx
import pytest

import chat_flow
from core import config


def test_acquire_timeout_default():
    assert config.ACQUIRE_TIMEOUT == pytest.approx(1500.0)


def test_chat_flow_uses_config_acquire_timeout():
    """chat_flow must take the timeout from config, not hardcode its own copy."""
    assert chat_flow.ACQUIRE_TIMEOUT == config.ACQUIRE_TIMEOUT


def test_acquire_timeout_from_env(monkeypatch):
    monkeypatch.setenv("ACQUIRE_TIMEOUT", "42")
    importlib.reload(config)
    try:
        assert config.ACQUIRE_TIMEOUT == pytest.approx(42.0)
    finally:
        # Restore the env BEFORE the final reload: reloading while the
        # monkeypatched value is still set would leave it baked into the
        # module for every later test (monkeypatch undoes the env only after
        # the test body).
        monkeypatch.delenv("ACQUIRE_TIMEOUT")
        importlib.reload(config)


def test_retry_after_default():
    assert config.RETRY_AFTER_S == 300


def test_chat_flow_uses_config_retry_after():
    """chat_flow must take Retry-After from config, not hardcode its own."""
    assert chat_flow.RETRY_AFTER_S == config.RETRY_AFTER_S


def test_retry_after_from_env(monkeypatch):
    monkeypatch.setenv("RETRY_AFTER_S", "42")
    importlib.reload(config)
    try:
        assert config.RETRY_AFTER_S == 42
    finally:
        monkeypatch.delenv("RETRY_AFTER_S")
        importlib.reload(config)


def test_connect_timeout_default():
    assert config.CONNECT_TIMEOUT == pytest.approx(10.0)


def test_hold_timeout_default():
    assert config.HOLD_TIMEOUT == pytest.approx(240.0)


def test_hold_timeout_from_env(monkeypatch):
    monkeypatch.setenv("HOLD_TIMEOUT", "42")
    importlib.reload(config)
    try:
        assert config.HOLD_TIMEOUT == pytest.approx(42.0)
    finally:
        monkeypatch.delenv("HOLD_TIMEOUT")
        importlib.reload(config)


def test_connect_timeout_from_env(monkeypatch):
    monkeypatch.setenv("CONNECT_TIMEOUT", "42")
    importlib.reload(config)
    try:
        assert config.CONNECT_TIMEOUT == pytest.approx(42.0)
    finally:
        monkeypatch.delenv("CONNECT_TIMEOUT")
        importlib.reload(config)


async def test_llama_client_bounds_connect_phase(monkeypatch):
    """The pooled client must bound the connect phase separately: a
    black-holed backend (SYN dropped) must fail within CONNECT_TIMEOUT and
    surface as a fast 502, not after REQUEST_TIMEOUT."""
    from backend.llama_client import LlamaClient, _request_timeout

    monkeypatch.setattr(config, "CONNECT_TIMEOUT", 3.0)
    timeout = _request_timeout()
    assert timeout.connect == 3.0
    assert timeout.read == config.REQUEST_TIMEOUT

    client = LlamaClient("http://127.0.0.1:9")
    try:
        assert client.client.timeout.connect == 3.0
        assert client.client.timeout.read == config.REQUEST_TIMEOUT
    finally:
        await client.close()


async def test_every_per_call_timeout_is_phased(monkeypatch):
    """Each explicit per-call timeout (chat stream/non-stream, slot action,
    delete) must be the phased timeout: a revert to the bare REQUEST_TIMEOUT
    float would put the connect phase back to 1500s on those calls."""
    from backend.llama_client import LlamaClient

    monkeypatch.setattr(config, "CONNECT_TIMEOUT", 3.0)
    client = LlamaClient("http://127.0.0.1:9")
    seen: list = []

    class _Resp:
        def __init__(self):
            self.status_code = 200
            self.text = "{}"
            self.headers: dict[str, str] = {}

        def json(self):
            return {"choices": []}

    async def _post(*args, **kwargs):
        seen.append(kwargs["timeout"])
        return _Resp()

    async def _send(*args, **kwargs):
        return _Resp()

    def _build_request(*args, **kwargs):
        seen.append(kwargs["timeout"])
        return object()

    async def _delete(*args, **kwargs):
        seen.append(kwargs["timeout"])
        return _Resp()

    client.client.post = _post
    client.client.send = _send
    client.client.build_request = _build_request
    client.client.delete = _delete
    try:
        await client.chat_completions({"messages": []}, slot_id=0, stream=False)
        await client.chat_completions({"messages": []}, slot_id=0, stream=True)
        await client.erase_slot(0)
        await client.delete_cache_file("key")
    finally:
        await client.close()

    assert len(seen) == 4, "chat (2) + slot action + delete must all pass a timeout"
    assert all(t.connect == 3.0 for t in seen), seen
    assert all(t.read == config.REQUEST_TIMEOUT for t in seen)


async def test_connect_timeout_bounds_real_blackholed_connect(monkeypatch):
    """Behavioral check of the README claim: a black-holed address (SYN
    dropped, no RST) fails with httpx.ConnectTimeout within CONNECT_TIMEOUT,
    not after REQUEST_TIMEOUT."""
    from backend.llama_client import LlamaClient

    monkeypatch.setattr(config, "CONNECT_TIMEOUT", 0.5)
    monkeypatch.setattr(config, "REQUEST_TIMEOUT", 60)
    client = LlamaClient("http://192.0.2.1:81")  # TEST-NET-1: unrouted
    try:
        t0 = time.monotonic()
        try:
            await client.chat_completions(
                {"messages": []}, slot_id=0, stream=False
            )
        except httpx.ConnectTimeout:
            pass
        except httpx.ConnectError:
            pytest.skip("network answers TEST-NET (RST instead of drop)")
        else:
            pytest.fail("expected ConnectTimeout against a black-holed address")
        elapsed = time.monotonic() - t0
        assert elapsed < 5.0, f"connect phase took {elapsed:.1f}s"
    finally:
        await client.close()
