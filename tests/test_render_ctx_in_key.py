# tests/test_render_ctx_in_key.py

"""Render-context params in the KV-cache key.

Top-level request params that change the RENDERED prompt (reasoning_effort,
enable_thinking, reasoning_budget_tokens, tools) are NOT part of the cache key
today: two requests with identical messages but a different reasoning level
hash to the same key. Restore then matches, yet llama.cpp renders a different
prompt and discards the KV (the cached_tokens=0 anomalies). These tests pin the
intended contract: a change in the reasoning/render params must produce a
different key."""

import pytest

import app as app_module
import hashing as hs

MSG = {"role": "user", "content": "same content"}
CTX = {"reasoning_effort": "low", "enable_thinking": True}


class FakeRequest:
    def __init__(self, data):
        self._data = data

    async def json(self):
        return self._data


class _Spy:
    """Records the request-side key and render_ctx of every chat() run."""

    def __init__(self):
        self.keys: list[str] = []
        self.ctxs: list[object] = []
        self._real = hs.request_prefix_values_async

    async def __call__(self, messages, model_id, wpb, include_reasoning=False, render_ctx=None):
        result = await self._real(messages, model_id, wpb, include_reasoning, render_ctx)
        self.keys.append(result[1])
        self.ctxs.append(render_ctx)
        return result


async def _setup(sm, monkeypatch, bodies) -> _Spy:
    """Run body through app.chat() and collect the computed cache keys."""
    app_module.app.state.sm = sm
    app_module.app.state.clients = [sm.backends[0]["client"]]
    spy = _Spy()
    monkeypatch.setattr(hs, "request_prefix_values_async", spy)
    for body in bodies:
        await app_module.chat(FakeRequest(body))
    return spy


def _body(**extra) -> dict:
    data = {"messages": [MSG], "stream": False}
    data.update(extra)
    return data


@pytest.mark.asyncio
async def test_same_params_same_key(sm, monkeypatch):
    """Identical messages and params always hash to the same key."""
    spy = await _setup(sm, monkeypatch, [_body(), _body()])
    assert len(spy.keys) == 2
    assert spy.keys[0] == spy.keys[1]


@pytest.mark.asyncio
async def test_different_reasoning_effort_different_key(sm, monkeypatch):
    """reasoning_effort changes the rendered prompt -> the key must change."""
    spy = await _setup(
        sm,
        monkeypatch,
        [_body(reasoning_effort="low"), _body(reasoning_effort="high")],
    )
    assert len(spy.keys) == 2
    assert spy.keys[0] != spy.keys[1]


@pytest.mark.asyncio
async def test_different_enable_thinking_different_key(sm, monkeypatch):
    """enable_thinking toggles template reasoning -> the key must change."""
    spy = await _setup(
        sm,
        monkeypatch,
        [_body(enable_thinking=True), _body(enable_thinking=False)],
    )
    assert len(spy.keys) == 2
    assert spy.keys[0] != spy.keys[1]


@pytest.mark.asyncio
async def test_different_reasoning_budget_different_key(sm, monkeypatch):
    """reasoning_budget_tokens changes the reasoning budget -> the key must
    change."""
    spy = await _setup(
        sm,
        monkeypatch,
        [_body(reasoning_budget_tokens=1024), _body(reasoning_budget_tokens=2048)],
    )
    assert len(spy.keys) == 2
    assert spy.keys[0] != spy.keys[1]


# --- save/continuation consistency across the render context -----------------


def test_no_render_ctx_keeps_legacy_bytes():
    """An empty render context keeps the legacy byte-identical prefix/key: no
    churn for requests without render-affecting params."""
    assert hs.render_ctx_leader(None) == ""
    assert hs.render_ctx_leader({}) == ""
    legacy = hs.request_prefix_values([MSG], "m1", 100)
    assert legacy == hs.request_prefix_values([MSG], "m1", 100, render_ctx=None)
    assert legacy == hs.request_prefix_values([MSG], "m1", 100, render_ctx={})


def test_different_render_ctx_different_key():
    """Identical messages, different render context -> different prefix, key,
    blocks and prefix hashes (the conversation the template renders differs)."""
    a = hs.request_prefix_values([MSG], "m1", 100, render_ctx=CTX)
    b = hs.request_prefix_values([MSG], "m1", 100, render_ctx={**CTX, "reasoning_effort": "high"})
    assert a[0] != b[0]
    assert a[1] != b[1]
    assert a[2] != b[2]
    assert a[3] != b[3]


def test_same_render_ctx_deterministic_key():
    """The same context always hashes to the same key (canonical JSON)."""
    eq = hs.request_prefix_values([MSG], "m1", 100, render_ctx=CTX)
    assert eq == hs.request_prefix_values([MSG], "m1", 100, render_ctx=CTX)


def test_saved_continuation_matches_with_same_render_ctx():
    """The saved conversation (prompt + response) hashed under a render ctx
    equals the continuation request's key when the client echoes the same ctx."""
    saved = [MSG, {"role": "assistant", "content": "b"}]
    _p, _bl, saved_hashes = hs.saved_conversation_values(
        [MSG], "b", "m1", 100, render_ctx=CTX
    )
    _p2, key, _b2, req_hashes, _w = hs.request_prefix_values(
        saved, "m1", 100, render_ctx=CTX
    )
    assert saved_hashes[-1] == key
    assert req_hashes[-1] == key


def test_saved_continuation_never_matches_different_render_ctx():
    """A continuation with a different render ctx must NOT match the saved
    meta: the restored KV would not fit the prompt the template renders."""
    _p, _bl, saved_hashes = hs.saved_conversation_values(
        [MSG], "b", "m1", 100, render_ctx=CTX
    )
    _p2, key, _b2, _h2, _w = hs.request_prefix_values(
        [MSG, {"role": "assistant", "content": "b"}], "m1", 100, render_ctx=""
    )
    other = hs.request_prefix_values(
        [MSG, {"role": "assistant", "content": "b"}],
        "m1",
        100,
        render_ctx={**CTX, "reasoning_effort": "high"},
    )
    assert key != saved_hashes[-1]
    assert other[1] != saved_hashes[-1]