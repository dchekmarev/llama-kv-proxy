# tests/test_context_admission_chat.py

"""Chat-pipeline wiring of context-budget admission.

_chat_once derives a per-request token demand -- estimated prompt tokens plus
the generation budget (min(MAX_TOKENS, the client's max_tokens/n_predict)) --
and passes it to SlotManager.acquire_for_request as `need_tokens`. The estimate
is only taken while CONTEXT_ADMISSION_ENABLED, and the demand is recorded in
decision.json under the `admission` key.
"""

import asyncio

import pytest

import chat_flow
import hashing as hs
from core import promstats
from obs import reqlog


@pytest.fixture()
def reqlog_dir(tmp_path, monkeypatch):
    """Point reqlog at a fresh temp dir so tests do not write into the repo."""
    monkeypatch.setattr(reqlog, "REQUEST_LOG_DIR", str(tmp_path))
    monkeypatch.setattr(reqlog, "REQUEST_LOG_MAX_GROUPS", 100)
    return tmp_path


async def _drain(timeout=3.0):
    import time

    deadline = time.time() + timeout
    while time.time() < deadline:
        tasks = list(reqlog._LOG_TASKS) + list(chat_flow._BG_SAVE_TASKS)
        if not tasks:
            return
        await asyncio.sleep(0.01)
    raise AssertionError("log tasks did not finish in time")


async def _run(
    sm,
    monkeypatch,
    data,
    *,
    est: int = 100,
    max_tokens: int = 1000,
    enabled: bool = True,
):
    """Drive one chat request, spying on the estimate and slot admission.

    Returns (response, captured_need_tokens, decision, estimate_calls).
    """
    monkeypatch.setattr(chat_flow, "MAX_TOKENS", max_tokens)
    monkeypatch.setattr(chat_flow, "CONTEXT_ADMISSION_ENABLED", enabled)

    est_calls: list[str] = []

    async def fake_est(text: str) -> int:
        est_calls.append(text)
        return est

    monkeypatch.setattr(hs, "estimate_prompt_tokens_async", fake_est)

    captured: dict = {}
    real_acquire = sm.acquire_for_request

    async def spy(model, restore_key=None, resolve_restore_key=None, need_tokens=None):
        captured["need_tokens"] = need_tokens
        return await real_acquire(model, restore_key, resolve_restore_key, need_tokens)

    monkeypatch.setattr(sm, "acquire_for_request", spy)

    emitted: dict = {}

    def spy_emit(decision, rid, ts):
        emitted["decision"] = dict(decision)

    monkeypatch.setattr(chat_flow, "_emit_decision", spy_emit)

    client = sm.backends[0]["client"]
    resp = await chat_flow.chat_flow(sm, [client], data)
    await _drain()
    return resp, captured["need_tokens"], emitted["decision"], est_calls


def _small(extra: dict | None = None) -> dict:
    data = {"messages": [{"role": "user", "content": "hi"}]}
    data.update(extra or {})
    return data


# --- need_tokens derivation --------------------------------------------------


async def test_max_tokens_bounds_the_generation_budget(sm, monkeypatch, meta_dir, reqlog_dir):
    resp, need, _, est_calls = await _run(
        sm, monkeypatch, _small({"max_tokens": 500})
    )
    assert resp.status_code == 200
    assert need == 600, "est_prompt(100) + min(MAX_TOKENS, 500)"
    assert len(est_calls) == 1


async def test_no_client_cap_uses_max_tokens(sm, monkeypatch, meta_dir, reqlog_dir):
    resp, need, _, _ = await _run(sm, monkeypatch, _small())
    assert resp.status_code == 200
    assert need == 1100, "est_prompt(100) + MAX_TOKENS(1000)"


async def test_client_cap_above_max_tokens_is_clamped(sm, monkeypatch, meta_dir, reqlog_dir):
    resp, need, _, _ = await _run(
        sm, monkeypatch, _small({"max_tokens": 9999})
    )
    assert resp.status_code == 200
    assert need == 1100, "est_prompt(100) + MAX_TOKENS(1000)"


async def test_n_predict_is_honored(sm, monkeypatch, meta_dir, reqlog_dir):
    resp, need, _, _ = await _run(sm, monkeypatch, _small({"n_predict": 50}))
    assert resp.status_code == 200
    assert need == 150, "est_prompt(100) + n_predict(50)"


async def test_max_tokens_preferred_over_n_predict(sm, monkeypatch, meta_dir, reqlog_dir):
    resp, need, _, _ = await _run(
        sm, monkeypatch, _small({"max_tokens": 500, "n_predict": 50})
    )
    assert resp.status_code == 200
    assert need == 600, "max_tokens wins when both are present"


@pytest.mark.parametrize("bad", ["x", True, False, 0, -5, 1.5, None, [], {}])
async def test_non_int_or_non_positive_cap_is_ignored(
    sm, monkeypatch, meta_dir, reqlog_dir, bad
):
    resp, need, _, _ = await _run(sm, monkeypatch, _small({"max_tokens": bad}))
    assert resp.status_code == 200
    assert need == 1100, f"max_tokens={bad!r} must be ignored"


# --- the enable switch -------------------------------------------------------


async def test_disabled_skips_estimate_and_sends_none(
    sm, monkeypatch, meta_dir, reqlog_dir
):
    resp, need, decision, est_calls = await _run(
        sm, monkeypatch, _small(), enabled=False
    )
    assert resp.status_code == 200
    assert need is None, "admission disabled => ungated acquire"
    assert est_calls == [], "no tokenizer cost when the feature is off"
    assert decision["admission"]["enabled"] is False
    assert decision["admission"]["est_prompt_tokens"] is None
    assert decision["admission"]["need_tokens"] is None
    assert decision["admission"]["gen_budget"] == 1000


# --- diagnostics -------------------------------------------------------------


async def test_decision_carries_the_admission_block(sm, monkeypatch, meta_dir, reqlog_dir):
    resp, need, decision, _ = await _run(
        sm, monkeypatch, _small({"max_tokens": 500})
    )
    assert resp.status_code == 200
    assert decision["admission"] == {
        "enabled": True,
        "est_prompt_tokens": 100,
        "gen_budget": 500,
        "need_tokens": need,
    }


async def test_estimate_is_observed_in_the_histogram(sm, monkeypatch, meta_dir, reqlog_dir):
    resp, _, _, _ = await _run(sm, monkeypatch, _small())
    assert resp.status_code == 200
    count = promstats.REGISTRY.get_sample_value(
        "llama_kv_proxy_estimated_prompt_tokens_count", {"model": "m1"}
    )
    total = promstats.REGISTRY.get_sample_value(
        "llama_kv_proxy_estimated_prompt_tokens_sum", {"model": "m1"}
    )
    assert count == 1.0
    assert total == 100.0
