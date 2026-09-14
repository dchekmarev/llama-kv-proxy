# tests/test_erase_before_chat.py

"""Prevention: a chat that was NOT preceded by a successful restore erases the
slot first, so it does not start on top of a stale or oversized prompt (which
can wedge llama.cpp in PROCESSING_PROMPT and busy-loop the server). A successful
restore already set the slot's prompt to the correct prefix, so it is skipped.
Controlled by ERASE_BEFORE_CHAT (on by default).

The erase decision lives inline in chat_flow, so these tests drive the real
pipeline with a mocked acquire (controlling the `restored` outcome) and a
non-stream request, then assert whether erase_slot was issued.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

import chat_flow
from llama_client import RESTORE_MISSING
from slot_manager import GSlot, SlotManager

G: GSlot = (0, "m1", 0)


@pytest.fixture()
def sm(monkeypatch):
    import slot_manager as sm_module

    monkeypatch.setattr(sm_module, "BACKENDS", [{"url": "http://be", "n_slots": 2}])
    manager = SlotManager()
    client = MagicMock()
    client.erase_slot = AsyncMock(return_value=True)
    client.save_slot = AsyncMock(return_value=True)
    client.restore_slot = AsyncMock(return_value=True)
    client.get_model_id_cached = AsyncMock(return_value="m1")
    client.get_loaded_model = AsyncMock(return_value="m1")
    client.chat_completions = AsyncMock(return_value={"choices": []})
    manager.set_clients([client])
    # Control the restore outcome directly instead of driving the hashing /
    # restore-candidate machinery: acquire returns a fixed (g, lock, restored).
    manager.acquire_for_request = AsyncMock(
        return_value=(G, asyncio.Lock(), None)
    )
    return manager


async def _run(sm, monkeypatch, restored, flag: bool):
    client = sm.backends[0]["client"]
    sm.acquire_for_request.return_value = (G, asyncio.Lock(), restored)
    monkeypatch.setattr(chat_flow, "ERASE_BEFORE_CHAT", flag)
    data = {"messages": [{"role": "user", "content": "hi"}], "stream": False}
    await chat_flow.chat_flow(sm, [client], data)
    return client


@pytest.mark.asyncio
async def test_no_restore_erases_when_flag_on(sm, monkeypatch):
    # The main bug fix: a non-restore chat (restored=None) clears the slot.
    client = await _run(sm, monkeypatch, restored=None, flag=True)
    client.erase_slot.assert_awaited_once()


@pytest.mark.asyncio
async def test_no_restore_keeps_slot_when_flag_off(sm, monkeypatch):
    client = await _run(sm, monkeypatch, restored=None, flag=False)
    client.erase_slot.assert_not_awaited()


@pytest.mark.asyncio
async def test_restore_hit_skips_erase(sm, monkeypatch):
    # A successful restore already set the prefix: do not erase it away.
    client = await _run(sm, monkeypatch, restored=True, flag=True)
    client.erase_slot.assert_not_awaited()


@pytest.mark.asyncio
async def test_restore_missing_erases(sm, monkeypatch):
    # The backend reported the cache file gone: the slot was not set up, so
    # clear any stale prompt before the chat.
    client = await _run(sm, monkeypatch, restored=RESTORE_MISSING, flag=True)
    client.erase_slot.assert_awaited_once()


@pytest.mark.asyncio
async def test_restore_failed_erases(sm, monkeypatch):
    # A transient restore failure: the slot was not set up, so clear it.
    client = await _run(sm, monkeypatch, restored=False, flag=True)
    client.erase_slot.assert_awaited_once()
