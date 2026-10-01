# chat_flow/_lru.py

"""Backend .bin purging and the post-save LRU cleanup check."""

import asyncio

import chat_flow
from backend.llama_client import LlamaClient
from cache import bin_cache

from . import _state

log = _state.log


async def _purge_backend_files(
    clients: list[LlamaClient], key_models: list[tuple[str, str | None]]
) -> None:
    # Backend .bin files are purged best-effort. llama.cpp intentionally has
    # no endpoint to delete files from --slot-save-path (the `erase` slot
    # action only clears in-memory state), so when the save directory is
    # mounted we remove the .bin files directly; the DELETE call stays as a
    # fallback for plain backends without a mount. A router backend needs the
    # model to route the delete to the right child.
    for key, model_id in key_models:
        for client in clients:
            await client.delete_cache_file(key, model=model_id)
        if chat_flow.BIN_CACHE_DIR:
            await asyncio.to_thread(bin_cache.delete_bin_file, chat_flow.BIN_CACHE_DIR, key)


def _schedule_lru_check() -> None:
    """Fire-and-forget LRU check after a slot write (save).

    Must not delay the response: the cleanup runs in a background task.
    Skipped when the .bin cache is disabled or a check is already in
    flight (concurrent runs would race on the same files). Call it only
    after the meta is written: a .bin without a meta looks orphaned and
    would be deleted by the very check it triggered.
    """
    if not chat_flow.BIN_CACHE_DIR or chat_flow.BIN_CACHE_MAX_MB <= 0 or chat_flow._lru_check_in_flight:
        return
    chat_flow._lru_check_in_flight = True

    async def _run() -> None:
        try:
            await asyncio.to_thread(
                bin_cache.clean_bin_cache, chat_flow.BIN_CACHE_DIR, chat_flow.BIN_CACHE_MAX_MB
            )
        except Exception as e:  # noqa: BLE001
            log.warning("lru_check_error: %s", e)
        finally:
            chat_flow._lru_check_in_flight = False

    task = asyncio.create_task(_run())
    chat_flow._LRU_TASKS.add(task)
    task.add_done_callback(chat_flow._LRU_TASKS.discard)
