# chat_flow/_diagnostics.py

"""Per-request decision record, token metrics, slot snapshot and error mapping."""

import chat_flow
import hashing as hs
from backend.llama_client import LlamaClient
from core import promstats
from obs import reqlog

from . import _state

log = _state.log


def _emit_decision(decision: dict, rid: str, ts: str) -> None:
    """Persist the per-request cache decision (fire-and-forget, idempotent).

    The same decision dict is threaded through the request pipeline (restore
    phase in chat_flow, save outcome in the background save), so exactly one
    writer emits it — the task that finishes last, or the small non-stream
    path in chat_flow. The "_emitted" marker is never serialized.
    """
    if not rid or not ts or decision.pop("_emitted", False):
        return
    decision["_emitted"] = True
    reqlog.log_file(
        "decision",
        rid,
        ts,
        {k: v for k, v in decision.items() if k != "_emitted"},
    )


def _set_save_outcome(decision: dict | None, **kw: object) -> None:
    """Record the save result in the decision dict (no-op without one)."""
    if decision is not None:
        decision["save"] = {"attempted": True, **kw}


def _record_tokens(model_id: str, usage: dict | None) -> None:
    """Count backend-reported tokens (prompt/completion/cached) into metrics."""
    if not isinstance(usage, dict):
        return
    for key, kind in (
        ("prompt_tokens", "prompt"),
        ("completion_tokens", "completion"),
        ("prompt_cached_tokens", "cached"),
    ):
        v = usage.get(key)
        if isinstance(v, (int, float)) and v > 0:
            promstats.tokens_total.labels(model=model_id, kind=kind).inc(v)


async def _snapshot_slot(
    client: LlamaClient, slot_id: int, model_id: str
) -> dict | None:
    """KV state of the just-restored slot (GET /slots), or None if unavailable.

    The recorded fields are the llama.cpp /slots subset that distinguishes a
    restore that populated the KV cache (n_past == restored prefix tokens)
    from one the backend reset (n_past == 0): the key datapoint for "restore
    reported ok but the whole prompt is reprocessed" investigations. Best
    effort — a missing /slots endpoint just yields None.
    """
    try:
        slots = await client.get_slots(model=model_id)
    except Exception:  # noqa: BLE001
        log.debug("slot_snapshot_fail slot=%d model=%s", slot_id, model_id)
        return None
    if isinstance(slots, list):
        for s in slots:
            if isinstance(s, dict) and s.get("id") == slot_id:
                return {
                    k: s.get(k)
                    for k in ("state", "n_ctx", "n_past", "n_tokens", "total_tokens")
                    if k in s
                }
    return None


def _provider_error_status(status: object) -> int:
    """Map a provider failure to the HTTP status for the client.

    A backend 4xx is a client fault and passes through unchanged (so client
    5xx retry/alerting logic does not fire); everything else (backend 5xx,
    connect errors, non-JSON bodies, missing/malformed status) is a genuine
    upstream failure and maps to 502. Shared by the stream and non-stream
    dispatch paths so both stay consistent (M-8).
    """
    if isinstance(status, int) and 400 <= status < 500:
        return status
    return 502


def _new_decision(
    is_big: bool,
    no_cache: bool,
    n_words: int,
    model_id: str,
    render_ctx: dict,
) -> dict:
    # Per-request cache decision, threaded through the request pipeline and
    # finally persisted as decision.json by whichever task finishes last
    # (restore phase here, save outcome in the background save or the stream
    # reader). Skipped (no emit) when neither a handler path completes.
    return {
        "is_big": is_big,
        "no_cache": no_cache,
        "n_words": n_words,
        "words_threshold": chat_flow.BIG_THRESHOLD_WORDS,
        "model": model_id,
        "render_ctx_sha256": hs.render_ctx_digest(render_ctx) or None,
        "restore": {
            "candidate_key": None,
            "candidate_ratio": None,
            "used_key": None,
            "outcome": None,
            "stale_meta_dropped": False,
        },
        "wait_inflight_save": False,
        "erase_done": False,
        "slot": None,
        "slot_before_chat": None,
    }
