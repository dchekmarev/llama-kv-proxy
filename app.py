# app.py

"""The proxy's HTTP surface: FastAPI app, routes, lifespan and background jobs.

Route map (the /proxy prefix keeps the native llama.cpp paths free for the
pass-through: /slots, /health, /tokenize, /completion, ... reach the backend):

- POST /v1/chat/completions  the cached chat pipeline (chat_flow)
- GET  /v1/models           the union of the backends' model lists
- GET  /proxy/health        per-backend probe + slot state
- GET  /proxy/slots         the discovered slot pools
- GET  /cache/stats         meta cache files/bytes/hits/misses
- GET,POST /cache/clear     drop the whole cache (metas + backend .bin)
- GET  /metrics             Prometheus text (proxy registry + backends)
- GET  /version             build identifier
- GET  /proxy/ui[/]         the live dashboard (page, state, request, events)
- ANY  /{path:path}         pass-through to the first backend

Everything blocking (meta scans, .bin cleanup) runs in a worker thread so the
event loop keeps serving requests, and every background job is a task that is
cancelled on shutdown.
"""

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable
from contextlib import asynccontextmanager, suppress
from typing import Any, cast

from fastapi import FastAPI, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

import bin_cache
import chat_flow
import hashing
import metrics
import promstats
import ui as ui_obs
import ui_page
import version as version_info
from config import (
    BACKENDS,
    BIN_CACHE_DIR,
    BIN_CACHE_MAX_MB,
    BIN_RECONCILE_INTERVAL_S,
    EVICT_INTERVAL_S,
    LOG_LEVEL,
    META_INDEX_ENABLED,
    META_INDEX_RECONCILE_INTERVAL_S,
    META_MAX_FILES,
    META_MAX_MB,
    META_TTL_H,
    MODEL_ID,
    SLOT_POLL_INTERVAL_S,
    STUCK_SLOT_THRESHOLD_S,
    UI_ENABLED,
    setup_logging,
)
from llama_client import LlamaClient
from request_id import new_request_id, request_id_var
from slot_manager import SlotManager

log = logging.getLogger(__name__)

# Hop-by-hop headers (RFC 9110 7.6.1) describe a single connection, and
# httpx recomputes the framing ones, so a stale client value would conflict.
_HOP_BY_HOP_HEADERS = frozenset(
    {"host", "content-length", "transfer-encoding", "connection"}
)

_PROMETHEUS_MEDIA_TYPE = "text/plain; version=0.0.4; charset=utf-8"

# First poll at which a slot was seen busy: (backend index, model, slot id) ->
# time.time(). The watchdog erases a slot that stays busy too long.
_stuck_slot_first_busy: dict[tuple[int, str, int], float] = {}

# Periodic meta-index<->disk reconcile task. A module global (not a lifespan
# local) so shutdown and tests can see it; None when the interval is 0.
_meta_index_reconcile_task: "asyncio.Task[None] | None" = None


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    """Build the backend clients, start the background jobs, close on exit."""
    global _meta_index_reconcile_task

    setup_logging(LOG_LEVEL)
    clients: list[LlamaClient] = [LlamaClient(be["url"]) for be in BACKENDS]
    sm = SlotManager()
    sm.set_clients(clients)
    application.state.clients = clients
    application.state.sm = sm

    if META_INDEX_ENABLED:
        indexed = await hashing.rebuild_index_async()
        log.info("meta_index_rebuilt metas=%d", indexed)

    tasks = [
        asyncio.create_task(_poll_slots_loop()),
        asyncio.create_task(_eviction_loop()),
        asyncio.create_task(_bin_reconcile_loop()),
    ]
    # The reconcile only exists to maintain the in-RAM index, so it needs both
    # the index and a non-zero interval.
    if META_INDEX_ENABLED and META_INDEX_RECONCILE_INTERVAL_S > 0:
        _meta_index_reconcile_task = asyncio.create_task(
            _meta_index_reconcile_loop()
        )
        tasks.append(_meta_index_reconcile_task)
    else:
        _meta_index_reconcile_task = None
    try:
        yield
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        _meta_index_reconcile_task = None
        for client in clients:
            with suppress(Exception):
                await client.close()
        log.info("shutdown_complete")


app = FastAPI(title="llama-kv-proxy", lifespan=lifespan)


@app.middleware("http")
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


# --- /v1/chat/completions ----------------------------------------------------


def _validate_body(data: dict[str, Any]) -> str | None:
    """Reject a badly typed body, or None when it is well-formed.

    Only the two fields the pipeline reads structurally are checked here: a
    wrong type must become a 400 with a readable message, not a 500 from deep
    inside hashing. The message always names the offending field.
    """
    if "messages" in data:
        messages = data["messages"]
        if not isinstance(messages, list) or not all(
            isinstance(m, dict) for m in messages
        ):
            return "'messages' must be a list of message objects"
    model = data.get("model")
    if model is not None and not isinstance(model, str):
        return "'model' must be a string"
    return None


@app.post("/v1/chat/completions")
async def chat(request: Request) -> Response:
    """Parse the body, validate it, then hand the request to chat_flow."""
    try:
        data = await request.json()
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"error": f"invalid JSON body: {e}"}, status_code=400)
    if not isinstance(data, dict):
        return JSONResponse(
            {"error": "request body must be a JSON object"}, status_code=400
        )
    error = _validate_body(data)
    if error is not None:
        return JSONResponse({"error": error}, status_code=400)
    return await chat_flow.chat_flow(app.state.sm, app.state.clients, data)


# --- models, health, slots ----------------------------------------------------


async def _fetch_models(client: Any) -> list[dict[str, Any]] | None:
    """One backend's /v1/models listing; None when it is down or unusable."""
    try:
        return await client.get_models()
    except Exception as e:  # noqa: BLE001
        log.warning("models_fetch_fail url=%s: %s", getattr(client, "url", "?"), e)
        return None


@app.get("/v1/models")
async def models() -> dict[str, Any]:
    """The union of the backends' model lists (deduped, first-seen order).

    A client needs something to ask for even while every backend is restarting,
    so the configured MODEL_ID is advertised when no backend answers.
    """
    listings = await asyncio.gather(
        *(_fetch_models(client) for client in app.state.clients)
    )
    data: list[dict[str, Any]] = []
    seen: set[str] = set()
    for listing in listings:
        for entry in listing or []:
            if not isinstance(entry, dict):
                continue
            model_id = entry.get("id")
            if not isinstance(model_id, str) or model_id in seen:
                continue
            seen.add(model_id)
            data.append(entry)
    return {"data": data or [{"id": MODEL_ID}]}


async def _probe(client: Any) -> dict[str, Any]:
    """One backend's health probe; a failure is reported, never raised."""
    try:
        return await client.health()
    except Exception as e:  # noqa: BLE001
        log.warning("health_probe_fail url=%s: %s", getattr(client, "url", "?"), e)
        return {"ok": False, "model_id": None, "url": getattr(client, "url", None)}


@app.get("/proxy/health")
async def health() -> dict[str, Any]:
    """Every backend's probe (concurrently) plus the current slot state."""
    backends = list(
        await asyncio.gather(*(_probe(client) for client in app.state.clients))
    )
    return {
        "ok": all(backend.get("ok") for backend in backends),
        "version": version_info.__version__,
        "backends": backends,
        "slots": app.state.sm.aggregated_state(),
    }


@app.get("/proxy/slots")
async def slots_state() -> dict[str, Any]:
    """The discovered slot pools as reported by the slot manager."""
    return {"slots": app.state.sm.aggregated_state()}


# --- cache administration -----------------------------------------------------


async def _key_model_pairs() -> dict[str, str | None]:
    """{meta key: model id} from a full meta scan, run off the event loop."""
    metas = await asyncio.to_thread(hashing.scan_all_meta)
    pairs: dict[str, str | None] = {}
    for meta in metas or []:
        if not isinstance(meta, dict):
            continue
        key = meta.get("key")
        if isinstance(key, str) and key:
            pairs[key] = meta.get("model_id")
    return pairs


async def _delete_backend_caches(
    clients: list[Any], key_models: list[tuple[str, str | None]]
) -> None:
    """Best-effort .bin removal on every backend for the given cache keys.

    llama.cpp has no endpoint to delete a file from --slot-save-path, so this is
    only a fallback for backends whose cache dir is not mounted into the proxy;
    failures are expected and ignored. The model is passed on because a router
    backend needs it to route the delete to the right child.
    """
    for key, model_id in key_models:
        for client in clients:
            try:
                await client.delete_cache_file(key, model=model_id)
            except Exception as e:  # noqa: BLE001
                log.warning("delete_cache_file_fail key=%s: %s", key, e)


@app.get("/cache/stats")
async def cache_stats() -> dict[str, Any]:
    """Meta cache files/bytes and the hit/miss counters."""
    return await asyncio.to_thread(hashing.cache_stats)


@app.api_route("/cache/clear", methods=["GET", "POST"])
async def cache_clear() -> dict[str, Any]:
    """Drop the whole cache: meta files, backend .bin files, and counters.

    The key -> model map is read first: once the metas are gone the model of a
    deleted key is unknown, and a router backend needs it to route the delete.
    """
    key_models = await _key_model_pairs()
    keys = await hashing.clear_all_meta_async()
    await _delete_backend_caches(
        app.state.clients, [(key, key_models.get(key)) for key in keys]
    )
    if BIN_CACHE_DIR:
        # Every .bin is an orphan now: no meta references it.
        await asyncio.to_thread(bin_cache.clear_bin_cache, BIN_CACHE_DIR)
    promstats.reset()
    await asyncio.to_thread(promstats.refresh_storage_gauges)
    return {"deleted": len(keys)}


@app.get("/metrics")
async def metrics_endpoint(model: str | None = None) -> Response:
    """Prometheus text: the proxy's own registry, then every backend's.

    `model` filters both halves (proxy samples by their model label, backend
    samples by the model whose /metrics was scraped).
    """
    proxy_text = await asyncio.to_thread(promstats.render, model)
    backend_text = await metrics.collect(app.state.clients, model)
    return Response(
        content=proxy_text + backend_text, media_type=_PROMETHEUS_MEDIA_TYPE
    )


@app.get("/version")
async def version() -> dict[str, str]:
    """Build identifier, read from the version module at call time."""
    return {"name": "llama-kv-proxy", "version": version_info.__version__}


# --- live dashboard (/proxy/ui) ----------------------------------------------


def _ui_not_found() -> Response:
    """The dashboard routes do not exist when UI_ENABLED is off."""
    return JSONResponse({"error": "not found"}, status_code=404)


def _ui_slots() -> list[dict[str, Any]]:
    """Slot rows annotated with the request id currently holding each slot."""
    holders = {
        (info.slot["backend"], info.slot["model"], info.slot["id"]): info.rid
        for info in ui_obs.registry.active.values()
        if info.slot is not None
    }
    rows: list[dict[str, Any]] = []
    for row in app.state.sm.aggregated_state():
        key = (row.get("backend"), row.get("model"), row.get("slot"))
        rows.append({**row, "busy_rid": holders.get(key)})
    return rows


def _ui_snapshot() -> dict[str, Any]:
    """The whole dashboard state: in-flight requests, history, slot grid."""
    return {**ui_obs.registry.snapshot(), "slots": _ui_slots()}


def _sse(payload: dict[str, Any]) -> str:
    """One server-sent event carrying a JSON payload."""
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


async def _ui_event_stream() -> AsyncIterator[str]:
    """SSE frames: the current snapshot, then every pushed token batch."""
    queue = ui_obs.registry.subscribe()
    try:
        yield _sse({"type": "snapshot", **_ui_snapshot()})
        while True:
            for event in await queue.get():
                yield _sse(event)
    finally:
        ui_obs.registry.unsubscribe(queue)


@app.get("/proxy/ui", include_in_schema=False)
@app.get("/proxy/ui/", include_in_schema=False)
async def ui_page_route() -> Response:
    """The dashboard page (one self-contained HTML document)."""
    if not UI_ENABLED:
        return _ui_not_found()
    return HTMLResponse(ui_page.PAGE)


@app.get("/proxy/ui/state", include_in_schema=False)
async def ui_state() -> Response:
    """Dashboard poll target: active requests, history and the slot grid."""
    if not UI_ENABLED:
        return _ui_not_found()
    return JSONResponse(_ui_snapshot())


@app.get("/proxy/ui/request/{rid}", include_in_schema=False)
async def ui_request(rid: str) -> Response:
    """Full prompt and response tail of one request (in flight or finished)."""
    if not UI_ENABLED:
        return _ui_not_found()
    prompt = ui_obs.registry.full_prompt(rid)
    tail = ui_obs.registry.response_tail(rid)
    if prompt is None or tail is None:
        return JSONResponse({"error": "unknown request id"}, status_code=404)
    content, reasoning = tail
    return JSONResponse(
        {
            "rid": rid,
            "prompt": prompt,
            "response": content,
            "response_reasoning": reasoning,
        }
    )


@app.get("/proxy/ui/events", include_in_schema=False)
async def ui_events() -> Response:
    """The live token stream of every in-flight request (SSE)."""
    if not UI_ENABLED:
        return _ui_not_found()
    return StreamingResponse(
        _ui_event_stream(),
        media_type="text/event-stream",
        # The dashboard must not see a cached or proxied-buffered stream.
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# --- pass-through -------------------------------------------------------------


def _header_items(headers: Any) -> Iterable[tuple[str, str]]:
    """(name, value) pairs of a header container.

    Starlette/httpx header mappings iterate over names only, while the raw
    wire format is a sequence of pairs; both shapes are accepted.
    """
    items = getattr(headers, "items", None)
    if callable(items):
        return cast("Iterable[tuple[str, str]]", items())
    return cast("Iterable[tuple[str, str]]", headers)


def _forwarded_headers(headers: Any) -> dict[str, str]:
    """Headers to forward, minus the ones that describe this hop only."""
    return {
        name: value
        for name, value in _header_items(headers)
        if name.lower() not in _HOP_BY_HOP_HEADERS
    }


async def _iter_upstream(upstream: Any) -> AsyncIterator[bytes]:
    """Yield the upstream body chunks in order, then close the response."""
    try:
        async for chunk in upstream.aiter_bytes():
            yield chunk
    finally:
        await upstream.aclose()


@app.api_route(
    "/{path:path}",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"],
    include_in_schema=False,
)
async def passthrough(path: str, request: Request) -> Response:
    """Forward an unhandled path to the first backend, streaming the response.

    The first backend only, consistently with /v1/models: the pass-through
    serves the native llama.cpp endpoints (slots, tokenize, completion, ...),
    which are single-server APIs. The upstream status is mirrored and the body
    is never buffered.
    """
    client = app.state.clients[0]
    built = client.client.build_request(
        request.method,
        "/" + path,
        params=request.url.query,
        content=await request.body(),
        headers=_forwarded_headers(request.headers),
    )
    upstream = await client.client.send(built, stream=True)
    return StreamingResponse(
        _iter_upstream(upstream),
        status_code=upstream.status_code,
        headers=_forwarded_headers(upstream.headers),
    )


# The catch-all is declared last on purpose: it matches every path, so it
# must be tried only after the proxy's own routes have been matched.

# --- background jobs ----------------------------------------------------------


async def _check_stuck_slots(
    backend_index: int, model: str, client: Any, slots: list[dict]
) -> None:
    """Erase slots that report is_processing for longer than the threshold.

    A wedged slot (llama.cpp stuck in PROCESSING_PROMPT) never finishes and the
    backend busy-loops on it; erasing its KV recovers the slot without a
    restart. The /slots payload has no per-slot timestamp, so the proxy tracks
    how long it has seen each slot busy. A threshold of 0 disables the
    watchdog, and the timer is re-armed after an erase so one wedged slot is not
    erased on every poll.
    """
    threshold = STUCK_SLOT_THRESHOLD_S
    if threshold <= 0:
        return
    now = time.time()
    for slot in slots:
        slot_id = cast(int, slot.get("id"))
        key = (backend_index, model, slot_id)
        if not slot.get("is_processing"):
            _stuck_slot_first_busy.pop(key, None)
            continue
        first_busy = _stuck_slot_first_busy.get(key)
        if first_busy is None:
            _stuck_slot_first_busy[key] = now
        elif now - first_busy >= threshold:
            _stuck_slot_first_busy[key] = now
            try:
                await client.erase_slot(slot_id, model=model)
            except Exception as e:  # noqa: BLE001
                log.warning("stuck_slot_erase_fail slot=%s: %s", key, e)
            else:
                promstats.stuck_slot_erases_total.labels(
                    backend=str(backend_index), model=model
                ).inc()
                log.warning(
                    "stuck_slot_erased slot=%s busy_for=%.0fs",
                    key,
                    now - first_busy,
                )


def _publish_slots_gauge(backend_index: int, model: str, slots: list[dict]) -> None:
    """Publish slots_total{backend,model,state} for one polled model.

    States that disappeared fall back to 0: a gauge is only as current as its
    last update, and a slot that was cut must not keep reporting forever.
    """
    counts: dict[str, int] = {}
    for slot in slots:
        state = str(slot.get("state") or "unknown")
        counts[state] = counts.get(state, 0) + 1
    for state in {*counts, "free", "busy"}:
        promstats.slots_total.labels(
            backend=str(backend_index), model=model, state=state
        ).set(counts.get(state, 0))


async def _poll_backend(backend_index: int, client: Any) -> None:
    """Refresh one backend's slot pools and run the stuck-slot watchdog.

    A failed or unresolvable discovery keeps the previous pool: dropping it
    would send every in-flight request onto the bootstrap slot and, with a
    literal "unknown" model key, merge all models into one cache namespace.
    """
    backend = str(backend_index)
    try:
        is_router = await client.is_router()
        if is_router:
            models = await client.get_active_models()
        else:
            model = await client.get_model_id()
            if model == "unknown":
                log.warning("poll_model_unknown backend=%d", backend_index)
                models = []
            else:
                models = [model]
        for model in models:
            if not model:
                continue
            slots = (
                await client.get_slots(model=model)
                if is_router
                else await client.get_slots()
            )
            if slots is None:
                continue
            app.state.sm.set_backend_slots(backend_index, model, slots)
            _publish_slots_gauge(backend_index, model, slots)
            await _check_stuck_slots(backend_index, model, client, slots)
        promstats.backend_up.labels(backend=backend).set(1)
    except Exception as e:  # noqa: BLE001
        promstats.backend_up.labels(backend=backend).set(0)
        log.warning("poll_slots_failed backend=%s: %s", backend, e)


async def _poll_slots() -> None:
    """One discovery pass over every backend (GET /slots)."""
    for backend_index, client in enumerate(app.state.clients):
        await _poll_backend(backend_index, client)


async def _poll_slots_loop() -> None:
    """Refresh the slot pools every SLOT_POLL_INTERVAL_S (0 disables)."""
    if SLOT_POLL_INTERVAL_S <= 0:
        return
    while True:
        try:
            await _poll_slots()
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("poll_slots_loop_failed")
        await asyncio.sleep(SLOT_POLL_INTERVAL_S)


async def _run_eviction() -> None:
    """One cache-maintenance pass: meta caps, backend purge, LRU, gauges.

    The key -> model map is read before the eviction, because a deleted meta can
    no longer tell which model its .bin belongs to (a router needs it to route
    the delete to the right child).
    """
    key_models = await _key_model_pairs()
    result = await hashing.evict_meta_async(
        ttl_hours=META_TTL_H, max_files=META_MAX_FILES, max_mb=META_MAX_MB
    )
    deleted = list(result.get("deleted") or [])
    if deleted:
        await _delete_backend_caches(
            app.state.clients, [(key, key_models.get(key)) for key in deleted]
        )
    if BIN_CACHE_DIR:
        await asyncio.to_thread(
            bin_cache.clean_bin_cache, BIN_CACHE_DIR, BIN_CACHE_MAX_MB
        )
    await asyncio.to_thread(promstats.refresh_storage_gauges)
    log.info(
        "eviction_run deleted=%d remaining=%s", len(deleted), result.get("remaining")
    )


async def _eviction_loop() -> None:
    """Run the eviction pass every EVICT_INTERVAL_S (0 disables)."""
    if EVICT_INTERVAL_S <= 0:
        return
    while True:
        try:
            await _run_eviction()
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("eviction_failed")
        await asyncio.sleep(EVICT_INTERVAL_S)


async def _bin_reconcile_loop() -> None:
    """Reconcile metas and .bin files in both directions, every interval.

    Drops a meta whose .bin is gone (the restore can never succeed) and a .bin
    whose meta is gone (nothing references it). Runs only when the .bin cache
    directory is mounted.
    """
    if BIN_RECONCILE_INTERVAL_S <= 0:
        return
    while True:
        await asyncio.sleep(BIN_RECONCILE_INTERVAL_S)
        if not BIN_CACHE_DIR:
            continue
        try:
            result = await asyncio.to_thread(
                bin_cache.reconcile_bin_cache, BIN_CACHE_DIR
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("bin_reconcile_failed")
            continue
        if result["deleted_metas"] or result["deleted_bins"]:
            log.info(
                "bin_reconcile deleted_metas=%d deleted_bins=%d",
                len(result["deleted_metas"]),
                len(result["deleted_bins"]),
            )


async def _meta_index_reconcile_loop() -> None:
    """Drop index entries whose meta file vanished, every interval.

    The index is kept live on proxy-driven writes and deletes, but external
    deletions (bin_cache reconcile/cleanup, an operator) leave ghost entries
    behind. The lookup resolves late: hashing.reconcile_index_async is read
    from the module on every round.
    """
    while True:
        await asyncio.sleep(META_INDEX_RECONCILE_INTERVAL_S)
        try:
            dropped = await hashing.reconcile_index_async()
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("meta_index_reconcile_failed")
            continue
        if dropped:
            log.info("meta_index_reconciled dropped=%d", len(dropped))
