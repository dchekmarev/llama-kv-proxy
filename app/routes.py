# app/routes.py

"""The HTTP endpoints: the chat pipeline, the model list, the health and slot
probes, the cache administration, the metrics, the live dashboard and the
pass-through.

The catch-all pass-through is declared last on purpose: it matches every path,
so it must be tried only after the proxy's own routes have been matched.
"""

import asyncio
import json
import logging
from collections.abc import AsyncIterator, Iterable
from typing import Any, cast

from fastapi import Request, Response
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

import app as app_pkg
import chat_flow
import hashing
from cache import bin_cache
from core import promstats
from core import version as version_info
from obs import metrics, ui_page
from obs import ui as ui_obs

log = logging.getLogger(__name__)

# Hop-by-hop headers (RFC 9110 7.6.1) describe a single connection, and
# httpx recomputes the framing ones, so a stale client value would conflict.
_HOP_BY_HOP_HEADERS = frozenset(
    {"host", "content-length", "transfer-encoding", "connection"}
)

_PROMETHEUS_MEDIA_TYPE = "text/plain; version=0.0.4; charset=utf-8"

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


@app_pkg.app.post("/v1/chat/completions")
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
    return await chat_flow.chat_flow(
        app_pkg.app.state.sm, app_pkg.app.state.clients, data
    )


# --- models, health, slots ----------------------------------------------------


async def _fetch_models(client: Any) -> list[dict[str, Any]] | None:
    """One backend's /v1/models listing; None when it is down or unusable."""
    try:
        return await client.get_models()
    except Exception as e:  # noqa: BLE001
        log.warning("models_fetch_fail url=%s: %s", getattr(client, "url", "?"), e)
        return None


@app_pkg.app.get("/v1/models")
async def models() -> dict[str, Any]:
    """The union of the backends' model lists (deduped, first-seen order).

    A client needs something to ask for even while every backend is restarting,
    so the configured MODEL_ID is advertised when no backend answers.
    """
    listings = await asyncio.gather(
        *(_fetch_models(client) for client in app_pkg.app.state.clients)
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
    return {"data": data or [{"id": app_pkg.MODEL_ID}]}


async def _probe(client: Any) -> dict[str, Any]:
    """One backend's health probe; a failure is reported, never raised."""
    try:
        return await client.health()
    except Exception as e:  # noqa: BLE001
        log.warning("health_probe_fail url=%s: %s", getattr(client, "url", "?"), e)
        return {"ok": False, "model_id": None, "url": getattr(client, "url", None)}


@app_pkg.app.get("/proxy/health")
async def health() -> dict[str, Any]:
    """Every backend's probe (concurrently) plus the current slot state."""
    backends = list(
        await asyncio.gather(
            *(_probe(client) for client in app_pkg.app.state.clients)
        )
    )
    return {
        "ok": all(backend.get("ok") for backend in backends),
        "version": version_info.__version__,
        "backends": backends,
        "slots": app_pkg.app.state.sm.aggregated_state(),
    }


@app_pkg.app.get("/proxy/slots")
async def slots_state() -> dict[str, Any]:
    """The discovered slot pools as reported by the slot manager."""
    return {"slots": app_pkg.app.state.sm.aggregated_state()}


# --- cache administration -----------------------------------------------------


@app_pkg.app.get("/cache/stats")
async def cache_stats() -> dict[str, Any]:
    """Meta cache files/bytes and the hit/miss counters."""
    return await asyncio.to_thread(hashing.cache_stats)


@app_pkg.app.post("/cache/clear")
async def cache_clear() -> dict[str, Any]:
    """Drop the whole cache: meta files, backend .bin files, and counters.

    The key -> model map is read first: once the metas are gone the model of a
    deleted key is unknown, and a router backend needs it to route the delete.
    """
    key_models = await app_pkg._key_model_pairs()
    keys = await hashing.clear_all_meta_async()
    await app_pkg._delete_backend_caches(
        app_pkg.app.state.clients, [(key, key_models.get(key)) for key in keys]
    )
    if app_pkg.BIN_CACHE_DIR:
        # Every .bin is an orphan now: no meta references it.
        await asyncio.to_thread(
            bin_cache.clear_bin_cache, app_pkg.BIN_CACHE_DIR
        )
    promstats.reset()
    await asyncio.to_thread(promstats.refresh_storage_gauges)
    return {"deleted": len(keys)}


@app_pkg.app.get("/metrics")
async def metrics_endpoint(model: str | None = None) -> Response:
    """Prometheus text: the proxy's own registry, then every backend's.

    `model` filters both halves (proxy samples by their model label, backend
    samples by the model whose /metrics was scraped).
    """
    proxy_text = await asyncio.to_thread(promstats.render, model)
    backend_text = await metrics.collect(app_pkg.app.state.clients, model)
    return Response(
        content=proxy_text + backend_text, media_type=_PROMETHEUS_MEDIA_TYPE
    )


@app_pkg.app.get("/version")
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
    for row in app_pkg.app.state.sm.aggregated_state():
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


@app_pkg.app.get("/proxy/ui", include_in_schema=False)
@app_pkg.app.get("/proxy/ui/", include_in_schema=False)
async def ui_page_route() -> Response:
    """The dashboard page (one self-contained HTML document)."""
    if not app_pkg.UI_ENABLED:
        return _ui_not_found()
    return HTMLResponse(ui_page.PAGE)


@app_pkg.app.get("/proxy/ui/state", include_in_schema=False)
async def ui_state() -> Response:
    """Dashboard poll target: active requests, history and the slot grid."""
    if not app_pkg.UI_ENABLED:
        return _ui_not_found()
    return JSONResponse(_ui_snapshot())


@app_pkg.app.get("/proxy/ui/request/{rid}", include_in_schema=False)
async def ui_request(rid: str) -> Response:
    """Full prompt and response tail of one request (in flight or finished)."""
    if not app_pkg.UI_ENABLED:
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


@app_pkg.app.get("/proxy/ui/events", include_in_schema=False)
async def ui_events() -> Response:
    """The live token stream of every in-flight request (SSE)."""
    if not app_pkg.UI_ENABLED:
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


@app_pkg.app.api_route(
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
    client = app_pkg.app.state.clients[0]
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
