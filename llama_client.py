# llama_client.py

"""HTTP client for one llama.cpp backend (llama-server).

One instance per ``BACKENDS`` entry; ``app`` builds them in its lifespan and
hands them to the slot manager and the chat flow. The client owns everything
backend-shaped:

* the wire paths llama-server exposes -- ``/v1/models``, ``/slots``,
  ``/slots/{id}``, ``/v1/chat/completions``, ``/metrics``;
* router awareness: a server started with ``--models-preset`` lists every
  preset model in ``/v1/models`` (each with a ``status`` field, one of them
  loaded) and needs a ``model`` on its slot endpoints to pick the right slot
  set, while a plain server serves a single model and has neither;
* the model-id cache (TTL + single-flight), so the request path never pays an
  HTTP round-trip to learn which model a backend serves or which model id a
  client-side alias names;
* the error mapping of the non-streaming completion: a provider failure comes
  back as ``{"object": "error", "status": ..., "message": ...}`` rather than an
  exception, so the caller can pass a client fault (4xx) through unchanged and
  map a genuine upstream failure (5xx, transport error, unreadable body) to
  502, with the backend's own text kept in ``raw``.

Every read here is soft: a down, wedged or unsupported backend degrades to
``None`` / ``False`` / ``[]`` and never raises, so one dead backend cannot take
the proxy down. The exception is the streaming completion, whose response object
the chat flow reads itself; that is why a request failure is reported through
the envelope rather than through an exception even there.
"""

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from typing import Any, Literal, overload

import httpx

import config

log = logging.getLogger(__name__)


# --- restore outcomes -------------------------------------------------------

# A slot restore reports a bool: True when the slot now holds the requested KV,
# False when it failed for an ordinary reason (the caller keeps the meta and a
# later request may hit it). The two sentinels name the failures that need a
# different decision, so a caller never has to parse a message:
#   RESTORE_MISSING -- the .bin is gone for good, drop the meta;
#   RESTORE_ERROR   -- the restore request itself failed (backend down), keep it.
# They are strings so they survive the JSON decision diagnostics, and falsy so
# a plain truthiness test reads them as "not restored".


class _Outcome(str):
    """A restore outcome sentinel: a nameable, JSON-safe, falsy value."""

    __slots__ = ()

    def __bool__(self) -> bool:
        return False


# Typed as Any so the sentinels compose with the bool restore result wherever
# the outcome is stored or passed on (they are not a bool at runtime, only
# distinguishable from one).
RESTORE_MISSING: Any = _Outcome("restore_missing")
RESTORE_ERROR: Any = _Outcome("restore_error")


# --- /v1/models interpretation ----------------------------------------------
# Two listings mean two very different backends. A plain server answers
# {"data": [{"id": "m"}]} -- one model, no status. A router (--models-preset)
# answers one entry per preset model, each with a status object; at most one is
# loaded and the rest are candidates waiting for it. Everything below is
# defensive about the entry shape: the fetch succeeding does not mean the body
# is well formed, and a malformed entry must never raise.


def _entries(data: object) -> list[dict[str, Any]]:
    """The object entries of a /v1/models data list, in backend order."""
    if not isinstance(data, list):
        return []
    return [entry for entry in data if isinstance(entry, dict)]


def _model_id(entry: dict[str, Any]) -> str | None:
    """The entry's model id, None when it has no usable one."""
    model_id = entry.get("id")
    if isinstance(model_id, str) and model_id:
        return model_id
    return None


def _load_status(entry: dict[str, Any]) -> str:
    """The entry's load status, lowercased and tolerant of the shape."""
    status = entry.get("status")
    if isinstance(status, dict):
        value = status.get("value")
        return value.lower() if isinstance(value, str) else ""
    if isinstance(status, str):
        return status.lower()
    return ""


def _is_router_listing(entries: list[dict[str, Any]]) -> bool:
    """True for a preset listing: an entry with a `status` field.

    That single field is what separates a multi-model router from a plain
    server, and it is present whether or not anything is loaded right now.
    """
    return any("status" in entry for entry in entries)


def _loaded_ids(entries: list[dict[str, Any]]) -> list[str]:
    """The ids of the loaded entries, in backend order."""
    return [
        model_id
        for entry in entries
        if (model_id := _model_id(entry)) and _load_status(entry) == "loaded"
    ]


def _served_model_id(data: object) -> str | None:
    """The model a backend will actually serve for the next request.

    A router names it through its loaded preset entry; a plain server serves
    the only model it has. A router with nothing loaded resolves to None on
    purpose: its first preset entry names a *different* model, and reporting
    that would fork the cache namespace and break alias resolution.
    """
    entries = _entries(data)
    if not entries:
        return None
    if _is_router_listing(entries):
        loaded = _loaded_ids(entries)
        return loaded[0] if loaded else None
    return _model_id(entries[0])


def _resolve_alias(data: object, name: str) -> str | None:
    """Map a client-supplied model name to a backend model id.

    The preset table (ids plus their aliases) is authoritative and does not
    depend on which model is loaded, so an alias keeps resolving across a
    backend restart -- that is what stops a restart from splitting the cache
    namespace in two. A model id always wins over an alias; an alias claimed by
    two models is ambiguous and resolves to nothing, because guessing would send
    the request to a model the caller did not ask for. Unusable entries are
    skipped rather than fatal: the fetch succeeded, it is just not resolvable.
    """
    by_alias: dict[str, str] = {}
    ambiguous: set[str] = set()
    for entry in _entries(data):
        model_id = _model_id(entry)
        if model_id is None:
            continue
        if model_id == name:
            return model_id
        aliases = entry.get("aliases")
        if not isinstance(aliases, list):
            continue
        for alias in aliases:
            if not isinstance(alias, str):
                continue
            if by_alias.setdefault(alias, model_id) != model_id:
                ambiguous.add(alias)
    if name in ambiguous:
        return None
    return by_alias.get(name)


def _error_body(message: str, status: int, raw: str = "") -> dict[str, Any]:
    """The provider-failure envelope of the non-streaming completion path.

    ``status`` is the structured backend status the caller maps (4xx passed
    through, everything upstream mapped to 502) and ``raw`` the backend's own
    body, so the caller can show the real error instead of a bare status code.
    """
    body: dict[str, Any] = {"object": "error", "message": message, "status": status}
    if raw:
        body["raw"] = raw
    return body


# --- model listing cache ----------------------------------------------------


class _ModelsCache:
    """The last known /v1/models listing and its single-flight coordination.

    ``list`` is the last *good* listing (None until the first success),
    ``_at`` the monotonic stamp of the last attempt, and ``_inflight`` the
    shared load that concurrent callers await instead of each firing their own
    /v1/models. A failed attempt updates ``_at`` too, so a down backend is
    retried on the retry interval instead of on every request.
    """

    __slots__ = ("_at", "_inflight", "list")

    def __init__(self) -> None:
        self.list: list[dict[str, Any]] | None = None
        self._at = 0.0
        self._inflight: asyncio.Future[list[dict[str, Any]] | None] | None = None

    def fresh(self, ttl: float) -> bool:
        """True while a known listing is still inside its TTL."""
        return self.list is not None and time.monotonic() - self._at < ttl

    def recently_failed(self, retry: float) -> bool:
        """True when a refresh is not worth attempting yet: nothing is known
        and the last attempt is younger than the retry interval."""
        return self.list is None and time.monotonic() - self._at < retry

    async def load(
        self, fetch: Callable[[], Awaitable[list[dict[str, Any]] | None]]
    ) -> list[dict[str, Any]] | None:
        """Load the listing once for all concurrent callers.

        The first caller starts the load and the rest await the same future, so
        a burst of requests on an expired cache costs one round-trip. A failed
        load reaches every waiter (each decides what a dead backend means) and
        the next call retries it. If the load dies with the caller that started
        it -- that client's disconnect took the shared request down -- a waiter
        that is still running starts a fresh load instead of dying with it.
        """
        while True:
            load = self._inflight
            leader = load is None
            if load is None:
                load = asyncio.ensure_future(fetch())
                self._inflight = load
            try:
                return await load
            except asyncio.CancelledError:
                if leader or not load.cancelled():
                    raise
                if self._inflight is load:
                    self._inflight = None
            finally:
                if leader and self._inflight is load:
                    self._inflight = None


class LlamaClient:
    """Async client for one backend at ``url``.

    The instance owns one pooled ``httpx.AsyncClient`` (``client``) and is
    closed once, at shutdown, by ``close()``.
    """

    def __init__(self, url: str) -> None:
        self.url = url.rstrip("/")
        self.client = httpx.AsyncClient(
            base_url=self.url,
            timeout=config.REQUEST_TIMEOUT,
        )
        self._models = _ModelsCache()
        # Router mode is detected from the /v1/models shape and remembered.
        # None = not detected yet; a failed detection is not remembered, so a
        # transient failure cannot pin a router to "plain" for the process life.
        self._is_router: bool | None = None

    # --- plumbing -----------------------------------------------------------

    async def close(self) -> None:
        """Close the pooled HTTP client (called once, on shutdown)."""
        await self.client.aclose()

    @staticmethod
    def _with_slot_id(
        body: dict[str, Any], slot_id: int | None
    ) -> tuple[dict[str, Any], dict[str, int]]:
        """A copy of the request body pinned to ``slot_id``, plus its query.

        llama-server accepts the target slot in three places at once and the
        builds in the wild disagree on which one they honour, so all of them are
        set: ``slot_id`` and ``id_slot`` at the top level, the same two inside
        ``options``, and the query parameters. ``options`` is only rewritten
        when there is a pin, so an unpinned request is forwarded untouched (and
        without a copy) and a malformed ``options`` is replaced rather than
        merged into.

        ``_slot_id`` is the internal record of the pin; llama-server ignores
        unknown body fields, and it makes the pin visible in the request log.
        """
        if slot_id is None:
            return body, {}
        options = body.get("options")
        pinned_options = dict(options) if isinstance(options, dict) else {}
        pinned_options.update({"slot_id": slot_id, "id_slot": slot_id})
        pinned = dict(body)
        pinned["options"] = pinned_options
        pinned["_slot_id"] = slot_id
        pinned["slot_id"] = slot_id
        pinned["id_slot"] = slot_id
        return pinned, {"slot_id": slot_id, "id_slot": slot_id}

    async def _cached_models(self) -> list[dict[str, Any]] | None:
        """The /v1/models listing, from cache when it is still fresh.

        A failed refresh keeps the last known listing: a transient backend
        failure must not fork the model namespace, and the next request is
        better served by a slightly stale id than by none. A fetch that fails
        with nothing known yet is throttled to the short retry interval, so a
        backend that is still starting up is polled instead of hammered.
        """
        if self._models.fresh(config.MODEL_ID_TTL):
            return self._models.list
        if self._models.recently_failed(config.UNKNOWN_MODEL_ID_RETRY):
            return None
        models = await self._models.load(self.get_models)
        self._models._at = time.monotonic()
        if models is None:
            return self._models.list
        self._models.list = models
        return models

    # --- model discovery ----------------------------------------------------

    async def get_models(self) -> list[dict[str, Any]] | None:
        """The backend's raw /v1/models entries, None on any failure.

        Verbatim on purpose: /v1/models is proxied to the caller as it is, and
        the router-aware parts of the client (the model id, the alias table,
        the active models) are all derived from the same listing. The timeout is
        explicit and short because this is probed on the request path -- a slow
        answer means the server is wedged, not busy.
        """
        try:
            response = await self.client.get(
                "/v1/models", timeout=config.MODEL_ID_TIMEOUT
            )
            response.raise_for_status()
            data = response.json().get("data")
        except Exception as e:  # noqa: BLE001
            log.debug("get_models_failed url=%s: %s", self.url, e)
            return None
        if not isinstance(data, list):
            log.warning("get_models_unexpected_shape url=%s", self.url)
            return None
        return data

    async def get_loaded_model(self) -> str | None:
        """The model the backend serves right now, None when undeterminable.

        Always a fresh listing: this is the uncached probe behind the health
        check and the router-aware slot discovery, where a stale answer would be
        worse than no answer.
        """
        return _served_model_id(await self.get_models())

    async def get_model_id(self) -> str:
        """``get_loaded_model`` with the "unknown" placeholder for a name that
        cannot be determined, so callers never have to handle None."""
        return await self.get_loaded_model() or "unknown"

    async def get_model_id_cached(self) -> str:
        """The served model id from the TTL cache, "unknown" if undeterminable.

        The hot path: every request resolves the effective model through here,
        so a request costs no /v1/models round-trip. "unknown" is a placeholder,
        not a model name -- the caller substitutes its configured MODEL_ID rather
        than letting a cache namespace be named after it.
        """
        return _served_model_id(await self._cached_models()) or "unknown"

    async def resolve_model_id_cached(self, name: str) -> str | None:
        """A client model name resolved to a backend model id, None if unknown.

        Reads the same cached listing as ``get_model_id_cached``, so a client
        that always sends an alias still costs one /v1/models per TTL.
        """
        return _resolve_alias(await self._cached_models(), name)

    async def is_router(self) -> bool:
        """True when the backend serves a --models-preset model set.

        Detected once from the listing shape and then remembered; slot
        discovery and slot operations branch on it (a router needs ``model`` on
        its slot endpoints, a plain server must not be sent one).
        """
        if self._is_router is None:
            models = await self.get_models()
            if models is None:
                return False
            self._is_router = _is_router_listing(_entries(models))
        return bool(self._is_router)

    async def get_active_models(self) -> list[str]:
        """The model ids a backend currently serves, for the /metrics scrape.

        A router reports only its loaded models (an unloaded one has no metrics
        worth scraping and would report the previous model's numbers under the
        new label), a plain server its single model, an undeterminable backend
        nothing at all.
        """
        entries = _entries(await self.get_models())
        if _is_router_listing(entries):
            return _loaded_ids(entries)
        served = _served_model_id(entries)
        return [served] if served else []

    async def get_metrics(self, model: str | None = None) -> str | None:
        """The backend's raw /metrics text, None on any failure.

        Short explicit timeout and a None on failure so one slow or dead
        backend cannot stall or break the aggregated scrape.
        """
        try:
            response = await self.client.get(
                "/metrics",
                params={"model": model} if model else None,
                timeout=config.METRICS_TIMEOUT,
            )
            response.raise_for_status()
            return response.text
        except Exception as e:  # noqa: BLE001
            log.debug("get_metrics_failed url=%s model=%s: %s", self.url, model, e)
            return None

    # --- health -------------------------------------------------------------

    async def health(self) -> dict[str, Any]:
        """A liveness probe of this backend; never raises.

        ``ok`` means the server answered /v1/models at all. ``model_id`` is what
        it says it serves, which is None for a router with nothing loaded -- an
        up server with no model is still up, and the two answers must not be
        collapsed into one.
        """
        models = await self.get_models()
        if models is None:
            return {"ok": False, "model_id": None, "url": self.url}
        return {"ok": True, "model_id": _served_model_id(models), "url": self.url}

    # --- slots --------------------------------------------------------------

    async def get_slots(self, model: str | None = None) -> list[dict[str, Any]] | None:
        """The backend's slot table, None when unsupported or unreachable.

        ``model`` narrows the table on a router, which serves one slot set per
        model; a plain server is asked without any query parameter at all.
        """
        try:
            response = await self.client.get(
                "/slots",
                params={"model": model} if model else None,
                timeout=config.MODEL_ID_TIMEOUT,
            )
            response.raise_for_status()
            data = response.json()
        except Exception as e:  # noqa: BLE001
            log.debug("get_slots_failed url=%s model=%s: %s", self.url, model, e)
            return None
        if not isinstance(data, list):
            log.warning("get_slots_unsupported url=%s model=%s", self.url, model)
            return None
        return data

    async def _slot_action(
        self, slot_id: int, action: str, payload: dict[str, Any]
    ) -> int | None:
        """POST one slot action and report the backend status, None on failure.

        The slot is addressed in the path and the action in the query, while a
        router routes by the model in the body. The status alone decides the
        outcome, so callers get a bool and never parse the answer body. The
        timeout is the long one: a save writes a .bin file and a restore reads
        one back, which is not an instant operation.
        """
        try:
            response = await self.client.post(
                f"/slots/{slot_id}",
                params={"action": action},
                json=payload,
                timeout=config.REQUEST_TIMEOUT,
            )
        except Exception as e:  # noqa: BLE001
            log.warning(
                "slot_action_failed action=%s slot=%d url=%s: %s",
                action,
                slot_id,
                self.url,
                e,
            )
            return None
        return response.status_code

    async def save_slot(self, slot_id: int, key: str, model: str | None = None) -> bool:
        """Persist a slot's KV cache under ``key`` (llama.cpp action=save).

        Any status below 400 counts. llama-server answers 200/204 when it wrote
        the file and 4xx/5xx when it could not -- an empty or still-busy slot is
        an ordinary outcome here, not an error -- so a save is a plain bool and
        the caller never has to distinguish "nothing to save" from "save failed".
        """
        payload: dict[str, Any] = {"filename": key}
        if model:
            payload["model"] = model
        status = await self._slot_action(slot_id, "save", payload)
        return status is not None and status < 400

    async def restore_slot(
        self, slot_id: int, key: str, model: str | None = None
    ) -> bool:
        """Load ``key``'s cache into a slot (llama.cpp action=restore).

        Unlike a save, only a strict 200 counts as restored: any other success
        code is a failure, so the caller keeps the meta and tries again later
        instead of dropping a cache that may well be intact. A 404 is the one
        decisive answer -- the .bin is gone -- and reports RESTORE_MISSING so
        the caller drops the meta rather than retrying a target that cannot
        come back. Everything else, a transport error included, is a plain
        False: the cache is probably still valid.
        """
        payload: dict[str, Any] = {"filename": key}
        if model:
            payload["model"] = model
        status = await self._slot_action(slot_id, "restore", payload)
        if status == 200:
            return True
        if status == 404:
            log.info("restore_missing key=%s slot=%d url=%s", key[:16], slot_id, self.url)
            return RESTORE_MISSING
        return False

    async def erase_slot(self, slot_id: int, model: str | None = None) -> bool:
        """Drop a slot's in-memory KV cache (llama.cpp action=erase).

        Best effort: the erase guards against a slot that would otherwise start
        on top of a stale prompt, so a failure is not worth failing the request
        over -- the request itself is the fallback.
        """
        payload: dict[str, Any] = {}
        if model:
            payload["model"] = model
        status = await self._slot_action(slot_id, "erase", payload)
        return status is not None and status < 400

    async def delete_cache_file(self, key: str, model: str | None = None) -> bool:
        """Ask the backend to drop a saved cache file.

        Soft by contract, because llama.cpp has no such endpoint on most builds:
        an older server answers 404, which simply means "no file to delete
        here", and a down backend raises. Both are False, so a purge of the whole
        cache can sweep every key without a branch on the backend version.
        """
        params: dict[str, Any] = {"filename": key}
        if model:
            params["model"] = model
        try:
            response = await self.client.delete(
                "/slots", params=params, timeout=config.REQUEST_TIMEOUT
            )
        except Exception as e:  # noqa: BLE001
            log.debug("delete_cache_file_failed key=%s url=%s: %s", key[:16], self.url, e)
            return False
        return response.status_code < 400

    # --- chat ---------------------------------------------------------------

    @overload
    async def chat_completions(
        self,
        body: dict[str, Any],
        slot_id: int | None,
        stream: Literal[True],
        model: str | None = None,
    ) -> httpx.Response:
        """Streamed completion: the open upstream response."""

    @overload
    async def chat_completions(
        self,
        body: dict[str, Any],
        slot_id: int | None = None,
        stream: bool = False,
        model: str | None = None,
    ) -> Any:
        """Non-streamed completion: the parsed body, or the error envelope."""

    async def chat_completions(
        self,
        body: dict[str, Any],
        slot_id: int | None = None,
        stream: bool = False,
        model: str | None = None,
    ) -> httpx.Response | dict[str, Any]:
        """POST a chat completion, pinned to ``slot_id``.

        Streamed: an open response whose upstream status is left untouched, so
        the caller reads ``aiter_raw`` and closes it, and can map a 4xx itself
        instead of seeing it as a parse failure.

        Non-streamed: the parsed body, with a body-level failure folded into the
        ``{"object": "error"}`` envelope rather than raised. That keeps one
        failure path: the caller reports the backend's own message and decides
        the HTTP status from the structured field, passing a client fault (4xx)
        through and mapping a genuine upstream failure (5xx, unreadable body) to
        502. A provider error the backend encoded in a 200 body is passed
        through untouched, since the caller already handles it.

        A transport failure (connect error, timeout) is *not* folded in: no
        backend response exists to report, so the ``httpx.HTTPError`` is left
        to propagate. The caller owns that classification — it answers 502 for
        an upstream failure where an enveloped 500 could read as a proxy bug.
        """
        if model is not None:
            body = {**body, "model": model}
        body, params = self._with_slot_id(body, slot_id)
        if stream:
            request = self.client.build_request(
                "POST",
                "/v1/chat/completions",
                json=body,
                params=params,
                timeout=config.REQUEST_TIMEOUT,
            )
            return await self.client.send(request, stream=True)
        response = await self.client.post(
            "/v1/chat/completions",
            json=body,
            params=params,
            timeout=config.REQUEST_TIMEOUT,
        )
        if response.status_code >= 400:
            return _error_body(
                f"provider returned HTTP {response.status_code}",
                response.status_code,
                raw=response.text,
            )
        try:
            payload = response.json()
        except ValueError:  # unreadable JSON (a real body, not a fault)
            return _error_body(
                "provider returned an unreadable body", 502, raw=response.text
            )
        if not isinstance(payload, dict):
            return _error_body(
                "provider returned a non-object body", 502, raw=response.text
            )
        return payload
