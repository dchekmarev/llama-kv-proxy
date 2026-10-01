# chat_flow/_model.py

"""Effective-model resolution for a chat request."""

import chat_flow
from backend.llama_client import LlamaClient
from backend.slot_manager import SlotManager

from . import _state

log = _state.log


async def _resolve_effective_model(
    sm: SlotManager,
    clients: list[LlamaClient],
    client_model: str | None,
    no_cache: bool,
) -> tuple[str, bool]:
    # Effective model: the client's model, else the loaded model, else MODEL_ID.
    # This single value drives the cache key, the slot pool, and the request.
    if client_model:
        effective_model = client_model
        # Model aliases: llama.cpp resolves aliases (e.g. "default") to the
        # real loaded model, but the proxy discovers slot pools only under the
        # real id (app._poll_slots). An alias would never find a pool and every
        # such request would collapse onto the bootstrap slot (0, alias, 0) and
        # cache into a separate namespace. So the name is mapped to the model id
        # the backend itself routes on, and the request is treated as that
        # model, sharing its pool and its cache keys with real-name requests.
        if not sm.has_pool(effective_model):
            # First choice: the backend's own preset table (ids + aliases). It
            # is served whether or not a model is loaded, so an alias resolves
            # during a restart too, and it stays unambiguous with several models
            # in the preset.
            resolved = await clients[0].resolve_model_id_cached(client_model)
            if resolved is None:
                # The name is not in the preset. A plain single-model backend
                # may still ignore the requested name and serve its only model,
                # so fall back to that when it is unambiguous.
                mid = await clients[0].get_model_id_cached()
                if mid != "unknown" and sm.discovered_models() == {mid}:
                    resolved = mid
            if resolved is not None:
                effective_model = resolved
                log.info(
                    "model_alias_resolved alias=%s resolved=%s",
                    client_model,
                    resolved,
                )
            else:
                # Truly unresolvable. The name still goes upstream unchanged --
                # the backend may well serve it -- but the request is proxied
                # without cache treatment: the alias must not become a cache
                # namespace, since the key is sha256(model_id + "\n" + prefix)
                # and such a namespace reuses none of what the real-name
                # requests cached, and would outlive the outage on disk. The
                # WARNING marks the window in which cache reuse is degraded.
                no_cache = True
                log.warning(
                    "model_alias_unresolved alias=%s discovered=%s",
                    client_model,
                    sorted(sm.discovered_models()),
                )
    else:
        # TTL-cached model id: no per-request HTTP round-trip. "unknown"
        # (never resolved) falls back to MODEL_ID.
        mid = await clients[0].get_model_id_cached()
        effective_model = mid if mid != "unknown" else chat_flow.MODEL_ID
    return effective_model, no_cache
