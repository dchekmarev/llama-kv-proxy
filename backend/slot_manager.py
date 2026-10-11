# backend/slot_manager.py

"""Exclusive backend slot pools: discovery, FIFO acquisition, KV bookkeeping.

Every request owns one backend slot exclusively, from the prompt eval until
the response is done, so the proxy mirrors what each backend reports through
GET /slots and hands the picked slot to the flow layer, which releases it
exactly once.

A slot is identified by `GSlot = (backend index, model id, slot id)`. The
model id is part of the identity because a router backend serves every model
on its own child with its own slots: slot 0 of model A and slot 0 of model
B are two different slots, and their pools never overlap.

Pools are discovery driven: the periodic poll (app) pushes a report through
set_backend_slots, and a request re-polls the pool it is about to use
(freshen_model), so a slot cut or a model reload is seen within a second
instead of at the next poll. A stale slot id is not cosmetic: llama.cpp maps
a slot id onto a physical slot with `id % n_slots`, so a stale id can wrap
onto a live slot and collide with that slot's save.

Acquisition is FIFO per model. Waiters park in a queue rather than on one
particular slot's lock, so every release wakes the longest waiting request
and a cancelled or timed out waiter removes itself instead of being served
after it is gone. Picking is lock scoped: the free check and the acquire
happen with no await in between, so two concurrent acquires can never be
handed the same slot, and a woken waiter re-picks instead of inheriting a
slot it never checked.

Per-slot bookkeeping, dropped when a slot leaves its pool:
- `_last_used`: the LRU mark, refreshed on acquire and on a successful save;
- `_last_saved`: the key whose KV the slot currently holds. A save does not
  clear the slot, so restoring that key into this very slot would only
  re-read the .bin and rebuild identical KV; the restore is skipped and
  counted as a hit instead.
"""

import asyncio
import logging
import time
from collections import deque
from collections.abc import Callable

from backend.llama_client import RESTORE_ERROR
from core import pause as pause_mod
from core import promstats
from core.config import BACKENDS, SKIP_RESTORE_SAME_SLOT, SLOT_FRESHEN_INTERVAL_S

log = logging.getLogger(__name__)

# (backend index, model id, backend slot id): the identity of one exclusive
# slot. A router gives every model its own slots, hence the model id.
GSlot = tuple[int, str, int]

# acquire_for_request's `restored`: True/False from LlamaClient.restore_slot,
# its RESTORE_MISSING / RESTORE_ERROR sentinels, or None when no restore was
# attempted (no restore key). The sentinels are opaque values, so the outcome
# is opaque here too and the call site compares it.
RestoreOutcome = object

# Fallback for a model no poll has discovered yet: slot 0 of the first
# backend. It keeps such a request working (and cacheable under its own
# model id) until the first discovery or freshen corrects the pool.
_BOOTSTRAP_BACKEND = 0
_BOOTSTRAP_SLOT = 0


def _slot_ids(slots: list[dict]) -> list[int] | None:
    """Slot ids of a /slots report, or None when the body is unusable.

    A malformed report must not be applied: an empty or partial pool would
    orphan the slots the running requests already hold.
    """
    ids: list[int] = []
    for slot in slots or []:
        if not isinstance(slot, dict) or not isinstance(slot.get("id"), int):
            return None
        ids.append(slot["id"])
    return ids


def _first_n_ctx(slots: list[dict]) -> int | None:
    """The `n_ctx` of the first slot that reports a usable one, else None.

    `n_ctx` is the shared context size (llama.cpp ``--parallel N
    --kv-unified``: the N slots share one context), not a per-slot budget:
    every slot of the pool reports the same value. A missing or non-positive
    value leaves the pool ungated.
    """
    for slot in slots or []:
        if not isinstance(slot, dict):
            continue
        n_ctx = slot.get("n_ctx")
        if isinstance(n_ctx, int) and not isinstance(n_ctx, bool) and n_ctx > 0:
            return n_ctx
    return None


class SlotManager:
    """Per-(backend, model) slot pools with FIFO, lock-scoped acquisition."""

    def __init__(self) -> None:
        # One entry per backend, mirroring config.BACKENDS plus the client
        # (attached by set_clients once the lifespan has built the clients).
        self.backends: list[dict] = []
        # (backend, model) -> slot ids, and the raw /slots report behind them.
        self._pools: dict[tuple[int, str], list[int]] = {}
        self._backend_slots: dict[tuple[int, str], list[dict]] = {}
        # Per-slot state, keyed by GSlot.
        self._locks: dict[GSlot, asyncio.Lock] = {}
        self._last_used: dict[GSlot, float] = {}
        self._last_saved: dict[GSlot, str] = {}
        # model -> FIFO queue of the requests parked on a fully busy pool,
        # each entry carrying its context demand so a waiter that cannot
        # progress is skipped in favour of a younger one that can.
        self._waiters: dict[str, deque[tuple[asyncio.Future[None], int | None]]] = {}
        # (backend, model) -> last freshen, the on-demand re-poll throttle.
        self._freshened: dict[tuple[int, str], float] = {}
        # Context-budget admission control, per (backend, model):
        #   _total_ctx     -- the shared context size (from /slots n_ctx, or
        #                     the /v1/models meta fallback); absent => unknown,
        #                     so the pool is not gated.
        #   _reserved      -- tokens reserved by the slots currently held.
        #   _slot_reserved -- what a held slot reserved, keyed by GSlot, so
        #                     release returns exactly what acquire took.
        self._total_ctx: dict[tuple[int, str], int] = {}
        self._reserved: dict[tuple[int, str], int] = {}
        self._slot_reserved: dict[GSlot, int] = {}

    # --- wiring ---------------------------------------------------------------

    def set_clients(self, clients: list) -> None:
        """Attach the per-backend clients, index aligned with config.BACKENDS.

        A backend without a client (or without config) still gets an entry,
        so a backend index always resolves to a slot-pool namespace.
        """
        self.backends = []
        for i in range(max(len(clients), len(BACKENDS))):
            entry = dict(BACKENDS[i]) if i < len(BACKENDS) else {}
            entry["client"] = clients[i] if i < len(clients) else None
            self.backends.append(entry)

    def set_backend_slots(
        self, backend_index: int, model: str, slots: list[dict]
    ) -> None:
        """Replace one pool with a fresh /slots report.

        The report replaces the pool (a reloaded model comes back with
        fewer slots, and merging would keep slot ids that no longer exist).
        Bookkeeping of the slots that left is dropped; the lock of a slot
        that is still held is kept, since its owner has to release it.
        """
        ids = _slot_ids(slots)
        if ids is None:
            log.warning(
                "slots_malformed be=%d model=%s slots=%r", backend_index, model, slots
            )
            return
        self._forget_gone(backend_index, model, set(ids))
        self._pools[(backend_index, model)] = ids
        self._backend_slots[(backend_index, model)] = list(slots)
        # The shared context size rides on the report; a report that carries
        # none leaves the pool ungated (the /v1/models fallback may fill it).
        total = _first_n_ctx(slots)
        if total is None:
            self._total_ctx.pop((backend_index, model), None)
        else:
            self._total_ctx[(backend_index, model)] = total

    def discovered_models(self) -> set[str]:
        """Model ids that at least one backend reported slots for.

        A pool with no slots is not a discovered model: it would otherwise
        disable the single-model fallback for every alias, and a pool is
        created for any name that was freshened once.
        """
        return {
            model
            for (backend, model), slots in self._pools.items()
            if slots or any(m == model for (b, m) in self._pools if b != backend)
        }

    def has_pool(self, model: str) -> bool:
        """Whether any backend reported slots under this model id.

        The flow layer uses it to tell a real model name from a client
        alias: an alias never has a pool of its own, it is mapped onto the
        id the backend actually routes on.
        """
        return any(m == model for _backend, m in self._pools)

    def _slots_for_model(self, model: str) -> list[GSlot]:
        """Every discovered slot of a model, across all backends."""
        return [
            (backend, m, sid)
            for (backend, m), ids in self._pools.items()
            if m == model
            for sid in ids
        ]

    def aggregated_state(self) -> list[dict]:
        """Every discovered slot as one dict, for /proxy/health and /proxy/slots.

        Each row merges what the backend reported for that slot with the
        proxy's own bookkeeping (the model, the LRU mark). The state
        vocabulary is the backend's own ("free"/"busy"), overlaid with
        is_processing so a slot wedged in prompt processing reads as busy
        even when the backend forgot to say so.
        """
        rows: list[dict] = []
        for (backend, model), ids in self._pools.items():
            reported = {
                s["id"]: s
                for s in self._backend_slots.get((backend, model), [])
                if isinstance(s, dict) and "id" in s
            }
            url = self.backends[backend].get("url") if backend < len(self.backends) else None
            for sid in ids:
                slot = reported.get(sid, {})
                processing = bool(slot.get("is_processing"))
                rows.append(
                    {
                        "backend": backend,
                        "model": model,
                        "slot": sid,
                        "url": url,
                        "state": "busy" if processing else slot.get("state") or "free",
                        "is_processing": processing,
                        "n_ctx": slot.get("n_ctx"),
                        "total_tokens": slot.get("total_tokens"),
                        "last_used": self._last_used.get((backend, model, sid)),
                    }
                )
        return rows

    # --- bookkeeping ----------------------------------------------------------

    def _lock_for(self, g: GSlot) -> asyncio.Lock:
        """The lock guarding a slot, created on first use."""
        lock = self._locks.get(g)
        if lock is None:
            lock = self._locks[g] = asyncio.Lock()
        return lock

    def _forget_gone(self, backend: int, model: str, live: set[int]) -> None:
        """Drop the bookkeeping of slots that are no longer in the pool."""
        known = set(self._locks) | set(self._last_used) | set(self._last_saved)
        for g in known:
            if g[0] == backend and g[1] == model and g[2] not in live:
                lock = self._locks.get(g)
                if lock is not None and lock.locked():
                    # Still owned by a request: its lock survives until the
                    # owner releases it, then release() drops everything.
                    self._last_used.pop(g, None)
                    self._last_saved.pop(g, None)
                    continue
                self._locks.pop(g, None)
                self._last_used.pop(g, None)
                self._last_saved.pop(g, None)

    def _in_pool(self, g: GSlot) -> bool:
        """Whether a slot is still part of its pool."""
        ids = self._pools.get((g[0], g[1]))
        return ids is not None and g[2] in ids

    # --- context budget -------------------------------------------------------

    def _reserve(self, g: GSlot, amount: int) -> None:
        """Record what a held slot reserved and publish the pool's total.

        Called with the slot's lock already held and before any await, so
        the reservation is exact for every acquire path.
        """
        pool = (g[0], g[1])
        self._slot_reserved[g] = amount
        total = self._reserved.get(pool, 0) + amount
        self._reserved[pool] = total
        promstats.context_reserved_tokens.labels(
            backend=str(g[0]), model=promstats.model_label(g[1])
        ).set(total)

    def _unreserve(self, g: GSlot) -> None:
        """Return a released slot's reservation to its pool.

        A slot with no record (admission was disabled for it, or it was
        already released) changes nothing: release is "exactly once" by
        contract, but a stray call must not drive the pool negative. The
        record of a slot cut from its pool while held is dropped here too.
        """
        amount = self._slot_reserved.pop(g, None)
        if amount is None:
            return
        pool = (g[0], g[1])
        total = self._reserved.get(pool, 0) - amount
        if total > 0:
            self._reserved[pool] = total
        else:
            self._reserved.pop(pool, None)
            total = 0
        promstats.context_reserved_tokens.labels(
            backend=str(g[0]), model=promstats.model_label(g[1])
        ).set(total)

    # --- picking --------------------------------------------------------------

    @staticmethod
    def _bootstrap_slot(model: str) -> GSlot:
        return (_BOOTSTRAP_BACKEND, model, _BOOTSTRAP_SLOT)

    def _is_busy(self, g: GSlot) -> bool:
        """Whether a slot is currently held (a free slot may have no lock yet)."""
        lock = self._locks.get(g)
        return lock is not None and lock.locked()

    def _fits(self, g: GSlot, need_tokens: int | None) -> bool:
        """Whether `need_tokens` fits the slot's pool after what is reserved.

        `need_tokens` None means admission is disabled. An unknown
        `total_ctx` (no report carried an `n_ctx`, and the fallback found
        none) also means no gate: without a context size there is nothing to
        admit against.
        """
        if need_tokens is None:
            return True
        total = self._total_ctx.get((g[0], g[1]))
        if total is None:
            return True
        return self._reserved.get((g[0], g[1]), 0) + need_tokens <= total

    def _oversized(self, model: str, need_tokens: int | None) -> bool:
        """Whether `need_tokens` fits no empty pool of the model's backends.

        Such a request can never be admitted by waiting for budget to free,
        so the gate is waived for it (best effort: a free slot if there is
        one, else park, exactly as with admission disabled). A backend with
        an unknown `total_ctx` counts as fitting, since its slots are
        ungated; with no candidate backend known, nothing is oversized.
        """
        if need_tokens is None:
            return False
        candidates = self._slots_for_model(model) or [self._bootstrap_slot(model)]
        for backend in {g[0] for g in candidates}:
            total = self._total_ctx.get((backend, model))
            if total is None or need_tokens <= total:
                return False
        return True

    def _get_free_or_oldest(
        self, model: str, need_tokens: int | None = None
    ) -> tuple[GSlot, asyncio.Lock]:
        """Pick a slot for a model: a free one, else the least recently used.

        Never blocks and never hands out a busy slot: the caller acquires
        the returned lock, and that acquire happens without an await in
        between, so the check and the claim are atomic.

        With `need_tokens` set, a free slot is only a candidate when the
        model's pool has the budget for it; when every free slot is over
        budget the oldest slot is returned instead (busy or not), so the
        caller parks and re-picks after a release.

        A free slot that just saved holds a usable KV cache, so it is used
        only when every free slot does: a request that does not want that
        key has nothing to gain from it, and a request that does (it is the
        continuation of that conversation) re-picks it as the oldest slot
        and skips the no-op restore.
        """
        candidates = self._slots_for_model(model) or [self._bootstrap_slot(model)]
        free = [g for g in candidates if not self._is_busy(g)]
        if need_tokens is not None:
            free = [g for g in free if self._fits(g, need_tokens)]
        if free:
            pick = next((g for g in free if g not in self._last_saved), free[0])
        else:
            pick = min(candidates, key=lambda g: self._last_used.get(g, 0.0))
        return pick, self._lock_for(pick)

    # --- waiting --------------------------------------------------------------

    def _can_progress(self, model: str, need_tokens: int | None) -> bool:
        """Whether a waiter with this demand can take a slot right now.

        True when admission does not gate it (no demand, or an oversized
        request the gate waives) or when some free slot's pool has the
        budget for it. `_wake_one` uses it to skip a parked waiter that
        would only re-park, so it does not hide a younger one that fits.
        """
        if need_tokens is None or self._oversized(model, need_tokens):
            return True
        candidates = self._slots_for_model(model) or [self._bootstrap_slot(model)]
        return any(
            not self._is_busy(g) and self._fits(g, need_tokens) for g in candidates
        )

    async def _park(self, model: str, need_tokens: int | None = None) -> None:
        """Wait for a release of the model's pool, in FIFO order.

        The queue is the wait list: it is what makes the wake order fair,
        and dropping a cancelled or timed out waiter from it keeps the next
        release from serving a request that is already gone.
        """
        queue = self._waiters.setdefault(model, deque())
        fut: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        entry = (fut, need_tokens)
        queue.append(entry)
        try:
            await fut
        finally:
            if entry in queue:
                queue.remove(entry)
            if not queue and self._waiters.get(model) is queue:
                del self._waiters[model]

    def _wake_one(self, model: str) -> None:
        """Wake the oldest parked waiter of a model that can take a slot now.

        Waking the plain oldest let a request that no longer fits (it
        re-parks) hide a younger one that does, leaving the just-freed
        slot and budget idle until an unrelated release. Skipping the
        waiters that cannot progress wakes the oldest that can; when none
        can, nobody is woken and the next release re-tries.
        """
        queue = self._waiters.get(model)
        if not queue:
            return
        for i, (fut, need_tokens) in enumerate(queue):
            if fut.done() or not self._can_progress(model, need_tokens):
                continue
            del queue[i]
            fut.set_result(None)
            break
        if not queue and self._waiters.get(model) is queue:
            del self._waiters[model]

    async def _take_slot(
        self, model: str, need_tokens: int | None = None
    ) -> tuple[GSlot, asyncio.Lock]:
        """Acquire a slot of the model, waiting FIFO while the pool is busy.

        With `need_tokens` set, a slot is only taken while the pool has the
        budget for it; a request that fits no empty pool at all is waived
        (and counted) so it is not parked forever.
        """
        while True:
            if self._oversized(model, need_tokens):
                # Cannot fit even an empty pool: admit it best effort onto a
                # free slot exactly as an ungated request would, else park.
                g, lock = self._get_free_or_oldest(model)
                if not lock.locked():
                    promstats.admission_oversized_total.labels(
                        model=promstats.model_label(model)
                    ).inc()
                    await lock.acquire()
                    return g, lock
                await self._park(model, need_tokens)
                continue
            g, lock = self._get_free_or_oldest(model, need_tokens)
            if not lock.locked() and self._fits(g, need_tokens):
                await lock.acquire()  # free: returns without yielding
                return g, lock
            await self._park(model, need_tokens)

    # --- request lifecycle ----------------------------------------------------

    async def acquire_for_request(
        self,
        model: str,
        restore_key: str | None = None,
        resolve_restore_key: Callable[[str], str] | None = None,
        need_tokens: int | None = None,
    ) -> tuple[GSlot, asyncio.Lock, RestoreOutcome, str | None]:
        """Take a slot for a request and restore a cache into it.

        Returns `(slot, lock, restored, used_key)`. `restored` is True when
        the slot holds the cache, False (or a llama_client sentinel) when a
        restore was attempted and failed, and None when none was attempted.
        `used_key` is the key actually restored: `restore_key` after being
        passed through `resolve_restore_key`, which re-points a candidate
        that a concurrent save has subsumed onto its replacement.

        `need_tokens` is the request's estimated context demand: the slot is
        only taken while the pool's reserved tokens plus `need_tokens` fit
        its total context, and the amount is reserved for as long as the
        slot is held. None disables admission (today's behavior).

        The lock is held on return in every path except cancellation: the
        caller owns the slot until it releases it, exactly once. A failure
        or a timeout of the restore must not leak it, and it must not
        release it either (the request still runs on that slot).
        """
        g, lock = await self._take_slot(model, need_tokens)
        if need_tokens is not None:
            # Reserve under the held lock before any await: release(g)
            # returns exactly this amount, including after a cut slot.
            self._reserve(g, need_tokens)
        # Occupying a slot is a unit of committed work for the pause drain:
        # the slot is held until release() returns it, whether that happens
        # through the pipeline's finally, the stream reader or a background
        # save. release() is the single point that ends the unit.
        pause_mod.activity_begin()
        # Occupying a slot is its "last used" moment, even for a small
        # request that never saves: it is the LRU order the next pick uses.
        self._last_used[g] = time.time()
        if restore_key is None:
            # The chat will run on whatever the slot holds, so the record of
            # which key it holds is no longer valid.
            self._last_saved.pop(g, None)
            return g, lock, None, None
        key = restore_key
        try:
            if resolve_restore_key is not None:
                key = resolve_restore_key(restore_key) or restore_key
            if SKIP_RESTORE_SAME_SLOT and self._last_saved.get(g) == key:
                # The slot already holds this key's KV (it saved it, or
                # restored it and was not used since): restoring would only
                # re-read the .bin and rebuild identical KV. A stale record
                # (e.g. a backend restart reusing the slot ids) costs at most
                # one full re-prefill, and a restore is only a speed
                # optimization, so a wrong skip never changes the output.
                self._last_saved.pop(g, None)
                promstats.restore_skipped_same_slot_total.labels(model=model).inc()
                log.debug("restore_skipped_same_slot g=%s key=%s", g, key[:16])
                return g, lock, True, key
            client = self.backends[g[0]]["client"]
            restored = await client.restore_slot(g[2], key, model=g[1])
        except asyncio.CancelledError:
            # Cancelled (or timed out by the caller's wait_for) while the
            # restore was in flight: this request never gets the slot.
            self.release(g)
            raise
        except Exception as e:  # noqa: BLE001
            # Backend down or the restore failed: the request proceeds
            # without a cache, still holding its slot. RESTORE_ERROR (not
            # False) so the caller keeps the meta and can retry it later.
            log.warning("restore_failed g=%s key=%s: %s", g, key[:16], e)
            return g, lock, RESTORE_ERROR, key
        if restored:
            # Only a successful restore changes what the slot holds; a
            # failed one leaves the previous record valid.
            self._last_saved[g] = key
        return g, lock, restored, key

    def release(self, g: GSlot) -> None:
        """Hand a slot back and wake the oldest waiter of its model.

        Synchronous on purpose: it runs from finally blocks and from
        cancellation paths, where an await could be interrupted. The caller
        must release exactly once per acquire: a second release would free
        a lock another request has acquired in the meantime.
        """
        # End the pause-drain unit this slot opened in acquire_for_request.
        # Idempotent: activity_end clamps at zero, so a slot released without
        # a paired acquire (a test) cannot drive the counter negative.
        pause_mod.activity_end()
        # Return the context budget this slot reserved (if admission was on),
        # even for a slot that was cut from its pool while it was held.
        self._unreserve(g)
        lock = self._locks.get(g)
        if lock is not None and lock.locked():
            lock.release()
        if not self._in_pool(g):
            # A slot that was cut from the pool while it was held: the lock
            # could not be freed when the pool was replaced, so drop it (and
            # the rest of the bookkeeping) now.
            self._locks.pop(g, None)
            self._last_used.pop(g, None)
            self._last_saved.pop(g, None)
        self._wake_one(g[1])

    async def save_after(self, g: GSlot, key: str) -> bool:
        """Save a slot's KV into a cache file and record what it now holds.

        A successful save refreshes the LRU mark (the slot just became
        valuable) and records the key it holds, so a later request that
        picks this slot and wants the same key skips the no-op restore. A
        failed save changes nothing: the mark stays and no key is recorded.
        Exceptions propagate to the caller, which logs them.
        """
        client = self.backends[g[0]]["client"]
        started = time.monotonic()
        try:
            ok = bool(await client.save_slot(g[2], key, model=g[1]))
        finally:
            promstats.save_duration_seconds.labels(model=g[1]).observe(
                time.monotonic() - started
            )
        if not ok:
            return False
        self._last_used[g] = time.time()
        self._last_saved[g] = key
        return True

    def forget_saved(self, g: GSlot, key: str) -> None:
        """Drop the record of what a slot holds, after that cache was discarded.

        save_after records the key and refreshes the LRU mark, but the caller
        may still throw the capture away (an empty .bin, say). The slot then
        holds no cacheable conversation, so the mark must not survive: it would
        keep the slot's LRU position fresh for a cache that does not exist, and
        a later no-op-restore decision would be made on a key with no meta.
        """
        if self._last_saved.get(g) == key:
            self._last_saved.pop(g, None)

    async def freshen_model(self, model: str) -> bool:
        """Re-poll the slots of one model, at most once per interval.

        Called right before a request picks a slot, so a slot cut or a
        model reload is seen within SLOT_FRESHEN_INTERVAL_S instead of at
        the next periodic poll (a stale id wraps onto a live physical slot).
        Throttled per (backend, model) pool, and never raises: a backend
        that is down, or a body it cannot be asked about, keeps the pool
        it has. Returns whether a pool changed.
        """
        changed = False
        now = time.time()
        for backend in range(len(self.backends)):
            pool = (backend, model)
            last = self._freshened.get(pool)
            if last is not None and now - last < SLOT_FRESHEN_INTERVAL_S:
                continue
            # Throttle before the call: a burst of concurrent requests for
            # this model must trigger one re-poll between them.
            self._freshened[pool] = now
            client = self.backends[backend].get("client")
            if client is None:
                continue
            try:
                router = await client.is_router()
                slots = (
                    await client.get_slots(model=model)
                    if router
                    else await client.get_slots()
                )
            except Exception as e:  # noqa: BLE001
                log.warning("freshen_failed be=%d model=%s: %s", backend, model, e)
                continue
            if slots is None:
                # The /slots poll failed (backend down): keep the pool, so
                # the requests running on it keep their slots.
                continue
            before = self._pools.get(pool, [])
            self.set_backend_slots(backend, model, slots)
            changed = changed or self._pools.get(pool, []) != before
            # /slots is often the only context source, but when it carried no
            # n_ctx, fall back to /v1/models so the pool is still gated.
            await self._learn_ctx(backend, model, client)
        return changed

    async def _learn_ctx(self, backend: int, model: str, client) -> None:
        """Cache a pool's context size from /v1/models when /slots lacks it.

        Optional and non-fatal: a missing method, a down backend or a
        malformed listing leaves `total_ctx` unknown, so the pool stays
        ungated.
        """
        pool = (backend, model)
        if pool in self._total_ctx:
            return
        get_ctx = getattr(client, "get_model_ctx", None)
        if get_ctx is None:
            return
        try:
            total = await get_ctx(model)
        except Exception as e:  # noqa: BLE001
            log.debug("model_ctx_failed be=%d model=%s: %s", backend, model, e)
            return
        if isinstance(total, int) and not isinstance(total, bool) and total > 0:
            self._total_ctx[pool] = total
