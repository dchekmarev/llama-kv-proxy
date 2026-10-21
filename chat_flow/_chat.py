# chat_flow/_chat.py

"""The /v1/chat/completions request pipeline."""

import asyncio
import time
from collections.abc import AsyncIterator

import httpx
from fastapi.responses import JSONResponse, Response, StreamingResponse

import chat_flow as chat_flow_pkg
import hashing as hs
from backend.llama_client import LlamaClient
from backend.slot_manager import GSlot, SlotManager
from core import pause as pause_mod
from core import promstats
from core.request_id import request_id_var
from obs import reqlog
from obs import ui as ui_obs

from . import _state
from ._kill import KillToken

log = _state.log


async def chat_flow(
    sm: SlotManager, clients: list[LlamaClient], data: dict
) -> Response:
    """Run the chat pipeline for an already-parsed JSON body.

    Wraps :func:`_chat_once` in the retry loop. A pause that lands before the
    attempt committed a response aborts it (PausedRetry); a retryable backend
    failure before the commit aborts it too (BackendRetry). Either way the loop
    waits/re-dispatches and re-runs the attempt from scratch, so a backend
    restart or a stale pooled connection never leaves the client with a bounce
    it did not have to see. The request group id, timestamp and request log are
    created once, so a retry does not duplicate them; the retry budget is
    BACKEND_RETRY_ATTEMPTS total attempts, spaced by exponential backoff.
    """
    t0 = time.time()
    t0_mono = time.monotonic()
    rid = request_id_var.get()
    ts = reqlog.new_group_ts()
    reqlog.log_file("request", rid, ts, data)
    attempt = 0
    while True:
        await pause_mod.wait_not_paused()
        try:
            return await _chat_once(
                sm,
                clients,
                data,
                rid,
                ts,
                t0,
                t0_mono,
                can_retry=attempt + 1 < chat_flow_pkg.BACKEND_RETRY_ATTEMPTS,
            )
        except chat_flow_pkg.PausedRetry:
            log.info("chat_paused_retry rid=%s", rid)
            # Not a backend failure: a pause re-runs the attempt with a fresh
            # retry budget.
            attempt = 0
            continue
        except chat_flow_pkg.BackendRetry as e:
            attempt += 1
            delay = chat_flow_pkg.backoff_seconds(
                attempt - 1, chat_flow_pkg.BACKEND_RETRY_BACKOFF_S
            )
            log.warning(
                "chat_backend_retry rid=%s attempt=%d/%d delay=%.2fs: %s",
                rid,
                attempt,
                chat_flow_pkg.BACKEND_RETRY_ATTEMPTS,
                delay,
                e,
            )
            if delay > 0:
                await asyncio.sleep(delay)


async def _prefetch_first_chunk(
    resp: httpx.Response, kill_token: KillToken
) -> tuple[AsyncIterator[bytes], bytes | None]:
    """Read the backend's first body chunk before the SSE response commits.

    For a stream the commit point is the first body byte, not the response
    headers: until something is sent the client has seen nothing, so a pause
    (or a kill) that lands during the prefill can abort the dispatch here and
    be handled before the client is given a 200 -- the retry loop re-runs the
    attempt, a kill becomes a plain 499. Returns the raw iterator (advanced
    past the first chunk) and that chunk, or `(iterator, None)` for an empty
    stream. On abort the upstream response is closed.
    """
    raw_iter = resp.aiter_raw()
    try:
        first = await chat_flow_pkg.race_kill(
            chat_flow_pkg.race_pause(raw_iter.__anext__()), kill_token
        )
    except StopAsyncIteration:
        return raw_iter, None
    except BaseException:
        try:
            await resp.aclose()
        except BaseException:
            # A second cancellation landing during the close must not mask the
            # abort: the original exception is what the caller handles (and a
            # real client disconnect is an abort too, so it still propagates).
            log.debug("prefetch_aclose_failed", exc_info=True)
        raise
    return raw_iter, first


async def _erase_stray(
    sm: SlotManager, g: GSlot | None, gate_passed: bool
) -> None:
    """Erase a served slot's KV before a retry, when the backend was reached.

    A failed attempt that had passed the backend gate may have left partial
    state on its slot (a half-written prefill from an aborted dispatch). The
    next attempt picks a slot of its own -- possibly this one -- so it must not
    inherit the garbage. Before the gate the backend was never reached, so there
    is nothing to erase (and erasing would only block up to CONNECT_TIMEOUT
    against a backend that is down). A failed erase is logged and ignored: it
    must not turn a retry into a failure.
    """
    if g is None or not gate_passed:
        return
    try:
        client = sm.backends[g[0]]["client"]
        await client.erase_slot(g[2], model=g[1])
    except Exception:
        log.debug("erase_before_retry_failed g=%s", g, exc_info=True)


async def _chat_once(
    sm: SlotManager,
    clients: list[LlamaClient],
    data: dict,
    rid: str,
    ts: str,
    t0: float,
    t0_mono: float,
    can_retry: bool,
) -> Response:
    """One attempt of the pipeline (re-run after a pause or backend failure)."""
    messages: list[dict] = data.get("messages") or []
    stream = bool(data.get("stream", False))
    # A client opts out of the KV cache entirely with cache_prompt:false:
    # such a request is proxied onto a free/oldest slot untouched -- no
    # restore search and no save of bin/meta. cache_write:false is the finer
    # control: the restore side stays on, only saving the result is off.
    # Neither flag participates in the cache key (they are not render-context
    # fields), so conversations that differ only by these flags share a key.
    no_cache = data.get("cache_prompt") is False
    cache_write = data.get("cache_write") is not False

    client_model = data.get("model") or None
    effective_model, no_cache = await chat_flow_pkg._resolve_effective_model(
        sm, clients, client_model, no_cache
    )

    # All request-side prefix values in one pass, off the event loop: the
    # words are tokenized once (blocks + word count share them) and the
    # per-message prefix hashes are computed incrementally (O(n), not O(n^2)).
    # The render-context leader (tools/reasoning params) is part of the prefix,
    # so differently-rendered conversations never share a key.
    render_ctx = chat_flow_pkg._render_ctx_of(data)
    prefix, key, blocks, prefix_hashes, n_words = await hs.request_prefix_values_async(
        messages, effective_model, chat_flow_pkg.WORDS_PER_BLOCK, chat_flow_pkg.REASONING_IN_KEY, render_ctx
    )
    is_big = n_words > chat_flow_pkg.BIG_THRESHOLD_WORDS
    # Only big requests get the cache treatment. A cache_prompt:false request
    # skips restore and save; a cache_write:false request keeps the restore
    # side and only skips the save.
    do_restore = is_big and not no_cache
    do_save = do_restore and cache_write

    # Live dashboard: register the request (no-op when the UI is off).
    ui_obs.req_start(
        rid,
        model=effective_model,
        stream=stream,
        n_words=n_words,
        is_big=is_big,
        key=key,
        messages=messages,
    )

    # Kill handle: from here on the request is addressable by its correlation
    # id, so it can be dropped from the slot queue or stopped mid-generation.
    # It is dropped again when the request is really over: the finally below,
    # or the stream reader for a streaming request (which outlives this call).
    kill_token = chat_flow_pkg.bind_kill(rid, effective_model)
    killed_flag = False

    def _req_outcome(outcome: str) -> None:
        promstats.requests_total.labels(
            model=metric_model, stream="true" if stream else "false", outcome=outcome
        ).inc()

    def _observe_latency() -> None:
        """Record duration/TTFT for a request that ends in a JSON error.

        These histograms used to be observed on the success path only, so every
        5xx was missing from them: a rate() over the duration under-reported
        failures, and a proxy that failed fast looked faster than one that
        succeeded slowly. No TTFT is recorded for an error -- no token was
        produced -- so only the duration goes in.
        """
        promstats.request_duration_seconds.labels(
            model=metric_model, stream="true" if stream else "false"
        ).observe(time.monotonic() - t0_mono)

    def _backend_retry(reason: str) -> None:
        """Count a proxy-issued retry (the attempt is about to be re-run)."""
        promstats.backend_retries_total.labels(
            model=metric_model, reason=reason
        ).inc()

    def _killed(
        slot: GSlot | None = None, used_key: str | None = None
    ) -> JSONResponse:
        """The answer for a request an operator killed (499).

        A kill that landed mid-generation drops the slot's "holds this key"
        record: llama.cpp kept every token it had already produced, so the
        slot no longer holds exactly the saved conversation, and a later
        same-key request would skip its restore (SKIP_RESTORE_SAME_SLOT) and
        continue on a wrong context. A kill in the queue never owned a slot
        and has nothing to invalidate.
        """
        nonlocal killed_flag
        killed_flag = True
        if slot is not None and used_key is not None:
            sm.forget_saved(slot, used_key)
        chat_flow_pkg.unregister_kill(rid)
        _req_outcome("killed")
        ui_obs.req_end(rid, status=ui_obs.STATUS_CANCELLED, error="killed")
        _observe_latency()
        reqlog.log_file("response", rid, ts, {"killed": True, "stage": kill_token.stage})
        log.warning("chat_killed rid=%s stage=%s slot=%s", rid, kill_token.stage, slot)
        return JSONResponse({"error": "killed"}, status_code=499)

    decision: dict = chat_flow_pkg._new_decision(
        is_big, no_cache, cache_write, n_words, effective_model, render_ctx
    )

    restore_key = await chat_flow_pkg._select_restore_candidate(
        prefix_hashes, blocks, effective_model, do_restore, no_cache, n_words, decision
    )

    log.info(
        "before_acquire is_big=%s restore_key=%s",
        is_big,
        restore_key[:16] if restore_key else None,
    )

    # Only a big request with a selected candidate can restore, so only it
    # needs a pending entry: a concurrent subsumed deletion of its key must be
    # able to substitute the replacement cache before the slot is acquired.
    if do_restore and restore_key is not None:
        chat_flow_pkg._register_pending_restore(restore_key)
    # Refresh the model's slot pools before picking a slot: a slot cut/reload on
    # the backend is otherwise only seen at the next periodic poll, and a stale
    # slot_id wraps onto a live physical slot (id % n_slots), risking a save/
    # restore collision. Rate-limited and non-fatal inside freshen_model.
    # Skip only for an alias that could not be resolved (chat_flow._model): the
    # name is proxied as-is without cache treatment, and refreshing it would
    # register a slot pool under the client's string, so the next request would
    # take the alias for a real model id and cache under it. A cache_prompt:false
    # request is a different case -- its name is real, only the reuse is off --
    # so it still refreshes, keeping the stale-slot_id guarantee.
    unresolved_alias = no_cache and client_model == effective_model
    # The metric label must stay a bounded set: a name the proxy could not
    # resolve is the client's own string, and every distinct value would mint a
    # permanent time series. The backend still receives the name it asked for.
    metric_model = promstats.UNRESOLVED_LABEL if unresolved_alias else effective_model
    if not unresolved_alias:
        await sm.freshen_model(effective_model)

    # Context-budget admission: the request must declare its token demand
    # (estimated prompt + generation) before it picks a slot, so the pool can
    # keep the sum of the in-flight demands within the backend's shared context
    # (llama.cpp --parallel N --kv-unified: the N slots share one -c context).
    # The client's requested cap (max_tokens or n_predict) bounds the generation
    # budget; MAX_TOKENS applies when it sends neither. With the feature off the
    # estimate (a CPU-bound tokenizer pass) is skipped entirely.
    client_cap: int | None = None
    for cap_key in ("max_tokens", "n_predict"):
        cap = data.get(cap_key)
        if isinstance(cap, int) and not isinstance(cap, bool) and cap > 0:
            client_cap = cap
            break
    gen_budget = (
        min(chat_flow_pkg.MAX_TOKENS, client_cap)
        if client_cap is not None
        else chat_flow_pkg.MAX_TOKENS
    )
    # Admission budgets a resolved model's pool; an unresolved alias has none:
    # it is proxied onto the ungated bootstrap slot without cache treatment, and
    # gating it would mint a per-alias context_reserved_tokens series (the same
    # cardinality leak the metric_model bucket below avoids). So it is not gated.
    admission_enabled = (
        chat_flow_pkg.CONTEXT_ADMISSION_ENABLED and not unresolved_alias
    )
    est_prompt: int | None = None
    need_tokens: int | None = None
    if admission_enabled:
        est_prompt = await hs.estimate_prompt_tokens_async(prefix)
        need_tokens = est_prompt + gen_budget
        promstats.estimated_prompt_tokens.labels(model=metric_model).observe(
            est_prompt
        )
    decision["admission"] = {
        "enabled": admission_enabled,
        "est_prompt_tokens": est_prompt,
        "gen_budget": gen_budget,
        "need_tokens": need_tokens,
    }

    t_acq = time.monotonic()
    try:
        try:
            g, _lock, restored, used_key = await asyncio.wait_for(
                chat_flow_pkg.race_kill(
                    chat_flow_pkg.race_pause(
                        sm.acquire_for_request(
                            effective_model,
                            restore_key if do_restore else None,
                            resolve_restore_key=chat_flow_pkg._resolve_restore_key if do_restore else None,
                            need_tokens=need_tokens,
                        ),
                    ),
                    kill_token,
                ),
                timeout=chat_flow_pkg.ACQUIRE_TIMEOUT,
            )
        except chat_flow_pkg.PausedRetry:
            # The pause aborted the wait before a slot was handed over (the
            # cancelled acquire owns any slot it had touched). Let the retry
            # loop re-run once the pause is over.
            raise
        except TimeoutError:
            log.error(
                "acquire_timeout is_big=%s restore_key=%s",
                is_big,
                restore_key[:16] if restore_key else None,
            )
            promstats.requests_total.labels(
                model=metric_model, stream="true" if stream else "false",
                outcome="acquire_timeout",
            ).inc()
            ui_obs.req_end(rid, status=ui_obs.STATUS_ERROR, error="acquire_timeout")
            _observe_latency()
            chat_flow_pkg.unregister_kill(rid)
            return JSONResponse(
                {"error": "all slots busy, please retry later"},
                status_code=503,
            )
        except chat_flow_pkg.QueueFullError:
            log.warning("queue_full model=%s", effective_model)
            promstats.requests_total.labels(
                model=metric_model, stream="true" if stream else "false",
                outcome="queue_full",
            ).inc()
            ui_obs.req_end(rid, status=ui_obs.STATUS_ERROR, error="queue_full")
            _observe_latency()
            chat_flow_pkg.unregister_kill(rid)
            return JSONResponse(
                {"error": "too many waiting requests, please retry later"},
                status_code=503,
            )
        except chat_flow_pkg.RequestKilled:
            killed_flag = True
            return _killed()
    finally:
        if do_restore and restore_key:
            chat_flow_pkg._unregister_pending_restore(restore_key)

    # The slot is released exactly once per request:
    # - successful stream: the reader task releases it (the generator owns
    #   the slot from this point on);
    # - successful big non-stream: the background save task releases it;
    # - every other path (a failed pre-dispatch step, errors, exceptions,
    #   cancellation): the finally at the end of this function.
    task_owns_slot = False
    # A streaming request stays killable after this function returns: the
    # reader task owns the slot from here on and unregisters the token itself.
    stream_handed_off = False
    # False until the backend gate passes: a kill that lands during the hold
    # must not touch the (by definition not serving) backend -- the erase
    # would only block up to CONNECT_TIMEOUT before the 499.
    gate_passed = False
    try:
        promstats.slot_wait_seconds.labels(model=metric_model).observe(
            time.monotonic() - t_acq
        )

        log.info("after_acquire g=%s key=%s restored=%s", g, key[:16], restored)

        be_id, _mid, slot_id = g
        client = clients[be_id]
        decision["restore"]["used_key"] = used_key
        decision["restore"]["outcome"] = restored
        decision["slot"] = {"backend": be_id, "model": effective_model, "id": slot_id}
        ui_obs.req_slot(rid, be_id, effective_model, slot_id)
        kill_token.stage = chat_flow_pkg.STAGE_GENERATING
        # Hold the connection while the picked backend is down or still
        # loading (bounded by HOLD_TIMEOUT) instead of bouncing the client
        # with 502 + Retry-After right away; the slot stays held for the
        # whole wait -- while the backend is down its slots are useless
        # anyway, and releasing would only move the wait into the acquire
        # queue. A kill during the hold unwinds through the RequestKilled
        # handler below (499).
        if not await chat_flow_pkg.race_pause(
            chat_flow_pkg.backend_gate(client, kill_token, sm)
        ):
            log.error("hold_timeout g=%s key=%s", g, key[:16])
            promstats.requests_total.labels(
                model=metric_model, stream="true" if stream else "false",
                outcome="hold_timeout",
            ).inc()
            ui_obs.req_end(rid, status=ui_obs.STATUS_ERROR, error="hold_timeout")
            _observe_latency()
            return chat_flow_pkg._error_response(
                {"error": "backend unavailable, please retry later"}, 502
            )
        gate_passed = True
        decision["slot_before_chat"] = await chat_flow_pkg._snapshot_slot(
            client, slot_id, effective_model
        )

        await chat_flow_pkg._settle_restore(
            slot_id, effective_model, restored, used_key, decision
        )

        body = dict(data)
        body.pop("cache_write", None)
        body["model"] = effective_model
        body["cache_prompt"] = bool(do_restore)
        body["n_keep"] = -1

        opts = dict(body.get("options") or {})
        opts.pop("cache_write", None)
        opts["slot_id"] = slot_id
        opts["id_slot"] = slot_id
        opts["n_keep"] = -1
        opts["cache_prompt"] = bool(do_restore)
        body["options"] = opts

        log.info(
            "dispatch be=%d slot=%d is_big=%s (restore_target=%s restored=%s model_id=%s)",
            be_id,
            slot_id,
            is_big,
            restore_key[:16] if restore_key else None,
            restored,
            effective_model,
        )

        if stream:
            resp = await chat_flow_pkg.race_kill(
                chat_flow_pkg.race_pause(
                    client.chat_completions(
                        body,
                        slot_id=slot_id,
                        stream=True,
                    ),
                ),
                kill_token,
            )
            if resp.status_code != 200:
                err_txt = await resp.aread()
                await resp.aclose()
                status = chat_flow_pkg._provider_error_status(resp.status_code)
                if can_retry and status == 502:
                    # A retryable upstream status: erase any partial state and
                    # let the wrapper re-dispatch instead of bouncing the client.
                    await _erase_stray(sm, g, gate_passed)
                    _backend_retry("upstream_5xx")
                    raise chat_flow_pkg.BackendRetry(f"backend {resp.status_code}")
                promstats.requests_total.labels(
                    model=metric_model, stream="true", outcome="error"
                ).inc()
                ui_obs.req_end(
                    rid,
                    status=ui_obs.STATUS_ERROR,
                    error=f"backend {resp.status_code}",
                )
                _observe_latency()
                return chat_flow_pkg._error_response(
                    {"error": err_txt.decode("utf-8", "ignore")},
                    status,
                )

            # The first backend chunk is read here, before the response is
            # committed: a pause during the prefill aborts the attempt (the
            # retry loop re-dispatches) instead of becoming a stream that has
            # to drain, and the client never sees headers for a stream that
            # was never produced.
            raw_iter, first_chunk = await _prefetch_first_chunk(resp, kill_token)
            gen = await chat_flow_pkg.start_stream_task(
                resp,
                g,
                key,
                prefix,
                blocks,
                effective_model,
                sm,
                do_save,
                prefix_hashes,
                clients,
                messages,
                rid,
                ts,
                decision,
                render_ctx=render_ctx,
                t0=t0_mono,
                metric_model=metric_model,
                kill_token=kill_token,
                used_key=used_key,
                raw_iter=raw_iter,
                first_chunk=first_chunk,
            )
            task_owns_slot = True
            stream_handed_off = True

            headers = {
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
            return StreamingResponse(
                gen,
                media_type="text/event-stream",
                headers=headers,
            )

        else:
            out = await chat_flow_pkg.race_kill(
                chat_flow_pkg.race_pause(
                    client.chat_completions(
                        body,
                        slot_id=slot_id,
                        stream=False,
                    ),
                ),
                kill_token,
            )
            if not isinstance(out, dict):
                _req_outcome("error")
                ui_obs.req_end(rid, status=ui_obs.STATUS_ERROR,
                              error="provider non-JSON body")
                _observe_latency()
                return chat_flow_pkg._error_response(
                    {"error": "provider non-JSON body"},
                    502,
                )

            # The client maps provider failures to {"object": "error", ...};
            # such a body must not be returned to the caller as HTTP 200.
            if out.get("object") == "error":
                log.error(
                    "provider_error key=%s status=%s message=%s",
                    key[:16],
                    out.get("status"),
                    out.get("message"),
                )
                status = chat_flow_pkg._provider_error_status(out.get("status"))
                if can_retry and status == 502:
                    # A transient provider 5xx: erase any partial state and let
                    # the wrapper re-dispatch instead of bouncing the client.
                    await _erase_stray(sm, g, gate_passed)
                    _backend_retry("upstream_5xx")
                    raise chat_flow_pkg.BackendRetry(
                        f"provider error {out.get('status')}"
                    )
                body = {"error": out.get("message") or "provider error"}
                if out.get("raw"):
                    body["raw"] = out["raw"]
                _req_outcome("error")
                ui_obs.req_end(rid, status=ui_obs.STATUS_ERROR,
                              error=str(out.get("message") or "provider error"))
                _observe_latency()
                return chat_flow_pkg._error_response(body, status)

            # The full backend body is the group's response.json (both sizes);
            # the assistant content is extracted for the prefix computation.
            response_text, response_reasoning, response_reasoning_field = (
                chat_flow_pkg._assistant_content(out)
            )
            reqlog.log_file("response", rid, ts, out)
            # Non-stream: the whole answer arrived at once — feed it to the
            # dashboard in a single batch (no incremental visibility exists).
            ui_obs.req_tokens(rid, response_text, response_reasoning)
            if isinstance(out.get("usage"), dict):
                ui_obs.req_usage(rid, out["usage"])
            ui_obs.req_ttft(rid, time.monotonic() - t0_mono)
            if do_save:
                # Save + meta + subsumed-meta cleanup (see the stream reader)
                # runs in the background: the .bin write must not delay the
                # JSON response. The task owns the slot and releases it, and
                # writes the group's prefix.json (reusing its saved values).
                save_task = asyncio.create_task(
                    chat_flow_pkg._background_save(
                        clients,
                        sm,
                        g,
                        key,
                        prefix,
                        blocks,
                        prefix_hashes,
                        effective_model,
                        messages,
                        response_text,
                        response_reasoning,
                        response_reasoning_field,
                        rid,
                        ts,
                        decision=decision,
                        render_ctx=render_ctx,
                    )
                )
                chat_flow_pkg._BG_SAVE_TASKS.add(save_task)
                save_task.add_done_callback(chat_flow_pkg._BG_SAVE_TASKS.discard)
                task_owns_slot = True
            else:
                # No background save for small requests: the saved
                # conversation (the next message's match target) is computed
                # in its own task so the response is not delayed.
                decision["save"] = {"attempted": False}
                chat_flow_pkg._emit_decision(decision, rid, ts)
                ptask = asyncio.create_task(
                    chat_flow_pkg._log_prefix_group(
                        rid,
                        ts,
                        messages,
                        effective_model,
                        response_text,
                        response_reasoning,
                        response_reasoning_field,
                        prefix,
                        blocks,
                        prefix_hashes,
                        render_ctx,
                    )
                )
                chat_flow_pkg._BG_SAVE_TASKS.add(ptask)
                ptask.add_done_callback(chat_flow_pkg._BG_SAVE_TASKS.discard)

            # Non-stream: the whole answer arrived at once, so the time to
            # first token equals the total duration.
            chat_flow_pkg._record_tokens(metric_model, out.get("usage"))
            dur = time.monotonic() - t0_mono
            promstats.request_duration_seconds.labels(
                model=metric_model, stream="false"
            ).observe(dur)
            promstats.ttft_seconds.labels(
                model=metric_model, stream="false"
            ).observe(dur)
            _req_outcome("ok")
            log.info(
                "json_done g=%s key=%s is_big=%s dur_ms=%d",
                g,
                key[:16],
                is_big,
                int((time.time() - t0) * 1000),
            )
            ui_obs.req_end(rid)
            return JSONResponse(content=out, status_code=200)

    except chat_flow_pkg.RequestKilled:
        # The request was killed between the slot acquisition and the answer.
        # The finally below gives the slot back; nothing is saved (a partial
        # KV cache is useless for restore and only wastes disk).
        killed_flag = True
        try:
            if g is not None and gate_passed:
                client = sm.backends[g[0]]["client"]
                await client.erase_slot(g[2], model=g[1])
        except Exception:
            log.debug("erase_after_kill_failed g=%s", g, exc_info=True)
        return _killed(g, used_key)
    except chat_flow_pkg.PausedRetry:
        # A pause landed after the slot was acquired but before the response
        # was committed: unwind so the wrapper can re-dispatch. The finally
        # below gives the slot back (no save), and no outcome is recorded --
        # the retry records the real one.
        raise
    except chat_flow_pkg.BackendRetry:
        # A retryable backend failure before the response committed (raised by
        # the branches above, or by the transport handler below): unwind so the
        # wrapper can re-dispatch. The finally gives the slot back, no save, and
        # no outcome is recorded -- the retry records the real one.
        raise
    except httpx.HTTPError as e:
        # A transport failure with no committed response: retry it only while
        # the budget lasts and only when it provably happened before the backend
        # acted -- a refused or timed-out connect, a pool timeout, or a stale
        # pooled connection ("Server disconnected without sending a response",
        # the common case) -- else report the genuine upstream failure (502 +
        # Retry-After). A read/write timeout or a connection reset after the
        # request was sent means generation may have run, and a retry would just
        # recompute the answer.
        if can_retry and chat_flow_pkg.is_retryable_transport_error(e):
            await _erase_stray(sm, g, gate_passed)
            _backend_retry("transport")
            raise chat_flow_pkg.BackendRetry(str(e)) from e
        log.exception("chat_upstream_error g=%s key=%s", g, key[:16])
        _req_outcome("error")
        ui_obs.req_end(rid, status=ui_obs.STATUS_ERROR, error=str(e))
        _observe_latency()
        return chat_flow_pkg._error_response({"error": str(e)}, 502)
    except Exception as e:
        log.exception("chat_error g=%s key=%s", g, key[:16])
        _req_outcome("error")
        ui_obs.req_end(rid, status=ui_obs.STATUS_ERROR, error=str(e))
        _observe_latency()
        return chat_flow_pkg._error_response({"error": str(e)}, 500)
    finally:
        if not stream_handed_off:
            chat_flow_pkg.unregister_kill(rid)
        if not task_owns_slot:
            log.info("slot_release g=%s key=%s via=finally", g, key[:16])
            sm.release(g)
            # Every path that did not hand the slot to a save/reader task ends
            # here: error returns (provider failure, upstream error, exception)
            # now persist the restore-phase diagnostics too — the very
            # decision.json needed to debug failures. Save was never attempted,
            # so mark it explicitly; _emit_decision is idempotent and no-ops
            # when this request already emitted one.
            decision.setdefault("save", {"attempted": False})
            chat_flow_pkg._emit_decision(decision, rid, ts)
