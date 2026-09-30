# tests/fake_backend.py

"""Fake llama.cpp backend for stress tests (no real LLM).

A real HTTP server (uvicorn) that mimics the llama.cpp endpoints the proxy
uses: /v1/models, /slots, POST /slots/{id} (save/restore/erase) and
POST /v1/chat/completions (stream + non-stream). Chat requests are answered
after a configurable delay (default 20 s) so the proxy's slot holding,
queueing and cleanup paths are exercised under realistic timing.

Fault injection via POST /_control (JSON):
  delay               seconds before the first byte / full answer
  mode                "ok" | "error" | "hang" | "drop_mid_stream"
  status_code         HTTP code for mode="error" (default 500)
  chunk_interval      seconds between SSE chunks (streaming)
  n_chunks            number of SSE chunks before [DONE]
  drop_after_chunks   mode="drop_mid_stream": kill the connection after N chunks
  slot_actions_fail   make save/restore/erase return 500

Diagnostics via GET /_stats:
  in_flight           chat requests currently being served
  total               chat requests started
  slot_violations     times two requests were in flight on the same slot
                      concurrently (a proxy bug: slots must be exclusive)
  max_in_flight       peak concurrent chat requests
  mode / delay        current fault configuration

Run:  python tests/fake_backend.py --port 9101 --n-slots 2
"""

import argparse
import asyncio
import json
import logging
import time

from fastapi import FastAPI, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse

log = logging.getLogger("fake_backend")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

app = FastAPI()

CFG = {
    "delay": 20.0,
    "mode": "ok",
    "status_code": 500,
    "chunk_interval": 0.5,
    "n_chunks": 20,
    "drop_after_chunks": 3,
    "slot_actions_fail": False,
}

N_SLOTS = 2
MODEL_ID = "fake-model"

_stats = {
    "in_flight": 0,
    "total": 0,
    "slot_violations": 0,
    "max_in_flight": 0,
    "slot_actions": 0,
}
# slot_id -> number of chat requests currently in flight on that slot
_slot_inflight: dict[int, int] = {}
# slot_id -> set of request ids still "live" on that slot. A request leaves
# the set when it completes OR when its client disconnects (the proxy has
# already released the slot by then and may legally dispatch a new request
# while this backend is still cleaning up). Only two *live* requests on one
# slot is a real proxy bug.
_slot_live: dict[int, set[int]] = {}
_req_counter = 0


def _start_disconnect_watcher(req: Request) -> tuple[asyncio.Future, asyncio.Task]:
    """Resolve `fut` as soon as the ASGI http.disconnect message arrives.

    A real backend (llama.cpp) frees the slot the moment it notices the
    client went away. Without reacting to the disconnect immediately, the
    fake only finds out when the next chunk write fails (up to
    `chunk_interval` later), which fakes a same-slot overlap that the proxy
    cannot avoid (it releases the slot as soon as *it* sees the client go).
    Returns (future, task); cancel the task when the request completes.
    """
    loop = asyncio.get_running_loop()
    fut: asyncio.Future = loop.create_future()

    async def _watch() -> None:
        try:
            while not fut.done():
                message = await req.receive()
                if message.get("type") == "http.disconnect":
                    if not fut.done():
                        fut.set_result(True)
                    return
        except Exception:  # noqa: BLE001
            if not fut.done():
                fut.set_result(False)

    task = loop.create_task(_watch())
    return fut, task


async def _sleep_or_disconnect(disc: asyncio.Future, seconds: float) -> bool:
    """Sleep up to `seconds`; return True early if the client disconnected."""
    end = time.monotonic() + seconds
    while True:
        if disc.done():
            return disc.result()
        remaining = end - time.monotonic()
        if remaining <= 0:
            return False
        try:
            return await asyncio.wait_for(asyncio.shield(disc), timeout=remaining)
        except asyncio.TimeoutError:
            continue


class MidStreamDrop(Exception):
    """Raised inside the stream generator to kill the TCP connection."""


@app.post("/_control")
async def control(req: Request):
    data = await req.json()
    for k, v in data.items():
        if k in CFG:
            CFG[k] = v
    return {"cfg": CFG}


@app.get("/_stats")
async def stats():
    return {**_stats, "cfg": CFG}


@app.get("/v1/models")
async def models():
    listing = {"data": [{"id": MODEL_ID}]}
    return listing


@app.get("/slots")
async def slots(model: str | None = Query(default=None)):
    return [
        {
            "id": i,
            "state": "busy" if _slot_inflight.get(i, 0) > 0 else "free",
            "n_ctx": 4096,
            "total_tokens": 0,
        }
        for i in range(N_SLOTS)
    ]


@app.post("/slots/{slot_id}")
async def slot_action(slot_id: int, action: str = Query(default=""), req: Request = None):
    if CFG["slot_actions_fail"]:
        _stats["slot_actions"] += 1
        return JSONResponse({"error": "forced failure"}, status_code=500)
    _stats["slot_actions"] += 1
    return {"status": "ok", "action": action, "slot_id": slot_id}


def _sse_chunk(idx: int, text: str) -> bytes:
    payload = json.dumps(
        {
            "id": f"gen-{idx}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": MODEL_ID,
            "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
        }
    )
    return f"data: {payload}\n\n".encode()


def _non_stream_body() -> dict:
    return {
        "id": "chatcmpl-fake",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": MODEL_ID,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "fake answer " * 20},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
    }


@app.post("/v1/chat/completions")
async def chat(req: Request):
    body = await req.json()
    stream = bool(body.get("stream", False))
    slot_id = None
    opts = body.get("options") or {}
    for src in (opts, body):
        for k in ("slot_id", "id_slot"):
            if isinstance(src.get(k), int):
                slot_id = src[k]
                break
        if slot_id is not None:
            break

    global _req_counter
    req_id = _req_counter
    _req_counter += 1

    _stats["total"] += 1
    _stats["in_flight"] += 1
    _stats["max_in_flight"] = max(_stats["max_in_flight"], _stats["in_flight"])
    if slot_id is not None:
        live = _slot_live.setdefault(slot_id, set())
        if live:
            _stats["slot_violations"] += 1
            log.warning(
                "SLOT_VIOLATION slot=%d new_req=%d live_reqs=%s",
                slot_id,
                req_id,
                sorted(live),
            )
        live.add(req_id)
        _slot_inflight[slot_id] = _slot_inflight.get(slot_id, 0) + 1

    disc_fut, disc_task = _start_disconnect_watcher(req)

    def _unlive():
        # Request is no longer "live" on its slot: either it completed or its
        # client disconnected (the proxy released the slot at disconnect).
        if slot_id is not None:
            live = _slot_live.get(slot_id)
            if live is not None:
                live.discard(req_id)
                if not live:
                    _slot_live.pop(slot_id, None)

    def _done():
        _stats["in_flight"] -= 1
        if slot_id is not None:
            _slot_inflight[slot_id] = max(0, _slot_inflight.get(slot_id, 0) - 1)
        _unlive()
        if not disc_task.done():
            disc_task.cancel()

    mode = CFG["mode"]

    if mode == "hang":
        async def hang_stream():
            try:
                yield b""
                while not await _sleep_or_disconnect(disc_fut, 1.0):
                    pass
            finally:
                _done()

        return StreamingResponse(hang_stream(), media_type="text/event-stream")

    if mode == "error":
        _done()
        return JSONResponse(
            {"error": "forced backend error"}, status_code=int(CFG["status_code"])
        )

    if mode == "drop_mid_stream":
        async def drop_stream():
            try:
                for i in range(int(CFG["drop_after_chunks"])):
                    yield _sse_chunk(i, f"tok{i} ")
                    if await _sleep_or_disconnect(disc_fut, CFG["chunk_interval"]):
                        _unlive()
                        return
                raise MidStreamDrop()
            finally:
                _done()

        return StreamingResponse(drop_stream(), media_type="text/event-stream")

    # mode == "ok"
    if stream:
        async def ok_stream():
            try:
                if await _sleep_or_disconnect(disc_fut, CFG["delay"]):
                    _unlive()
                    return
                for i in range(int(CFG["n_chunks"])):
                    yield _sse_chunk(i, f"tok{i} ")
                    if i < int(CFG["n_chunks"]) - 1 and await _sleep_or_disconnect(
                        disc_fut, CFG["chunk_interval"]
                    ):
                        _unlive()
                        return
                yield b"data: [DONE]\n\n"
            finally:
                _done()

        return StreamingResponse(ok_stream(), media_type="text/event-stream")

    if await _sleep_or_disconnect(disc_fut, CFG["delay"]):
        _done()
        return JSONResponse({"error": "client disconnected"}, status_code=499)
    _done()
    return _non_stream_body()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=9101)
    ap.add_argument("--n-slots", type=int, default=2)
    args = ap.parse_args()
    global N_SLOTS
    N_SLOTS = args.n_slots
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
