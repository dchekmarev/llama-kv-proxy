# reqlog.py

"""Request/response/prefix logging with group rotation.

Every /v1/chat/completions request produces one group of JSON files in
REQUEST_LOG_DIR, all sharing the {timestamp_ms}.{request_id} prefix:

- {ts}.{rid}.request.json   the client request body as received
- {ts}.{rid}.response.json  the backend response: the JSON body for
                            non-stream, or the assembled stream (content,
                            reasoning, completion status, error) for streams,
                            plus the final SSE usage/timings chunk when the
                            backend sent one
- {ts}.{rid}.prefix.json    the conversation prefix (prompt + assistant
                            response) that the next message's request will be
                            matched against, plus its cache key
- {ts}.{rid}.raw.json       (stream only) the raw SSE stream as received
- {ts}.{rid}.decision.json  the proxy's per-request cache decision: big/small,
                            restore candidate + outcome, erase, slot KV state
                            after restore, and the save result

Only the newest REQUEST_LOG_MAX_GROUPS groups are kept: after every write the
directory is rescanned and the oldest groups (all their files) are deleted.
Writes are fire-and-forget background tasks (strongly referenced, like the
reader/save tasks in chat_flow) so logging never adds latency to the
response; every failure is logged and swallowed, never propagated.
"""

import asyncio
import json
import logging
import os
import re
import tempfile
import time

from config import REQUEST_LOG_DIR, REQUEST_LOG_MAX_GROUPS

log = logging.getLogger(__name__)

# {timestamp_ms}.{request_id}.{type}.json — the group is the ts.rid prefix.
_GROUP_RE = re.compile(r"^(\d+)\.([^.]+)\.(request|response|prefix|raw|decision)\.json$")

# Strong references to in-flight log-write tasks (the event loop keeps only
# weak references; without this a task could be GC-collected before it runs
# and the file would never land).
_LOG_TASKS: "set[asyncio.Task]" = set()


def new_group_ts() -> str:
    """Millisecond timestamp for a new group: sortable, collision-safe."""
    return str(int(time.time() * 1000))


def _atomic_write_json(path: str, payload: object) -> None:
    """Atomically write JSON to path: a temp file in the same directory plus
    os.replace, so a crash mid-write never leaves a truncated/corrupt target.
    The temp file is removed if the write fails."""
    fd, tmp = tempfile.mkstemp(
        dir=os.path.dirname(path), prefix=".reqlog-", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


def _rotate() -> None:
    """Delete the oldest groups beyond REQUEST_LOG_MAX_GROUPS.

    Files not matching the group naming are left untouched. A group being
    written is always the newest, so rotation never deletes in-flight files.
    """
    if REQUEST_LOG_MAX_GROUPS <= 0:
        return
    try:
        names = os.listdir(REQUEST_LOG_DIR)
    except FileNotFoundError:
        return
    groups: dict[tuple[int, str], list[str]] = {}
    for name in names:
        m = _GROUP_RE.match(name)
        if m:
            groups.setdefault((int(m.group(1)), m.group(2)), []).append(name)
    if len(groups) <= REQUEST_LOG_MAX_GROUPS:
        return
    for _group, files in sorted(groups.items())[:-REQUEST_LOG_MAX_GROUPS]:
        for name in files:
            try:
                os.remove(os.path.join(REQUEST_LOG_DIR, name))
            except FileNotFoundError:
                pass
    log.debug("reqlog_rotated kept=%d", REQUEST_LOG_MAX_GROUPS)


def _write(ftype: str, rid: str, ts: str, payload: object) -> None:
    path = os.path.join(REQUEST_LOG_DIR, f"{ts}.{rid}.{ftype}.json")
    _atomic_write_json(path, payload)
    _rotate()


async def _safe_write(ftype: str, rid: str, ts: str, payload: object) -> None:
    try:
        await asyncio.to_thread(_write, ftype, rid, ts, payload)
    except Exception as e:  # noqa: BLE001
        log.warning("reqlog_write_fail type=%s rid=%s: %s", ftype, rid, e)


def log_file(ftype: str, rid: str, ts: str, payload: object) -> None:
    """Schedule one file of a request group for writing (fire-and-forget).

    Never raises and never blocks the caller: the write and the rotation run
    in a background task. A no-op when logging is disabled (empty dir) or the
    group id is missing.
    """
    if not REQUEST_LOG_DIR or not rid or not ts:
        return
    task = asyncio.create_task(_safe_write(ftype, rid, ts, payload))
    _LOG_TASKS.add(task)
    task.add_done_callback(_LOG_TASKS.discard)
