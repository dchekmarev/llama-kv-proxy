# AGENTS.md — llama-kv-proxy

## Verification (MANDATORY before calling a task done)

Run all three; CI (`.github/workflows/ci.yml`) runs the same on every push/PR:

    ruff check .
    python -m mypy app/ backend/ cache/ chat_flow/ core/ hashing/ obs/
    python -m pytest tests/ -q

Dev deps: `pip install -r requirements.txt -r requirements-dev.txt mypy`.

## Architecture (do not break)

- Layering — the layer below never imports the one above:
  `core` <- `backend` / `cache` / `hashing` <- `chat_flow` / `obs` <- `app`.
- Import by package name (`from core import config`), never a flat top-level module.
- Single uvicorn worker only: the slot manager keeps per-process state (locks, LRU
  marks); multiple workers route two requests to the same slot.

## Documentation (MANDATORY)

The README is the source of truth. A change touching env vars, endpoints, file/dir
layout, schemas or behavior is incomplete until the README is updated in the same
change — no "update docs later".

1. Dry, no water — facts, numbers, commands; no filler, no emoji, no "in this document…".
2. Point form over prose — a rule that fits on one line does not get a paragraph.
3. No duplication — one source of truth per fact; other places link to it. A doc that
   contradicts a doc is a bug: fix both in one change.
4. Delete, don't annotate — stale text is removed, not left next to its replacement.
5. Verify claims — every command/path/default in a doc is checked against the code, not memory.
