# promstats.py

"""Proxy-level Prometheus metrics (llama_kv_proxy_*).

A dedicated CollectorRegistry keeps the proxy metrics isolated from the
default registry and from the backend llama_server_* text that is merged
into /metrics. Every metric name is prefixed with llama_kv_proxy_ so the
two halves of the scrape never collide.

Label cardinality is bounded by design: model, backend index, stream flag,
outcome, tier, reason and state are all small fixed sets. Request ids and
cache keys never appear in labels.
"""

import glob
import os
import re

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

from config import BIN_CACHE_DIR, META_DIR

REGISTRY = CollectorRegistry()

P = "llama_kv_proxy"

REQUEST_BUCKETS = (0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 300, 600)
TTFT_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30)
SLOT_WAIT_BUCKETS = (0.001, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 30, 60)
SAVE_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1, 2.5, 10, 30, 120)
RATIO_BUCKETS = (0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0)

requests_total = Counter(
    f"{P}_requests_total",
    "Chat requests handled by the proxy, by outcome.",
    ["model", "stream", "outcome"],
    registry=REGISTRY,
)
request_duration_seconds = Histogram(
    f"{P}_request_duration_seconds",
    "Total chat request duration (start to response end).",
    ["model", "stream"],
    buckets=REQUEST_BUCKETS,
    registry=REGISTRY,
)
ttft_seconds = Histogram(
    f"{P}_ttft_seconds",
    "Time to first token (request start to first backend chunk).",
    ["model", "stream"],
    buckets=TTFT_BUCKETS,
    registry=REGISTRY,
)
tokens_total = Counter(
    f"{P}_tokens_total",
    "Tokens reported by the backend, by kind (prompt/completion/cached).",
    ["model", "kind"],
    registry=REGISTRY,
)
restore_total = Counter(
    f"{P}_restore_total",
    "Restore candidate selection results for big requests.",
    ["model", "outcome"],
    registry=REGISTRY,
)
restore_ratio = Histogram(
    f"{P}_restore_ratio",
    "Fraction of the request prompt covered by the restored cache.",
    ["model"],
    buckets=RATIO_BUCKETS,
    registry=REGISTRY,
)
slot_wait_seconds = Histogram(
    f"{P}_slot_wait_seconds",
    "Time spent acquiring a backend slot.",
    ["model"],
    buckets=SLOT_WAIT_BUCKETS,
    registry=REGISTRY,
)
meta_files = Gauge(
    f"{P}_meta_files", "Meta files on disk.", registry=REGISTRY
)
meta_bytes = Gauge(
    f"{P}_meta_bytes", "Total size of meta files on disk.", registry=REGISTRY
)
bin_bytes = Gauge(
    f"{P}_bin_bytes", "Total size of .bin cache files on disk.", registry=REGISTRY
)
saves_total = Counter(
    f"{P}_saves_total",
    "KV cache saves attempted after big requests, by outcome.",
    ["model", "outcome"],
    registry=REGISTRY,
)
save_duration_seconds = Histogram(
    f"{P}_save_duration_seconds",
    "Backend slot save duration.",
    ["model"],
    buckets=SAVE_BUCKETS,
    registry=REGISTRY,
)
slots_total = Gauge(
    f"{P}_slots_total",
    "Backend slots by state (from /slots polling).",
    ["backend", "model", "state"],
    registry=REGISTRY,
)
stuck_slot_erases_total = Counter(
    f"{P}_stuck_slot_erases_total",
    "Slots erased by the stuck-slot watchdog.",
    ["backend", "model"],
    registry=REGISTRY,
)
backend_up = Gauge(
    f"{P}_backend_up",
    "Backend reachability from the slot poll (1 up, 0 down).",
    ["backend"],
    registry=REGISTRY,
)
evictions_total = Counter(
    f"{P}_evictions_total",
    "Cache entries deleted, by reason.",
    ["reason"],
    registry=REGISTRY,
)
restore_tier_total = Counter(
    f"{P}_restore_tier_total",
    "Restore hits by search tier (index / t1_disk / t2_blocks).",
    ["tier"],
    registry=REGISTRY,
)
stale_meta_drops_total = Counter(
    f"{P}_stale_meta_drops_total",
    "Stale metas dropped after a restore reported the .bin missing.",
    ["model"],
    registry=REGISTRY,
)
inflight_save_waits_total = Counter(
    f"{P}_inflight_save_waits_total",
    "Continuations that waited for an in-flight save, by result.",
    ["model", "result"],
    registry=REGISTRY,
)
backend_scrape_failures_total = Counter(
    f"{P}_backend_scrape_failures_total",
    "Backend /metrics scrape failures.",
    ["backend"],
    registry=REGISTRY,
)

# Valid restore-search tiers (restore_tier_total label values).
TIERS = ("index", "t1_disk", "t2_blocks")


def reset() -> None:
    """Clear every metric (test isolation)."""
    for m in (
        requests_total,
        tokens_total,
        restore_total,
        saves_total,
        stuck_slot_erases_total,
        evictions_total,
        restore_tier_total,
        stale_meta_drops_total,
        inflight_save_waits_total,
        backend_scrape_failures_total,
    ):
        m.clear()
    for m in (
        request_duration_seconds,
        ttft_seconds,
        restore_ratio,
        slot_wait_seconds,
        save_duration_seconds,
    ):
        m.clear()
    for g in (meta_files, meta_bytes, bin_bytes, slots_total, backend_up):
        g.clear()


def counter_sum(metric: Counter, **labels: str) -> float:
    """Sum a counter's samples matching all label filters (public API).

    Used to expose the aggregate hit/miss/tier counts in /cache/stats.
    """
    total = 0.0
    for m in REGISTRY.collect():
        if m.name != metric._name or m.type != "counter":
            continue
        for s in m.samples:
            # Skip the *_created timestamp samples prometheus_client emits
            # alongside every counter.
            if not s.name.endswith("_total"):
                continue
            if all(s.labels.get(k) == v for k, v in labels.items()):
                total += s.value
    return total


def _scan_dir(dirpath: str) -> tuple[int, int]:
    """(file count, total bytes) of regular files in dirpath."""
    n = 0
    total = 0
    if not dirpath:
        return n, total
    for path in glob.glob(os.path.join(dirpath, "*")):
        if not os.path.isfile(path):
            continue
        try:
            total += os.path.getsize(path)
            n += 1
        except OSError:
            continue
    return n, total


_MODEL_LABEL = re.compile(r'model="((?:[^"\\]|\\.)*)"')


def render(model: str | None = None) -> str:
    """The registry as Prometheus text.

    When `model` is given, sample lines carrying a different model label are
    dropped (unlabeled lines and comment lines are kept), mirroring the
    backend-metrics filter of the /metrics endpoint.
    """
    text = generate_latest(REGISTRY).decode()
    if model is None:
        return text
    out: list[str] = []
    for line in text.splitlines():
        if line.startswith("#"):
            out.append(line)
            continue
        m = _MODEL_LABEL.search(line)
        if m is None or m.group(1) == model:
            out.append(line)
    return "\n".join(out) + "\n"


def refresh_storage_gauges() -> None:
    """Set the meta/bin storage gauges from a full disk scan.

    Synchronous (disk I/O): callers run it in a worker thread.
    """
    n, total = _scan_dir(META_DIR)
    meta_files.set(n)
    meta_bytes.set(total)
    bin_bytes.set(_scan_dir(BIN_CACHE_DIR)[1])
