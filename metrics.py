# metrics.py

# -*- coding: utf-8 -*-

"""
Aggregate per-model backend /metrics into a single Prometheus target.

The proxy exposes GET /metrics as a valid Prometheus scrape target: for each
backend it discovers the active (loaded) models, fetches each model's
/metrics?model=X, adds model and backend labels to every metric line, and
returns the merged body. # HELP/# TYPE lines are deduplicated by metric name.
Partial failures (a down backend or a failed per-model fetch) are tolerated:
the remaining models are still reported.
"""

import asyncio
import logging

from llama_client import LlamaClient

log = logging.getLogger(__name__)


def _escape(value: str) -> str:
    """Escape a Prometheus label value (backslash, double quote, newline)."""
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _split_labels(block: str) -> list[tuple[str, str]]:
    """Split a Prometheus label block (a="b",c="d") into (name, quoted_value).

    Quote-aware: a comma inside a quoted value does not split labels. The value
    is kept raw (still quoted) so it can be re-emitted unchanged.
    """
    out: list[tuple[str, str]] = []
    i, n = 0, len(block)
    while i < n:
        while i < n and block[i] in " \t":
            i += 1
        if i >= n:
            break
        eq = block.find("=", i)
        if eq < 0:
            break
        name = block[i:eq].strip()
        j = eq + 1
        while j < n and block[j] in " \t":
            j += 1
        if j < n and block[j] == '"':
            k = j + 1
            while k < n:
                if block[k] == "\\":
                    k += 2
                    continue
                if block[k] == '"':
                    break
                k += 1
            quoted = block[j : k + 1]
            i = k + 1
        else:
            k = j
            while k < n and block[k] != ",":
                k += 1
            quoted = '"' + block[j:k].strip() + '"'
            i = k
        out.append((name, quoted))
        if i < n and block[i] == ",":
            i += 1
    return out


def _metric_name(line: str) -> str:
    """The metric name of a sample line (up to the first { or whitespace)."""
    i = 0
    n = len(line)
    while i < n and line[i] not in "{ \t":
        i += 1
    return line[:i]


def _merge_labels(block: str, labels: dict[str, str]) -> str:
    """Merge new labels into an existing label block; emit sorted by name.

    An existing label of the same name is replaced. Sorting gives a
    deterministic output regardless of the backend's label order.
    """
    existing = _split_labels(block) if block.strip() else []
    d = {name: quoted for name, quoted in existing}
    for k, v in labels.items():
        d[k] = f'"{_escape(v)}"'
    return ",".join(f"{k}={d[k]}" for k in sorted(d))


def _relabel_line(line: str, labels: dict[str, str]) -> str:
    """Add the given labels to a single sample line (replacing any existing)."""
    name = _metric_name(line)
    i = len(name)
    if i < len(line) and line[i] == "{":
        depth = 0
        j = i
        while j < len(line):
            if line[j] == "{":
                depth += 1
            elif line[j] == "}":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        block = line[i + 1 : j]
        tail = line[j + 1 :].strip()
    else:
        block = ""
        tail = line[i:].strip()
    return f"{name}{{{_merge_labels(block, labels)}}} {tail}"


def _collect_comment(line: str, comments: dict[str, dict[str, str]]) -> None:
    """Record a # HELP / # TYPE line under its metric name (drop # EOF etc.)."""
    parts = line.split(None, 2)
    if len(parts) < 3 or parts[1] not in ("HELP", "TYPE"):
        return
    kind = parts[1].lower()
    rest = parts[2]
    name = rest.split(None, 1)[0] if rest else ""
    if not name:
        return
    comments.setdefault(name, {})[kind] = rest


def relabel(
    raw: str, labels: dict[str, str]
) -> tuple[list[str], dict[str, dict[str, str]]]:
    """Relabel raw Prometheus text with the given labels.

    Returns (sample_lines, comments): every sample line carries the new labels
    (an existing label of the same name is replaced) and comments maps metric
    name -> {"help": ..., "type": ...} for the # HELP/# TYPE lines. Blank
    lines and non-HELP/TYPE comments (# EOF) are dropped.
    """
    lines: list[str] = []
    comments: dict[str, dict[str, str]] = {}
    for raw_line in raw.splitlines():
        s = raw_line.strip()
        if not s:
            continue
        if s.startswith("#"):
            _collect_comment(s, comments)
            continue
        lines.append(_relabel_line(s, labels))
    return lines, comments


def merge(parts: list[tuple[list[str], dict[str, dict[str, str]]]]) -> str:
    """Combine per-model relabeled parts into one valid Prometheus body.

    # HELP/# TYPE are deduplicated by metric name (first wins); sample lines
    are grouped under their metric, each preceded by its (deduped) comments.
    Returns an empty string when there is nothing to report.
    """
    lines_by_name: dict[str, list[str]] = {}
    comments: dict[str, dict[str, str]] = {}
    for lines, com in parts:
        for line in lines:
            lines_by_name.setdefault(_metric_name(line), []).append(line)
        for name, entry in com.items():
            merged = comments.setdefault(name, {})
            for kind, body in entry.items():
                merged.setdefault(kind, body)
    out: list[str] = []
    for name in sorted(set(lines_by_name) | set(comments)):
        entry = comments.get(name, {})
        if "help" in entry:
            out.append(f"# HELP {entry['help']}")
        if "type" in entry:
            out.append(f"# TYPE {entry['type']}")
        out.extend(lines_by_name.get(name, []))
    return ("\n".join(out) + "\n") if out else ""


async def collect(
    clients: list[LlamaClient], model: str | None = None
) -> str:
    """Fetch /metrics for every active model across all backends; merged text.

    Each sample line is labelled with model and backend (the backend index).
    When `model` is given, only that model's metrics are included. Partial
    failures (a down backend or a failed per-model fetch) are skipped and
    logged, never failing the whole scrape.
    """

    async def client_parts(be_id: int, client: LlamaClient) -> list:
        models = await client.get_active_models()
        if model is not None:
            models = [m for m in models if m == model]
        if not models:
            return []
        raws = await asyncio.gather(
            *(client.get_metrics(m) for m in models), return_exceptions=True
        )
        parts: list = []
        for m, raw in zip(models, raws):
            if isinstance(raw, BaseException):
                log.warning(
                    "metrics_fetch_fail backend=%d model=%s: %s", be_id, m, raw
                )
                continue
            if not raw:
                continue
            parts.append(relabel(raw, {"model": m, "backend": str(be_id)}))
        return parts

    results = await asyncio.gather(
        *(client_parts(i, c) for i, c in enumerate(clients)),
        return_exceptions=True,
    )
    flat: list = []
    for r in results:
        if isinstance(r, BaseException):
            log.warning("metrics_client_fail: %s", r)
        else:
            flat.extend(r)
    return merge(flat)
