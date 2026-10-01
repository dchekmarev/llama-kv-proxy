# tests/test_promstats.py

"""promstats.py: proxy-level Prometheus metrics (llama_kv_proxy_*).

- reset() clears every metric (test isolation, also autouse in conftest);
- counter_sum() sums counter samples by label filter (backs /cache/stats);
- render() emits the registry as text, optionally filtered by model label;
- refresh_storage_gauges() sets the meta/bin gauges from a full disk scan;
- the /metrics endpoint serves proxy metrics followed by the backend text.
"""

from unittest.mock import AsyncMock, MagicMock

import app as app_module
import chat_flow
import promstats

# --- counter_sum -------------------------------------------------------------


def test_counter_sum_filters_by_labels():
    c = promstats.restore_total
    c.labels(model="m1", outcome="hit").inc()
    c.labels(model="m1", outcome="hit").inc()
    c.labels(model="m1", outcome="miss").inc()
    c.labels(model="m2", outcome="hit").inc()
    assert promstats.counter_sum(c, model="m1", outcome="hit") == 2.0
    assert promstats.counter_sum(c, model="m1") == 3.0
    assert promstats.counter_sum(c, outcome="miss") == 1.0
    assert promstats.counter_sum(c) == 4.0


def test_counter_sum_zero_when_absent():
    assert promstats.counter_sum(promstats.restore_total, model="nope") == 0.0


# --- render ------------------------------------------------------------------


def test_render_contains_all_metrics():
    promstats.requests_total.labels(model="m1", stream="true", outcome="ok").inc()
    promstats.meta_files.set(3)
    text = promstats.render()
    # prometheus_client emits labels in alphabetical order.
    assert (
        'llama_kv_proxy_requests_total{model="m1",outcome="ok",stream="true"} 1.0'
        in text
    )
    assert "llama_kv_proxy_meta_files 3.0" in text


def test_render_filters_by_model_label():
    promstats.requests_total.labels(model="m1", stream="true", outcome="ok").inc()
    promstats.requests_total.labels(model="m2", stream="true", outcome="ok").inc()
    promstats.meta_files.set(3)

    text = promstats.render(model="m1")
    assert 'model="m1"' in text
    assert 'model="m2"' not in text
    # Unlabeled gauges and comment lines survive the filter.
    assert "llama_kv_proxy_meta_files 3.0" in text
    assert "# HELP llama_kv_proxy_meta_files" in text


# --- refresh_storage_gauges ---------------------------------------------------


def test_refresh_storage_gauges_scans_dirs(tmp_path, monkeypatch):
    meta = tmp_path / "meta"
    binc = tmp_path / "bin"
    meta.mkdir()
    binc.mkdir()
    (meta / "a.json").write_text("x" * 10)
    (meta / "b.json").write_text("y" * 20)
    (binc / "c.bin").write_text("z" * 40)
    monkeypatch.setattr(promstats, "META_DIR", str(meta))
    monkeypatch.setattr(promstats, "BIN_CACHE_DIR", str(binc))

    promstats.refresh_storage_gauges()

    text = promstats.render()
    assert "llama_kv_proxy_meta_files 2.0" in text
    assert "llama_kv_proxy_meta_bytes 30.0" in text
    assert "llama_kv_proxy_bin_bytes 40.0" in text


def test_refresh_storage_gauges_missing_dirs(tmp_path, monkeypatch):
    monkeypatch.setattr(promstats, "META_DIR", str(tmp_path / "nope"))
    monkeypatch.setattr(promstats, "BIN_CACHE_DIR", "")

    promstats.refresh_storage_gauges()

    text = promstats.render()
    assert "llama_kv_proxy_meta_files 0.0" in text
    assert "llama_kv_proxy_meta_bytes 0.0" in text
    assert "llama_kv_proxy_bin_bytes 0.0" in text


# --- /metrics endpoint ---------------------------------------------------------


async def test_metrics_endpoint_serves_proxy_and_backend_metrics():
    mock = MagicMock()
    mock.get_active_models = AsyncMock(return_value=["m1"])
    mock.get_metrics = AsyncMock(return_value="foo 1")
    app_module.app.state.clients = [mock]

    promstats.requests_total.labels(model="m1", stream="true", outcome="ok").inc()

    resp = await app_module.metrics_endpoint()

    body = resp.body.decode()
    assert (
        'llama_kv_proxy_requests_total{model="m1",outcome="ok",stream="true"} 1.0'
        in body
    )
    assert 'foo{backend="0",model="m1"} 1' in body


async def test_metrics_endpoint_model_filter_applies_to_proxy_metrics():
    mock = MagicMock()
    mock.get_active_models = AsyncMock(return_value=["m1", "m2"])
    mock.get_metrics = AsyncMock(side_effect=lambda m: f"foo {m}")
    app_module.app.state.clients = [mock]

    promstats.requests_total.labels(model="m1", stream="true", outcome="ok").inc()
    promstats.requests_total.labels(model="m2", stream="true", outcome="ok").inc()

    resp = await app_module.metrics_endpoint(model="m2")

    body = resp.body.decode()
    assert 'model="m2"' in body
    assert 'model="m1"' not in body


class _req:
    """Minimal stand-in for the Starlette request object chat() reads."""

    def __init__(self, data):
        self._data = data

    async def json(self):
        return self._data


def test_model_label_rejects_unusable_names():
    """model_label is the syntactic guard: a name with whitespace, control
    characters, or an absurd length is not a model id and must not become a
    label."""
    assert promstats.model_label("llama-3.1-70b") == "llama-3.1-70b"
    assert promstats.model_label("Qwen/Qwen2.5-7B-Instruct") == "Qwen/Qwen2.5-7B-Instruct"
    for junk in ("a" * 200, "weird name", "", "\x00evil", "new\nline", "tab\there"):
        assert promstats.model_label(junk) == promstats.UNRESOLVED_LABEL, junk


async def test_unresolved_alias_does_not_create_a_label_series(sm, monkeypatch):
    """End to end: an alias the proxy could not resolve is counted under the
    fixed bucket, whatever the client sent -- including a short, perfectly
    well-formed name, which the syntactic guard alone would let through."""
    client = sm.backends[0]["client"]
    client.get_slots = AsyncMock(return_value=[{"id": 0, "model": "m1"}])
    app_module.app.state.sm = sm
    app_module.app.state.clients = [client]
    sm.set_backend_slots(0, "m1", [{"id": 0}])
    sm.set_backend_slots(0, "m2", [{"id": 0}])

    monkeypatch.setattr(chat_flow, "BIG_THRESHOLD_WORDS", 1)
    monkeypatch.setattr(chat_flow, "_save_and_write_meta", AsyncMock(return_value=True))

    data = {
        "messages": [{"role": "user", "content": "hello world"}],
        "stream": False,
        "model": "default",
    }
    await app_module.chat(_req(data))

    body = promstats.render()
    assert f'model="{promstats.UNRESOLVED_LABEL}"' in body
    assert 'model="default"' not in body, (
        "an unresolved alias must not get its own metric series"
    )


async def test_error_responses_are_observed_in_the_histograms(sm, monkeypatch):
    """A failed request is a request: its latency must reach the histograms.

    They were only observed on the success path, so a rate() or
    histogram_quantile() over them silently under-reported every 5xx, and a
    proxy that failed fast looked faster than one that succeeded slowly."""
    client = sm.backends[0]["client"]
    client.chat_completions = AsyncMock(return_value={"object": "error", "message": "nope"})
    app_module.app.state.sm = sm
    app_module.app.state.clients = [client]
    sm.set_backend_slots(0, "m1", [{"id": 0}])

    monkeypatch.setattr(chat_flow, "BIG_THRESHOLD_WORDS", 1)
    resp = await app_module.chat(
        _req({"messages": [{"role": "user", "content": "hi"}], "stream": False})
    )
    assert resp.status_code >= 500

    body = promstats.render()
    # The error request is counted, i.e. the histogram has an observation.
    assert (
        'llama_kv_proxy_request_duration_seconds_count{model="m1",stream="false"} 1.0'
        in body
    )
