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
