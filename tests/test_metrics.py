# tests/test_metrics.py

"""metrics.py: relabel + merge backend /metrics into one Prometheus target.

- relabel adds model/backend labels to every metric line and collects
  # HELP/# TYPE per metric name (dropping # EOF and blank lines);
- merge dedupes # HELP/# TYPE by metric name and groups lines per metric;
- collect fetches /metrics for every active model across backends (optionally
  filtered by model), tolerating partial failures;
- the /metrics endpoint returns the merged body as text/plain.
"""

from unittest.mock import AsyncMock, MagicMock

import app as app_module
import metrics as pm
from llama_client import LlamaClient

# --- relabel ---------------------------------------------------------------


def test_relabel_adds_labels_to_unlabeled_line():
    lines, comments = pm.relabel("foo_total 123", {"model": "m1", "backend": "0"})
    assert lines == ['foo_total{backend="0",model="m1"} 123']
    assert comments == {}


def test_relabel_adds_labels_to_labeled_line():
    lines, _ = pm.relabel('foo_total{slot="0"} 456', {"model": "m1", "backend": "0"})
    assert lines == ['foo_total{backend="0",model="m1",slot="0"} 456']


def test_relabel_replaces_existing_model_label():
    lines, _ = pm.relabel('foo_total{model="old"} 1', {"model": "new", "backend": "0"})
    assert lines == ['foo_total{backend="0",model="new"} 1']


def test_relabel_escapes_label_values():
    lines, _ = pm.relabel("foo 1", {"model": 'a"b', "backend": "0"})
    assert lines == ['foo{backend="0",model="a\\"b"} 1']


def test_relabel_preserves_timestamp():
    lines, _ = pm.relabel("foo 1 1234567890", {"model": "m1", "backend": "0"})
    assert lines == ['foo{backend="0",model="m1"} 1 1234567890']


def test_relabel_collects_help_and_type_comments():
    raw = (
        "# HELP foo_total Help text\n"
        "# TYPE foo_total counter\n"
        "foo_total 1\n"
    )
    lines, comments = pm.relabel(raw, {"model": "m1", "backend": "0"})
    assert lines == ['foo_total{backend="0",model="m1"} 1']
    assert comments == {
        "foo_total": {"help": "foo_total Help text", "type": "foo_total counter"}
    }


def test_relabel_drops_eof_and_blanks():
    raw = "# HELP foo Help\nfoo 1\n# EOF\n\n"
    lines, comments = pm.relabel(raw, {"model": "m1", "backend": "0"})
    assert lines == ['foo{backend="0",model="m1"} 1']
    assert comments["foo"]["help"] == "foo Help"


# --- merge -----------------------------------------------------------------


def test_merge_dedupes_comments_and_keeps_all_lines():
    raw = "# HELP foo Help\n# TYPE foo counter\nfoo 1"
    l1, c1 = pm.relabel(raw, {"model": "m1", "backend": "0"})
    l2, c2 = pm.relabel(raw.replace("foo 1", "foo 2"), {"model": "m2", "backend": "0"})
    out = pm.merge([(l1, c1), (l2, c2)])
    lines = out.splitlines()
    assert lines.count("# HELP foo Help") == 1
    assert lines.count("# TYPE foo counter") == 1
    assert 'foo{backend="0",model="m1"} 1' in lines
    assert 'foo{backend="0",model="m2"} 2' in lines


def test_merge_empty_is_empty():
    assert pm.merge([]) == ""


def test_merge_groups_comments_before_their_lines():
    raw = "# HELP foo Help\n# TYPE foo counter\nfoo 1"
    l1, c1 = pm.relabel(raw, {"model": "m1", "backend": "0"})
    out = pm.merge([(l1, c1)])
    lines = out.splitlines()
    assert lines.index("# HELP foo Help") < lines.index('foo{backend="0",model="m1"} 1')
    assert lines.index("# TYPE foo counter") < lines.index('foo{backend="0",model="m1"} 1')


# --- collect ---------------------------------------------------------------


def _mock_client(active_models, metrics_by_model):
    c = MagicMock(spec=LlamaClient)
    c.get_active_models = AsyncMock(return_value=active_models)
    c.get_metrics = AsyncMock(side_effect=lambda m: metrics_by_model.get(m))
    return c


async def test_collect_merges_across_models():
    c0 = _mock_client(["m1", "m2"], {"m1": "foo 1", "m2": "foo 2"})
    out = await pm.collect([c0])
    lines = out.splitlines()
    assert 'foo{backend="0",model="m1"} 1' in lines
    assert 'foo{backend="0",model="m2"} 2' in lines


async def test_collect_labels_with_backend_index():
    c0 = _mock_client(["m1"], {"m1": "foo 1"})
    c1 = _mock_client(["m1"], {"m1": "foo 2"})
    out = await pm.collect([c0, c1])
    lines = out.splitlines()
    assert 'foo{backend="0",model="m1"} 1' in lines
    assert 'foo{backend="1",model="m1"} 2' in lines


async def test_collect_filters_by_model():
    c0 = _mock_client(["m1", "m2"], {"m1": "foo 1", "m2": "foo 2"})
    out = await pm.collect([c0], model="m2")
    lines = out.splitlines()
    assert 'foo{backend="0",model="m2"} 2' in lines
    assert not any('model="m1"' in l for l in lines)


async def test_collect_tolerates_partial_failure():
    def _side_effect(m):
        if m == "m1":
            return "foo 1"
        raise RuntimeError("down")

    c0 = MagicMock(spec=LlamaClient)
    c0.get_active_models = AsyncMock(return_value=["m1", "m2"])
    c0.get_metrics = AsyncMock(side_effect=_side_effect)
    out = await pm.collect([c0])
    lines = out.splitlines()
    assert 'foo{backend="0",model="m1"} 1' in lines
    assert not any('model="m2"' in l for l in lines)


async def test_collect_tolerates_client_failure():
    c0 = MagicMock(spec=LlamaClient)
    c0.get_active_models = AsyncMock(side_effect=Exception("backend down"))
    assert await pm.collect([c0]) == ""


# --- /metrics endpoint -----------------------------------------------------


def test_metrics_route_registered():
    paths = {r.path for r in app_module.app.routes}
    assert "/metrics" in paths


async def test_metrics_endpoint_returns_merged_text():
    mock = MagicMock()
    mock.get_active_models = AsyncMock(return_value=["m1"])
    mock.get_metrics = AsyncMock(return_value="foo 1")
    app_module.app.state.clients = [mock]

    resp = await app_module.metrics_endpoint()

    assert resp.media_type.startswith("text/plain")
    body = resp.body.decode()
    assert 'foo{backend="0",model="m1"} 1' in body


async def test_metrics_endpoint_filters_by_model():
    mock = MagicMock()
    mock.get_active_models = AsyncMock(return_value=["m1", "m2"])
    mock.get_metrics = AsyncMock(side_effect=lambda m: f"foo {m}")
    app_module.app.state.clients = [mock]

    resp = await app_module.metrics_endpoint(model="m2")

    body = resp.body.decode()
    assert 'model="m2"' in body
    assert 'model="m1"' not in body
