# tests/test_version.py

"""The /version endpoint reports the proxy name and version so operators can
identify the running build."""

import app as app_module
from version import __version__


async def test_version_endpoint():
    """GET /version returns the fixed name and the current version."""
    resp = await app_module.version()
    assert resp["name"] == "llama-kv-proxy"
    assert resp["version"] == __version__


def test_version_is_semver():
    """The version string is a three-part numeric semver."""
    parts = __version__.split(".")
    assert len(parts) == 3, f"expected x.y.z, got {__version__!r}"
    assert all(p.isdigit() for p in parts), f"non-numeric part in {__version__!r}"
