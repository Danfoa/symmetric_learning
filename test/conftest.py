from __future__ import annotations

import importlib
import os
from pathlib import Path

import escnn.group as escnn_group


def pytest_sessionstart(session) -> None:  # noqa: D103
    cache_dir = Path(os.environ.get("ESCNN_CACHE_DIR", "/tmp/escnn_cache"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    escnn_group.__cache_path__ = str(cache_dir)

    import escnn.group._clebsh_gordan as cg

    importlib.reload(cg)
