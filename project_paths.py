"""Portable default paths shared by SFRA command-line entry points."""

from __future__ import annotations

import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent


def _environment_path(name: str, fallback: Path) -> Path:
    """Resolve an optional environment override and expand ``~``."""
    return Path(os.environ.get(name, str(fallback))).expanduser().resolve()


DATA_DIR = _environment_path("SFRA_DATA_DIR", Path.home() / "data")
EXPS_DIR = _environment_path(
    "SFRA_EXPS_DIR", Path.home() / "classification" / "exps"
)
