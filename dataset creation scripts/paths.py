"""Project paths: CSV data lives in ``../data`` relative to this package directory."""

from __future__ import annotations

from pathlib import Path

# This file is in ``dataset creation scripts/``; repo root is one level up.
_PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _PACKAGE_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = DATA_DIR / "results"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"


def data_path(*parts: str) -> Path:
    """``data_path('foo.csv')`` → ``<repo>/data/foo.csv``."""
    return DATA_DIR.joinpath(*parts)


def checkpoint_path_for(output_file: str | Path) -> Path:
    """``checkpoint_<basename>`` next to ``output_file`` (same directory)."""
    p = Path(output_file)
    return p.parent / f"checkpoint_{p.name}"
