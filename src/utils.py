from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def project_root() -> Path:
	return PROJECT_ROOT


def data_path(*parts: str | Path) -> Path:
	return PROJECT_ROOT.joinpath("data", *map(str, parts))

