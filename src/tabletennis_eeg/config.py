from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    return data or {}


def deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_configuration(root: Path) -> dict[str, Any]:
    config = {}
    for filename in ["dataset.yaml", "analysis.yaml", "channels.yaml", "sensor_regions.yaml"]:
        config = deep_merge(config, read_yaml(root / "config" / filename))
    return config


def load_model_configuration(root: Path, model_id: str) -> dict[str, Any]:
    path = root / "config" / "models" / f"{model_id}.yaml"
    if not path.exists():
        available = sorted(item.stem for item in path.parent.glob("*.yaml"))
        raise KeyError(f"Unknown model_id={model_id!r}. Available: {available}")
    return read_yaml(path)["model"]


def available_model_ids(root: Path) -> list[str]:
    return sorted(path.stem for path in (root / "config" / "models").glob("*.yaml"))
