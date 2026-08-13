"""Configuration loading helpers.

The bundled ``.yaml`` files are JSON-compatible YAML so the project has no hard
dependency on PyYAML. If PyYAML is installed, ordinary YAML is accepted too.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict


def load_config(path: str | Path | None, defaults: Dict[str, Any] | None = None) -> Dict[str, Any]:
    config: Dict[str, Any] = copy.deepcopy(defaults) if defaults else {}
    if path is None:
        return config
    text = Path(path).read_text(encoding="utf-8")
    loaded = _loads(text)
    if loaded is None:
        return config
    if not isinstance(loaded, dict):
        raise ValueError(f"Configuration root must be a mapping: {path}")
    return deep_merge(config, loaded)


def _loads(text: str) -> Any:
    stripped = text.strip()
    if not stripped:
        return {}
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise ValueError(
                "Config is not JSON-compatible and PyYAML is not installed. "
                "Use JSON syntax in .yaml files or install PyYAML."
            ) from exc
        return yaml.safe_load(stripped)


def deep_merge(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result
