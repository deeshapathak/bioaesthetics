"""Load and validate the single pinned config file (A6: config-driven, pinned)."""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG = Path("config/m3.yaml")


class ConfigError(ValueError):
    """Raised when the config is missing required keys or has bad values."""


@dataclasses.dataclass
class Config:
    """Thin typed wrapper around the parsed YAML.

    Access nested values with :meth:`get` (dotted path) or the raw ``data``
    dict. Paths are resolved relative to the config file's parent so the
    pipeline is runnable from anywhere.
    """

    data: dict[str, Any]
    root: Path

    # -- scalars commonly reached for -------------------------------------
    @property
    def seed(self) -> int:
        return int(self.data["seed"])

    @property
    def mode(self) -> str:
        mode = self.data.get("mode", "synthetic")
        if mode not in ("synthetic", "real"):
            raise ConfigError(f"mode must be 'synthetic' or 'real', got {mode!r}")
        return mode

    # -- paths (resolved relative to repo root) ---------------------------
    def path(self, key: str) -> Path:
        p = Path(self.data["paths"][key])
        return p if p.is_absolute() else (self.root / p)

    @property
    def cache_dir(self) -> Path:
        return self.path("cache")

    @property
    def outputs_dir(self) -> Path:
        return self.path("outputs")

    @property
    def data_dir(self) -> Path:
        return self.path("data")

    # -- generic dotted access -------------------------------------------
    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self.data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def require(self, dotted: str) -> Any:
        sentinel = object()
        val = self.get(dotted, sentinel)
        if val is sentinel:
            raise ConfigError(f"missing required config key: {dotted}")
        return val


_REQUIRED_KEYS = (
    "seed",
    "paths.cache",
    "paths.outputs",
    "signatures.padj",
    "signatures.top_n",
    "connectivity.method",
    "validation.go_no_go.recovery_auc_min",
)


def load_config(path: str | Path | None = None) -> Config:
    """Parse and validate ``config/m3.yaml`` (or a given path)."""
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    if not cfg_path.exists():
        raise ConfigError(f"config file not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    # the config lives in <root>/config/m3.yaml, so root is two parents up
    cfg = Config(data=data, root=cfg_path.resolve().parent.parent)
    for key in _REQUIRED_KEYS:
        cfg.require(key)
    # ensure work dirs exist
    cfg.cache_dir.mkdir(parents=True, exist_ok=True)
    cfg.outputs_dir.mkdir(parents=True, exist_ok=True)
    return cfg
