"""Tiny artifact store for the staged, cached pipeline (A6).

Each stage writes its outputs under ``<cache>/<stage>/`` and reads the prior
stage's outputs from disk, so every stage is independently rerunnable and
cached. We deliberately use plain formats (CSV / JSON / NPZ) so artifacts are
inspectable without the package installed.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_LOG_CONFIGURED = False


def get_logger(name: str = "m3") -> logging.Logger:
    global _LOG_CONFIGURED
    if not _LOG_CONFIGURED:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-5s  %(name)s  %(message)s",
                                                datefmt="%H:%M:%S"))
        root = logging.getLogger("m3")
        root.addHandler(handler)
        root.setLevel(logging.INFO)
        _LOG_CONFIGURED = True
    return logging.getLogger(name)


class Store:
    """Read/write artifacts for one stage under ``<cache>/<stage>/``."""

    def __init__(self, cache_dir: Path, stage: str):
        self.dir = Path(cache_dir) / stage
        self.dir.mkdir(parents=True, exist_ok=True)
        self.stage = stage

    # -- paths ------------------------------------------------------------
    def file(self, name: str) -> Path:
        return self.dir / name

    def exists(self, name: str) -> bool:
        return self.file(name).exists()

    # -- json -------------------------------------------------------------
    def write_json(self, name: str, obj: Any) -> Path:
        p = self.file(name)
        with p.open("w", encoding="utf-8") as fh:
            json.dump(obj, fh, indent=2, default=_json_default)
        return p

    def read_json(self, name: str) -> Any:
        with self.file(name).open("r", encoding="utf-8") as fh:
            return json.load(fh)

    # -- pandas -----------------------------------------------------------
    def write_csv(self, name: str, df: pd.DataFrame, index: bool = True) -> Path:
        p = self.file(name)
        df.to_csv(p, index=index)
        return p

    def read_csv(self, name: str, index_col: int | None = 0) -> pd.DataFrame:
        return pd.read_csv(self.file(name), index_col=index_col)

    # -- numpy ------------------------------------------------------------
    def write_npz(self, name: str, **arrays: np.ndarray) -> Path:
        p = self.file(name)
        np.savez_compressed(p, **arrays)
        return p

    def read_npz(self, name: str) -> Any:
        return np.load(self.file(name), allow_pickle=False)


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serializable: {type(o)}")
