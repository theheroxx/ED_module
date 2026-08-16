"""
common/io_utils.py
==================
Small, boring helpers so every step saves artifacts the same way and records
a machine-readable manifest that later steps (and the math model / GNN) can
discover without guessing filenames.
"""
from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd


def ensure_dir(path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p



def save_df(df: pd.DataFrame, path) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    if path.suffix == ".parquet":
        try:
            df.to_parquet(path, index=False)
            return path
        except Exception:
            path = path.with_suffix(".csv")
    df.to_csv(path, index=False)
    return path




def save_json(obj, path) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)
    return path


def write_manifest(out_dir, step_name: str, artifacts: dict, extra: dict | None = None):

    #Record what a step produced -> outputs/<step>/manifest.json.
    out_dir = Path(out_dir)
    manifest = {
        "step": step_name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "artifacts": {k: str(Path(v).resolve()) for k, v in artifacts.items()},
    }
    if extra:
        manifest.update(extra)
    save_json(manifest, out_dir / "manifest.json")
    return manifest
