from __future__ import annotations

import hashlib
import json
import os
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd

from src.project_paths import processed_dir


_CACHE_FORCE: ContextVar[bool] = ContextVar("stage_cache_force", default=False)
_CACHE_ENABLED: ContextVar[bool] = ContextVar("stage_cache_enabled", default=True)


@contextmanager
def stage_cache_context(*, force: bool = False, enabled: bool = True):
    force_token = _CACHE_FORCE.set(bool(force))
    enabled_token = _CACHE_ENABLED.set(bool(enabled))
    try:
        yield
    finally:
        _CACHE_FORCE.reset(force_token)
        _CACHE_ENABLED.reset(enabled_token)


def dataframe_fingerprint(df: pd.DataFrame | None) -> str:
    """Stable-ish content fingerprint for cache invalidation."""
    if df is None:
        return "none"

    h = hashlib.sha256()
    h.update(str(df.shape).encode("utf-8"))
    h.update("\0".join(map(str, df.columns)).encode("utf-8"))
    h.update("\0".join(map(str, df.dtypes)).encode("utf-8"))

    if not df.empty:
        hashed = pd.util.hash_pandas_object(df, index=True).to_numpy()
        h.update(hashed.tobytes())
    return h.hexdigest()


def file_fingerprint(path: str | Path) -> str:
    p = Path(path)
    h = hashlib.sha256()
    h.update(str(p).encode("utf-8"))
    if p.is_file():
        h.update(p.read_bytes())
    else:
        h.update(b"<missing>")
    return h.hexdigest()


def combined_fingerprint(parts: Iterable[object]) -> str:
    h = hashlib.sha256()
    for part in parts:
        h.update(str(part).encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


def stage_cache_dir(strategy_id: str) -> Path:
    return processed_dir(strategy_id) / "stages" / strategy_id


def load_or_build_stage(
    *,
    strategy_id: str,
    stage_name: str,
    fingerprint: str,
    builder: Callable[[], pd.DataFrame],
    force: bool | None = None,
) -> pd.DataFrame:
    """Load a cached stage CSV when its manifest fingerprint matches."""
    enabled = _CACHE_ENABLED.get()
    force_rebuild = _CACHE_FORCE.get() if force is None else bool(force)
    cache_dir = stage_cache_dir(strategy_id)
    csv_path = cache_dir / f"{stage_name}.csv"
    manifest_path = cache_dir / f"{stage_name}.manifest.json"

    if enabled and not force_rebuild and csv_path.exists() and manifest_path.exists():
        try:
            with open(manifest_path, encoding="utf-8") as f:
                manifest = json.load(f)
            if manifest.get("fingerprint") == fingerprint:
                columns = manifest.get("columns", [])
                if not columns:
                    print(f"[CACHE] {strategy_id}:{stage_name} -> empty DataFrame")
                    return pd.DataFrame()
                print(f"[CACHE] {strategy_id}:{stage_name} -> {csv_path}")
                return pd.read_csv(csv_path)
        except Exception as exc:
            print(f"[CACHE MISS] {strategy_id}:{stage_name} manifest read failed: {exc}")

    df = builder()
    if df is None:
        df = pd.DataFrame()
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Stage builder must return a pandas DataFrame: {strategy_id}:{stage_name}")

    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp_csv = csv_path.with_suffix(".csv.tmp")
    tmp_manifest = manifest_path.with_suffix(".manifest.json.tmp")
    df.to_csv(tmp_csv, index=False)
    manifest = {
        "schema_version": 1,
        "strategy_id": strategy_id,
        "stage_name": stage_name,
        "fingerprint": fingerprint,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "rows": int(len(df)),
        "columns": list(map(str, df.columns)),
    }
    with open(tmp_manifest, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    os.replace(tmp_csv, csv_path)
    os.replace(tmp_manifest, manifest_path)
    print(f"[BUILD] {strategy_id}:{stage_name} -> {csv_path}")
    return df
