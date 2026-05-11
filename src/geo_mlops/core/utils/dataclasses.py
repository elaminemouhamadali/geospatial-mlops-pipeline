from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Mapping
from pathlib import Path
import json

def _as_plain_dict(obj: Any) -> Dict[str, Any]:
    if obj is None:
        return {}

    if is_dataclass(obj):
        return asdict(obj)

    if isinstance(obj, dict):
        return _to_jsonable(obj)

    return {"value": str(obj)}


def _load_json(path: str | Path) -> Dict[str, Any]:
    p = Path(path)

    if not p.exists():
        raise FileNotFoundError(f"JSON file not found: {p}")

    obj = json.loads(p.read_text())

    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object at root of {p}")

    return obj

def _to_jsonable(obj: Any) -> Any:
    if is_dataclass(obj):
        return _to_jsonable(asdict(obj))

    if isinstance(obj, Path):
        return str(obj)

    if isinstance(obj, Mapping):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple, set)):
        return [_to_jsonable(v) for v in obj]

    return obj

def _ensure_path(value: str | Path) -> Path:
    return value if isinstance(value, Path) else Path(value)

def _unique_group_values(df, group_col: str) -> list[str]:
    if group_col not in df.columns:
        raise ValueError(f"Expected group column {group_col!r} in split dataframe.")
    return sorted(map(str, df[group_col].dropna().unique().tolist()))

