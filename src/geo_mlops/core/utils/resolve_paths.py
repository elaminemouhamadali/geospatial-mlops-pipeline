from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Dict


def _resolve_dataset_roots(
    dataset_root_path: Path,
) -> List[Path]:
    root = Path(dataset_root_path)

    if not root.exists():
        raise FileNotFoundError(f"dataset_root does not exist: {root}")

    if not root.is_dir():
        raise NotADirectoryError(f"dataset_root is not a directory: {root}")

    roots = [p for p in sorted(root.iterdir()) if p.is_dir()]

    return roots


def _relaxed_lookup(stem: str, mapping: Dict[str, Path]) -> Optional[Path]:
    p = mapping.get(stem)
    if p is not None:
        return p
    for k, v in mapping.items():
        if stem.startswith(k) or k.startswith(stem):
            return v
    return None

def _tif_map(root: Path) -> dict[str, Path]:
    if not root.is_dir():
        return {}
    paths = list(root.glob("*.tif")) + list(root.glob("*.tiff"))
    return {p.stem: p for p in sorted(paths)}