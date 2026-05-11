from __future__ import annotations

from pathlib import Path
from typing import Any

from geo_mlops.core.config.loader import load_cfg, require_section


def build_evaluation_cfg(task_cfg_path: str | Path) -> dict[str, Any]:
    cfg = load_cfg(task_cfg_path)
    return require_section(cfg, "evaluation")
