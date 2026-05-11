from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

@dataclass(frozen=True)
class GateCheckResult:
    metric: str
    scope: str                 # e.g. "micro", "macro", "class:building"
    comparator: str            # ">=", "<=", etc.
    threshold: float
    actual: float
    passed: bool

@dataclass(frozen=True)
class GateContract:
    gate_dir_path: Path

    gate_name: str             # "gate_a" or "gate_b"
    task: str

    decision: str              # "pass" | "fail"
    passed: bool

    checks: List[Dict[str, Any]]
    summary: Dict[str, Any]    # counts, key metrics, etc.

    threshold_spec: Dict[str, Any]