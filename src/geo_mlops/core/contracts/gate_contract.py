from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class GateCheckResult:
    metric: str
    scope: str  # e.g. "micro", "macro", "class:building"
    comparator: str  # ">=", "<=", etc.
    threshold: float
    actual: float
    passed: bool


@dataclass(frozen=True)
class GateContract:
    gate_dir_path: Path

    gate_name: str  # "gate_a" or "gate_b"
    task: str

    decision: str  # "pass" | "fail"
    passed: bool

    checks: list[dict[str, Any]]
    summary: dict[str, Any]  # counts, key metrics, etc.

    threshold_spec: dict[str, Any]
