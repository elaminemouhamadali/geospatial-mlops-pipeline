from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from geo_mlops.core.utils.dataclasses import _load_json
from pathlib import Path
from typing import Any, Dict, Optional

import mlflow
from mlflow.tracking import MlflowClient

REGISTRY_MANIFEST_NAME = "registry_result.json"


@dataclass(frozen=True)
class RegistryResult:
    action: str
    model_name: str
    model_version: str
    model_uri: Optional[str]
    gate_name: str
    gate_passed: bool
    mlflow_run_id: Optional[str]
    registry_uri: Optional[str]
    tracking_uri: Optional[str]
    details: Dict[str, Any]


def _require_gate_passed(gate_contract_path: str | Path) -> Dict[str, Any]:
    gate = _load_json(gate_contract_path)

    if not bool(gate.get("passed", False)):
        gate_name = gate.get("gate_name", "<unknown>")
        decision = gate.get("decision", "<unknown>")
        raise RuntimeError(
            f"Gate did not pass; refusing registry transition. "
            f"gate={gate_name!r}, decision={decision!r}"
        )

    return gate


def _write_registry_result(
    result: RegistryResult,
    out_dir: Optional[str | Path],
) -> Optional[Path]:
    if out_dir is None:
        return None

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    path = out / REGISTRY_MANIFEST_NAME
    path.write_text(json.dumps(asdict(result), indent=2))
    return path


def _set_model_version_alias_safe(
    client: MlflowClient,
    *,
    model_name: str,
    alias: str,
    version: str,
) -> None:
    """
    MLflow aliases are available in newer MLflow versions.
    If unavailable, we skip aliases and rely on tags.
    """
    if not alias:
        return

    if hasattr(client, "set_registered_model_alias"):
        client.set_registered_model_alias(
            name=model_name,
            alias=alias,
            version=version,
        )


def register_candidate_model(
    *,
    model_name: str,
    mlflow_run_id: str,
    gate_contract_path: str | Path,
    model_artifact_path: str = "model",
    tracking_uri: Optional[str] = None,
    registry_uri: Optional[str] = None,
    candidate_alias: str = "candidate",
    out_dir: Optional[str | Path] = None,
    extra_tags: Optional[Dict[str, str]] = None,
) -> RegistryResult:
    """
    Register a model version after Gate A passes.

    Assumes the training run logged an MLflow model at:
        runs:/<mlflow_run_id>/<model_artifact_path>

    Example:
        runs:/abc123/model
    """
    gate = _require_gate_passed(gate_contract_path)

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    if registry_uri:
        mlflow.set_registry_uri(registry_uri)

    client = MlflowClient()

    model_uri = f"runs:/{mlflow_run_id}/{model_artifact_path}"

    registered = mlflow.register_model(
        model_uri=model_uri,
        name=model_name,
    )

    version = str(registered.version)

    tags = {
        "stage": "candidate",
        "registry_action": "register_candidate",
        "gate_name": str(gate.get("gate_name", "")),
        "gate_decision": str(gate.get("decision", "")),
        "gate_contract_path": str(gate_contract_path),
        "source_mlflow_run_id": mlflow_run_id,
    }
    tags.update(extra_tags or {})

    for key, value in tags.items():
        client.set_model_version_tag(
            name=model_name,
            version=version,
            key=str(key),
            value=str(value),
        )

    _set_model_version_alias_safe(
        client,
        model_name=model_name,
        alias=candidate_alias,
        version=version,
    )

    result = RegistryResult(
        action="register_candidate",
        model_name=model_name,
        model_version=version,
        model_uri=model_uri,
        gate_name=str(gate.get("gate_name", "")),
        gate_passed=True,
        mlflow_run_id=mlflow_run_id,
        registry_uri=registry_uri,
        tracking_uri=tracking_uri,
        details={
            "candidate_alias": candidate_alias,
            "gate_contract_path": str(gate_contract_path),
        },
    )

    _write_registry_result(result, out_dir)
    return result


def promote_model_to_production(
    *,
    model_name: str,
    model_version: str,
    gate_contract_path: str | Path,
    tracking_uri: Optional[str] = None,
    registry_uri: Optional[str] = None,
    production_alias: str = "production",
    archive_candidate_alias: bool = False,
    out_dir: Optional[str | Path] = None,
    extra_tags: Optional[Dict[str, str]] = None,
) -> RegistryResult:
    """
    Promote an existing registered model version after Gate B passes.
    """
    gate = _require_gate_passed(gate_contract_path)

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    if registry_uri:
        mlflow.set_registry_uri(registry_uri)

    client = MlflowClient()
    version = str(model_version)

    tags = {
        "stage": "production",
        "registry_action": "promote_production",
        "gate_name": str(gate.get("gate_name", "")),
        "gate_decision": str(gate.get("decision", "")),
        "gate_contract_path": str(gate_contract_path),
    }
    tags.update(extra_tags or {})

    for key, value in tags.items():
        client.set_model_version_tag(
            name=model_name,
            version=version,
            key=str(key),
            value=str(value),
        )

    _set_model_version_alias_safe(
        client,
        model_name=model_name,
        alias=production_alias,
        version=version,
    )

    if archive_candidate_alias and hasattr(client, "delete_registered_model_alias"):
        try:
            client.delete_registered_model_alias(
                name=model_name,
                alias="candidate",
            )
        except Exception:
            pass

    result = RegistryResult(
        action="promote_production",
        model_name=model_name,
        model_version=version,
        model_uri=None,
        gate_name=str(gate.get("gate_name", "")),
        gate_passed=True,
        mlflow_run_id=None,
        registry_uri=registry_uri,
        tracking_uri=tracking_uri,
        details={
            "production_alias": production_alias,
            "gate_contract_path": str(gate_contract_path),
        },
    )

    _write_registry_result(result, out_dir)
    return result