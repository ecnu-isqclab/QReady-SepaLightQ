from __future__ import annotations

import json
from pathlib import Path

from qnas.common.schema import CircuitSpec


def load_circuit(path: str | Path) -> CircuitSpec:
    path = Path(path)
    data = json.loads(path.read_text())
    if "circuit" in data:
        data = data["circuit"]
    return CircuitSpec.from_dict(data)


def save_circuit(spec: CircuitSpec, path: str | Path, *, indent: int = 2) -> None:
    spec.validate()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(spec.to_dict(), indent=indent, sort_keys=True) + "\n")


def save_search_result(
    spec: CircuitSpec,
    path: str | Path,
    *,
    parameters: dict,
    training: dict,
    indent: int = 2,
) -> None:
    spec.validate()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "qnas_search_result_v1",
        "circuit": spec.to_dict(),
        "parameters": parameters,
        "training": training,
    }
    path.write_text(json.dumps(payload, indent=indent, sort_keys=True) + "\n")


def load_circuit_dir(path: str | Path) -> list[CircuitSpec]:
    path = Path(path)
    return [load_circuit(item) for item in sorted(path.glob("*.json"))]
