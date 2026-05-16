from __future__ import annotations

import json

from qnas.common.schema import CircuitSpec


def circuit_fingerprint(spec: CircuitSpec) -> str:
    payload = [
        (gate.op, tuple(gate.wires), gate.source, gate.index)
        for gate in spec.gates
    ]
    return json.dumps(payload, sort_keys=True)


def deduplicate_circuits(specs: list[CircuitSpec]) -> list[CircuitSpec]:
    seen: set[str] = set()
    unique: list[CircuitSpec] = []
    for spec in specs:
        key = circuit_fingerprint(spec)
        if key not in seen:
            seen.add(key)
            unique.append(spec)
    return unique

