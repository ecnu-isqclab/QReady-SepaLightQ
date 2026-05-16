from __future__ import annotations

import random
from collections.abc import Sequence

from qnas.common.gate_set import gate_info
from qnas.common.schema import CircuitSpec, GateSpec


def linear_edges(n_qubits: int) -> list[tuple[int, int]]:
    return [(idx, idx + 1) for idx in range(n_qubits - 1)]


def ring_edges(n_qubits: int) -> list[tuple[int, int]]:
    edges = linear_edges(n_qubits)
    if n_qubits > 2:
        edges.append((n_qubits - 1, 0))
    return edges


def _edges(n_qubits: int, pattern: str) -> list[tuple[int, int]]:
    if pattern == "linear":
        return linear_edges(n_qubits)
    if pattern == "ring":
        return ring_edges(n_qubits)
    raise ValueError(f"Unknown entangle_pattern {pattern!r}.")


def sample_layered_circuit(
    *,
    name: str,
    n_qubits: int = 4,
    n_layers: int = 3,
    oneq_ops: Sequence[str] = ("rx", "ry", "rz"),
    twoq_ops: Sequence[str] = ("cx", "cz", "crx", "cry", "crz"),
    entangle_pattern: str = "linear",
    input_encoding: str = "ry",
    rng: random.Random | None = None,
) -> CircuitSpec:
    rng = rng or random.Random()
    gates: list[GateSpec] = []

    for qubit in range(n_qubits):
        gates.append(GateSpec(input_encoding, [qubit], "input", qubit))

    weight_index = 0
    edges = _edges(n_qubits, entangle_pattern)
    for layer_idx in range(n_layers):
        for qubit in range(n_qubits):
            op = rng.choice(list(oneq_ops)).lower()
            info = gate_info(op)
            source = "weight" if info.num_params else "none"
            index = weight_index if info.num_params else None
            gates.append(GateSpec(op, [qubit], source, index))
            weight_index += info.num_params

        for edge in edges:
            op = rng.choice(list(twoq_ops)).lower()
            info = gate_info(op)
            source = "weight" if info.num_params else "none"
            index = weight_index if info.num_params else None
            gates.append(GateSpec(op, list(edge), source, index))
            weight_index += info.num_params

    spec = CircuitSpec(
        name=name,
        n_qubits=n_qubits,
        n_inputs=n_qubits,
        n_weights=weight_index,
        measured_qubits=list(range(n_qubits)),
        gates=gates,
        meta={
            "sampler": "layered_random",
            "n_layers": n_layers,
            "entangle_pattern": entangle_pattern,
        },
    )
    spec.validate()
    return spec
