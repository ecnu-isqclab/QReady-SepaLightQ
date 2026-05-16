from __future__ import annotations

from qnas.common.gate_set import gate_info
from qnas.common.schema import CircuitSpec


def structural_metrics(spec: CircuitSpec) -> dict[str, int]:
    n_1q = 0
    n_2q = 0
    for gate in spec.gates:
        info = gate_info(gate.op)
        if info.is_two_qubit:
            n_2q += 1
        else:
            n_1q += 1
    return {
        "n_qubits": spec.n_qubits,
        "n_gates": len(spec.gates),
        "n_1q_gates": n_1q,
        "n_2q_gates": n_2q,
        "depth_proxy": len(spec.gates),
        "n_weights": spec.n_weights,
    }

