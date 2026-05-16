from __future__ import annotations

from qnas.common.schema import CircuitSpec, GateSpec


def load_tensorcircuit():
    try:
        import numpy as np

        if not hasattr(np, "ComplexWarning"):
            np.ComplexWarning = RuntimeWarning
        import tensorcircuit as tc
        import tensorflow as tf  # noqa: F401
    except ImportError as exc:
        raise ImportError("tensorcircuit and tensorflow are required for TensorCircuit backend.") from exc
    backend = tc.set_backend("tensorflow")
    return tc, backend


def resolve_gate_value(gate: GateSpec, inputs, weights):
    if gate.source == "input":
        return inputs[gate.index]
    if gate.source == "weight":
        return weights[gate.index]
    return None


def apply_gate(circuit, gate: GateSpec, value) -> None:
    op = gate.op.lower()
    wires = gate.wires

    if op in {"skip", "identity"}:
        return
    if op in {"rx", "ry", "rz"}:
        getattr(circuit, op)(wires[0], theta=value)
    elif op in {"h", "x", "sx"}:
        name = "H" if op == "h" and hasattr(circuit, "H") else op
        getattr(circuit, name)(wires[0])
    elif op in {"cx", "cnot"}:
        circuit.cnot(wires[0], wires[1])
    elif op == "cz":
        circuit.cz(wires[0], wires[1])
    elif op in {"crx", "cry", "crz", "rxx", "ryy", "rzz"}:
        getattr(circuit, op)(wires[0], wires[1], theta=value)
    else:
        raise NotImplementedError(f"TensorCircuit backend does not support gate {op!r}.")


def build_circuit(spec: CircuitSpec, inputs, weights):
    tc, _ = load_tensorcircuit()
    circuit = tc.Circuit(spec.n_qubits)
    for gate in spec.gates:
        apply_gate(circuit, gate, resolve_gate_value(gate, inputs, weights))
    return tc, circuit


def expectation_fn(spec: CircuitSpec):
    tc, backend = load_tensorcircuit()

    def run(inputs, weights):
        circuit = tc.Circuit(spec.n_qubits)
        for gate in spec.gates:
            apply_gate(circuit, gate, resolve_gate_value(gate, inputs, weights))
        return backend.stack(
            [
                backend.real(circuit.expectation([tc.gates.z(), [qubit]]))
                for qubit in spec.measured_qubits
            ]
        )

    return run


def state_fn(spec: CircuitSpec):
    tc, _ = load_tensorcircuit()

    def run(inputs, weights):
        circuit = tc.Circuit(spec.n_qubits)
        for gate in spec.gates:
            apply_gate(circuit, gate, resolve_gate_value(gate, inputs, weights))
        return circuit.state()

    return run
