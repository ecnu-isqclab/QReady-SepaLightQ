from __future__ import annotations

from qnas.common.schema import CircuitSpec, GateSpec


def load_qiskit():
    try:
        from qiskit import QuantumCircuit
        from qiskit.circuit import ParameterVector
    except ImportError as exc:
        raise ImportError("qiskit is required for Qiskit conversion.") from exc
    return QuantumCircuit, ParameterVector


def _resolve_parameter(gate: GateSpec, x_params, theta_params):
    if gate.source == "input":
        return x_params[gate.index]
    if gate.source == "weight":
        return theta_params[gate.index]
    return None


def circuit_spec_to_qiskit(spec: CircuitSpec):
    QuantumCircuit, ParameterVector = load_qiskit()
    circuit = QuantumCircuit(spec.n_qubits)
    x_params = ParameterVector("x", spec.n_inputs)
    theta_params = ParameterVector("theta", spec.n_weights)

    for gate in spec.gates:
        op = gate.op.lower()
        wires = gate.wires
        param = _resolve_parameter(gate, x_params, theta_params)
        if op in {"skip", "identity"}:
            continue
        if op in {"rx", "ry", "rz"}:
            getattr(circuit, op)(param, wires[0])
        elif op in {"h", "x", "sx"}:
            getattr(circuit, op)(wires[0])
        elif op in {"cx", "cnot"}:
            circuit.cx(wires[0], wires[1])
        elif op == "cz":
            circuit.cz(wires[0], wires[1])
        elif op in {"crx", "cry", "crz"}:
            getattr(circuit, op)(param, wires[0], wires[1])
        elif op in {"rxx", "ryy", "rzz"}:
            getattr(circuit, op)(param, wires[0], wires[1])
        else:
            raise NotImplementedError(f"Qiskit converter does not support gate {op!r}.")
    return circuit
