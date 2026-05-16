from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from qnas.common.io import load_circuit
from qnas.backends.qiskit_converter import circuit_spec_to_qiskit
from qnas.metrics.hardware import hardware_cost
from qnas.metrics.structural import structural_metrics


def load_payload(path: str | Path) -> dict:
    data = json.loads(Path(path).read_text())
    if "circuit" in data:
        return data
    return {"format": "circuit_spec_v1", "circuit": data}


def parse_args():
    parser = argparse.ArgumentParser(description="Inspect a CircuitSpec json file.")
    parser.add_argument("path")
    parser.add_argument("--qiskit-draw", action="store_true", help="Print the Qiskit text circuit.")
    parser.add_argument("--bind-params", action="store_true", help="Bind saved theta values before drawing/QASM.")
    parser.add_argument("--show-params", action="store_true", help="Print exported theta[index] values.")
    parser.add_argument("--qasm", action="store_true", help="Print OpenQASM if supported by installed Qiskit.")
    return parser.parse_args()


def exported_theta_values(spec, payload: dict) -> list[float] | None:
    params = payload.get("parameters")
    if not params:
        return None
    theta_1q = params.get("theta_1q")
    theta_2q = params.get("theta_2q")
    if theta_1q is None or theta_2q is None:
        return None

    values: list[float] = []
    oneq_cursor = 0
    twoq_cursor = 0
    for gate in spec.gates:
        if gate.source != "weight":
            continue
        if len(gate.wires) == 1:
            layer = oneq_cursor // spec.n_qubits
            qubit = oneq_cursor % spec.n_qubits
            values.append(float(theta_1q[layer][qubit]))
            oneq_cursor += 1
        elif len(gate.wires) == 2:
            edges_per_layer = max(1, spec.n_qubits - 1)
            layer = twoq_cursor // edges_per_layer
            edge = twoq_cursor % edges_per_layer
            values.append(float(theta_2q[layer][edge]))
            twoq_cursor += 1
    return values


def bind_theta_parameters(circuit, theta_values: list[float]):
    mapping = {}
    for parameter in circuit.parameters:
        name = getattr(parameter, "name", "")
        if name.startswith("theta[") and name.endswith("]"):
            index = int(name[len("theta[") : -1])
            if index < len(theta_values):
                mapping[parameter] = theta_values[index]
    if hasattr(circuit, "assign_parameters"):
        return circuit.assign_parameters(mapping, inplace=False)
    return circuit.bind_parameters(mapping)


def main():
    args = parse_args()
    payload = load_payload(args.path)
    spec = load_circuit(args.path)
    metrics = structural_metrics(spec)
    metrics["hardware_cost"] = hardware_cost(spec)
    theta_values = exported_theta_values(spec, payload)

    if args.qiskit_draw or args.qasm:
        try:
            circuit = circuit_spec_to_qiskit(spec)
        except ImportError as exc:
            raise SystemExit(str(exc)) from exc
        if args.bind_params:
            if theta_values is None:
                raise SystemExit("No saved theta values found in this json. Re-run toy_search_fit.py first.")
            circuit = bind_theta_parameters(circuit, theta_values)
    else:
        output = {"circuit": spec.to_dict(), "metrics": metrics}
        if "training" in payload:
            output["training"] = payload["training"]
        if "parameters" in payload:
            output["parameter_keys"] = sorted(payload["parameters"].keys())
        print(json.dumps(output, indent=2, sort_keys=True))

    if args.show_params:
        if theta_values is None:
            raise SystemExit("No saved theta values found in this json. Re-run toy_search_fit.py first.")
        print("\n# Exported theta values")
        for index, value in enumerate(theta_values):
            print(f"theta[{index}] = {value:.10f}")

    if args.qiskit_draw:
        print("\n# Qiskit circuit")
        print(circuit.draw(output="text"))

    if args.qasm:
        print("\n# OpenQASM")
        if hasattr(circuit, "qasm"):
            print(circuit.qasm())
        else:
            try:
                from qiskit import qasm3
            except ImportError as exc:
                raise ImportError("Installed Qiskit does not expose circuit.qasm() or qiskit.qasm3.") from exc
            print(qasm3.dumps(circuit))


if __name__ == "__main__":
    main()
