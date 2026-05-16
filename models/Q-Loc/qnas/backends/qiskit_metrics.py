from __future__ import annotations

from qnas.backends.qiskit_converter import circuit_spec_to_qiskit
from qnas.common.schema import CircuitSpec


def qiskit_basic_metrics(spec: CircuitSpec) -> dict[str, int | dict[str, int]]:
    circuit = circuit_spec_to_qiskit(spec)
    return {
        "qiskit_depth": int(circuit.depth()),
        "qiskit_size": int(circuit.size()),
        "qiskit_count_ops": {str(k): int(v) for k, v in circuit.count_ops().items()},
    }


def qiskit_transpile_metrics(spec: CircuitSpec, **transpile_kwargs) -> dict[str, int | dict[str, int]]:
    try:
        from qiskit import transpile
    except ImportError as exc:
        raise ImportError("qiskit is required for transpile metrics.") from exc

    circuit = transpile(circuit_spec_to_qiskit(spec), **transpile_kwargs)
    return {
        "transpiled_depth": int(circuit.depth()),
        "transpiled_size": int(circuit.size()),
        "transpiled_count_ops": {str(k): int(v) for k, v in circuit.count_ops().items()},
    }
