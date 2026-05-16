from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


VALID_SOURCES = {"input", "weight", "none"}


@dataclass(frozen=True)
class GateSpec:
    """One operation in a quantum circuit IR."""

    op: str
    wires: list[int]
    source: str = "none"
    index: int | None = None

    def validate(self, n_qubits: int) -> None:
        if self.source not in VALID_SOURCES:
            raise ValueError(f"Invalid gate source {self.source!r}; expected one of {sorted(VALID_SOURCES)}.")
        if self.source == "none" and self.index is not None:
            raise ValueError("source='none' gates must not carry an index.")
        if self.source != "none" and self.index is None:
            raise ValueError(f"Parameterized gate {self.op!r} with source={self.source!r} needs index.")
        if not self.wires:
            raise ValueError(f"Gate {self.op!r} must have at least one wire.")
        for wire in self.wires:
            if wire < 0 or wire >= n_qubits:
                raise ValueError(f"Wire {wire} out of range for n_qubits={n_qubits}.")

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"op": self.op, "wires": list(self.wires), "source": self.source}
        if self.index is not None:
            data["index"] = self.index
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GateSpec":
        return cls(
            op=str(data["op"]).lower(),
            wires=[int(wire) for wire in data["wires"]],
            source=str(data.get("source", "none")),
            index=None if data.get("index") is None else int(data["index"]),
        )


@dataclass
class CircuitSpec:
    """Serializable circuit IR shared by search NAS and sample NAS."""

    name: str
    n_qubits: int
    n_inputs: int
    n_weights: int
    measured_qubits: list[int]
    gates: list[GateSpec]
    meta: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.n_qubits <= 0:
            raise ValueError("n_qubits must be positive.")
        if self.n_inputs < 0 or self.n_weights < 0:
            raise ValueError("n_inputs and n_weights must be non-negative.")
        for qubit in self.measured_qubits:
            if qubit < 0 or qubit >= self.n_qubits:
                raise ValueError(f"Measured qubit {qubit} out of range.")
        for gate in self.gates:
            gate.validate(self.n_qubits)
            if gate.source == "input" and gate.index is not None and gate.index >= self.n_inputs:
                raise ValueError(f"Input index {gate.index} out of range for n_inputs={self.n_inputs}.")
            if gate.source == "weight" and gate.index is not None and gate.index >= self.n_weights:
                raise ValueError(f"Weight index {gate.index} out of range for n_weights={self.n_weights}.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "n_qubits": self.n_qubits,
            "n_inputs": self.n_inputs,
            "n_weights": self.n_weights,
            "measured_qubits": list(self.measured_qubits),
            "gates": [gate.to_dict() for gate in self.gates],
            "meta": dict(self.meta),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CircuitSpec":
        spec = cls(
            name=str(data["name"]),
            n_qubits=int(data["n_qubits"]),
            n_inputs=int(data["n_inputs"]),
            n_weights=int(data["n_weights"]),
            measured_qubits=[int(q) for q in data["measured_qubits"]],
            gates=[GateSpec.from_dict(gate) for gate in data["gates"]],
            meta=dict(data.get("meta", {})),
        )
        spec.validate()
        return spec

