from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SearchSpaceSpec:
    n_qubits: int = 4
    n_layers: int = 3
    oneq_ops: tuple[str, ...] = ("rx", "ry", "rz")
    twoq_ops: tuple[str, ...] = ("crx", "cry", "crz")
    entangle_pattern: str = "linear"
    input_encoding: str = "ry"

    @property
    def twoq_edges(self) -> list[tuple[int, int]]:
        if self.entangle_pattern == "linear":
            return [(idx, idx + 1) for idx in range(self.n_qubits - 1)]
        if self.entangle_pattern == "ring":
            edges = [(idx, idx + 1) for idx in range(self.n_qubits - 1)]
            if self.n_qubits > 2:
                edges.append((self.n_qubits - 1, 0))
            return edges
        raise ValueError(f"Unknown entangle_pattern {self.entangle_pattern!r}.")

