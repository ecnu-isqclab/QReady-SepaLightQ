from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GateInfo:
    op: str
    num_wires: int
    num_params: int
    hardware_cost: float = 1.0

    @property
    def is_two_qubit(self) -> bool:
        return self.num_wires == 2


GATE_SET: dict[str, GateInfo] = {
    "skip": GateInfo("skip", 1, 0, 0.0),
    "identity": GateInfo("identity", 1, 0, 0.0),
    "rx": GateInfo("rx", 1, 1, 1.0),
    "ry": GateInfo("ry", 1, 1, 1.0),
    "rz": GateInfo("rz", 1, 1, 0.5),
    "h": GateInfo("h", 1, 0, 0.8),
    "sx": GateInfo("sx", 1, 0, 0.8),
    "x": GateInfo("x", 1, 0, 0.8),
    "cx": GateInfo("cx", 2, 0, 8.0),
    "cz": GateInfo("cz", 2, 0, 7.0),
    "crx": GateInfo("crx", 2, 1, 9.0),
    "cry": GateInfo("cry", 2, 1, 9.0),
    "crz": GateInfo("crz", 2, 1, 8.5),
    "rxx": GateInfo("rxx", 2, 1, 9.5),
    "ryy": GateInfo("ryy", 2, 1, 9.5),
    "rzz": GateInfo("rzz", 2, 1, 9.0),
}


def gate_info(op: str) -> GateInfo:
    op = op.lower()
    if op not in GATE_SET:
        raise KeyError(f"Unsupported gate {op!r}.")
    return GATE_SET[op]
