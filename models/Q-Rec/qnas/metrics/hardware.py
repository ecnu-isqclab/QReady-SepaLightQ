from __future__ import annotations

from qnas.common.gate_set import gate_info
from qnas.common.schema import CircuitSpec


def hardware_cost(
    spec: CircuitSpec,
    *,
    gate_costs: dict[str, float] | None = None,
    edge_costs: dict[tuple[int, int], float] | None = None,
) -> float:
    total = 0.0
    for gate in spec.gates:
        cost = gate_costs.get(gate.op, gate_info(gate.op).hardware_cost) if gate_costs else gate_info(gate.op).hardware_cost
        if edge_costs and len(gate.wires) == 2:
            edge = (gate.wires[0], gate.wires[1])
            cost += edge_costs.get(edge, edge_costs.get((edge[1], edge[0]), 0.0))
        total += float(cost)
    return total

