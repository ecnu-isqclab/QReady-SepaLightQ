from __future__ import annotations

import torch
import torch.nn as nn

from qnas.common.gate_set import gate_info
from qnas.common.schema import CircuitSpec, GateSpec
from qnas.search_nas.search_space import SearchSpaceSpec


class SearchableQuantumLayer(nn.Module):
    """Online NAS layer parameter container.

    This class owns architecture logits alpha_* and variational weights theta_*.
    The executable soft TensorCircuit forward should be implemented here later;
    export_circuit_spec() already converts the current argmax architecture into
    the shared CircuitSpec IR used by the sampled path.
    """

    def __init__(self, search_space: SearchSpaceSpec):
        super().__init__()
        self.search_space = search_space
        n_edges = len(search_space.twoq_edges)
        self.alpha_1q = nn.Parameter(torch.zeros(search_space.n_layers, search_space.n_qubits, len(search_space.oneq_ops)))
        self.alpha_2q = nn.Parameter(torch.zeros(search_space.n_layers, n_edges, len(search_space.twoq_ops)))
        self.theta_1q = nn.Parameter(torch.randn(search_space.n_layers, search_space.n_qubits) * 0.02)
        self.theta_2q = nn.Parameter(torch.randn(search_space.n_layers, n_edges) * 0.02)

    @staticmethod
    def _as_batch_angle(theta: torch.Tensor, batch_size: int) -> torch.Tensor:
        if theta.ndim == 0:
            return theta.expand(batch_size)
        return theta

    @staticmethod
    def _rotation_matrix(op: str, theta: torch.Tensor, batch_size: int) -> torch.Tensor:
        theta = SearchableQuantumLayer._as_batch_angle(theta, batch_size)
        half = theta / 2
        zeros = torch.zeros_like(theta)
        c = torch.cos(half).to(torch.complex64)
        s = torch.sin(half).to(torch.complex64)
        z = zeros.to(torch.complex64)
        one = torch.ones_like(theta).to(torch.complex64)
        minus_i_s = -1j * s

        if op == "rx":
            return torch.stack(
                [
                    torch.stack([c, minus_i_s], dim=-1),
                    torch.stack([minus_i_s, c], dim=-1),
                ],
                dim=-2,
            )
        if op == "ry":
            return torch.stack(
                [
                    torch.stack([c, -s], dim=-1),
                    torch.stack([s, c], dim=-1),
                ],
                dim=-2,
            )
        if op == "rz":
            return torch.stack(
                [
                    torch.stack([torch.exp(-0.5j * theta), z], dim=-1),
                    torch.stack([z, torch.exp(0.5j * theta)], dim=-1),
                ],
                dim=-2,
            )
        if op in {"skip", "identity"}:
            return torch.stack(
                [
                    torch.stack([one, z], dim=-1),
                    torch.stack([z, one], dim=-1),
                ],
                dim=-2,
            )
        if op in {"cx", "cz"}:
            return torch.stack(
                [
                    torch.stack([z if op == "cx" else one, one if op == "cx" else z], dim=-1),
                    torch.stack([one if op == "cx" else z, z if op == "cx" else -one], dim=-1),
                ],
                dim=-2,
            )
        raise NotImplementedError(f"Torch search forward does not support gate {op!r}.")

    def _apply_single(self, state: torch.Tensor, mat: torch.Tensor, wire: int) -> torch.Tensor:
        batch_size = state.shape[0]
        n_qubits = self.search_space.n_qubits
        tensor = state.reshape(batch_size, *([2] * n_qubits))
        other_dims = [idx + 1 for idx in range(n_qubits) if idx != wire]
        perm = [0] + other_dims + [wire + 1]
        moved = tensor.permute(perm).reshape(batch_size, -1, 2)
        moved = torch.einsum("bij,brj->bri", mat, moved)
        moved = moved.reshape(batch_size, *([2] * (n_qubits - 1)), 2)
        inv = [0] * (n_qubits + 1)
        for idx, dim in enumerate(perm):
            inv[dim] = idx
        return moved.permute(inv).reshape(batch_size, -1)

    def _apply_controlled(self, state: torch.Tensor, mat: torch.Tensor, control: int, target: int) -> torch.Tensor:
        batch_size = state.shape[0]
        n_qubits = self.search_space.n_qubits
        tensor = state.reshape(batch_size, *([2] * n_qubits))
        other_dims = [idx + 1 for idx in range(n_qubits) if idx not in {control, target}]
        perm = [0] + other_dims + [control + 1, target + 1]
        moved = tensor.permute(perm).reshape(batch_size, -1, 2, 2)
        control_zero = moved[:, :, 0, :]
        control_one = torch.einsum("bij,brj->bri", mat, moved[:, :, 1, :])
        moved = torch.stack([control_zero, control_one], dim=2)
        moved = moved.reshape(batch_size, *([2] * (n_qubits - 2)), 2, 2)
        inv = [0] * (n_qubits + 1)
        for idx, dim in enumerate(perm):
            inv[dim] = idx
        return moved.permute(inv).reshape(batch_size, -1)

    def _expectations(self, state: torch.Tensor) -> torch.Tensor:
        batch_size = state.shape[0]
        n_qubits = self.search_space.n_qubits
        probs = state.abs().pow(2).reshape(batch_size, *([2] * n_qubits))
        outputs = []
        for qubit in range(n_qubits):
            plus = probs.select(qubit + 1, 0).sum(dim=tuple(range(1, n_qubits)))
            minus = probs.select(qubit + 1, 1).sum(dim=tuple(range(1, n_qubits)))
            outputs.append(plus - minus)
        return torch.stack(outputs, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Differentiable torch statevector forward for small toy/search tests."""
        space = self.search_space
        batch_size = x.shape[0]
        state = torch.zeros(batch_size, 2 ** space.n_qubits, device=x.device, dtype=torch.complex64)
        state[:, 0] = 1.0 + 0.0j

        for qubit in range(space.n_qubits):
            mat = self._rotation_matrix(space.input_encoding, x[:, qubit], batch_size).to(x.device)
            state = self._apply_single(state, mat, qubit)

        prob_1q = torch.softmax(self.alpha_1q, dim=-1)
        prob_2q = torch.softmax(self.alpha_2q, dim=-1)
        for layer_idx in range(space.n_layers):
            for qubit in range(space.n_qubits):
                for op_idx, op in enumerate(space.oneq_ops):
                    if op in {"skip", "identity"}:
                        continue
                    theta = prob_1q[layer_idx, qubit, op_idx] * self.theta_1q[layer_idx, qubit]
                    mat = self._rotation_matrix(op, theta, batch_size).to(x.device)
                    state = self._apply_single(state, mat, qubit)

            for edge_idx, (control, target) in enumerate(space.twoq_edges):
                for op_idx, op in enumerate(space.twoq_ops):
                    if op in {"skip", "identity"}:
                        continue
                    theta = prob_2q[layer_idx, edge_idx, op_idx] * self.theta_2q[layer_idx, edge_idx]
                    if op in {"crx", "cry", "crz"}:
                        mat = self._rotation_matrix(op[1:], theta, batch_size).to(x.device)
                        state = self._apply_controlled(state, mat, control, target)
                    elif op in {"cx", "cz"}:
                        mat = self._rotation_matrix(op, theta, batch_size).to(x.device)
                        state = self._apply_controlled(state, mat, control, target)
                    else:
                        raise NotImplementedError(f"Torch search forward does not support gate {op!r}.")

        return self._expectations(state).to(dtype=x.dtype)

    def arch_parameters(self) -> list[nn.Parameter]:
        return [self.alpha_1q, self.alpha_2q]

    def quantum_weight_parameters(self) -> list[nn.Parameter]:
        return [self.theta_1q, self.theta_2q]

    def export_circuit_spec(self, name: str = "search_arch") -> CircuitSpec:
        space = self.search_space
        gates: list[GateSpec] = []
        for qubit in range(space.n_qubits):
            gates.append(GateSpec(space.input_encoding, [qubit], "input", qubit))

        weight_index = 0
        oneq_choice = self.alpha_1q.detach().argmax(dim=-1).cpu()
        twoq_choice = self.alpha_2q.detach().argmax(dim=-1).cpu()
        for layer_idx in range(space.n_layers):
            for qubit in range(space.n_qubits):
                op = space.oneq_ops[int(oneq_choice[layer_idx, qubit])]
                if op in {"skip", "identity"}:
                    continue
                info = gate_info(op)
                source = "weight" if info.num_params else "none"
                index = weight_index if info.num_params else None
                gates.append(GateSpec(op, [qubit], source, index))
                weight_index += info.num_params

            for edge_idx, edge in enumerate(space.twoq_edges):
                op = space.twoq_ops[int(twoq_choice[layer_idx, edge_idx])]
                if op in {"skip", "identity"}:
                    continue
                info = gate_info(op)
                source = "weight" if info.num_params else "none"
                index = weight_index if info.num_params else None
                gates.append(GateSpec(op, list(edge), source, index))
                weight_index += info.num_params

        spec = CircuitSpec(
            name=name,
            n_qubits=space.n_qubits,
            n_inputs=space.n_qubits,
            n_weights=weight_index,
            measured_qubits=list(range(space.n_qubits)),
            gates=gates,
            meta={"source": "search_nas", "search_space": space.__dict__},
        )
        spec.validate()
        return spec
