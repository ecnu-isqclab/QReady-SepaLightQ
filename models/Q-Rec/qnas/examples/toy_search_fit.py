from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn as nn
import torch.nn.functional as F

from qnas.common.io import save_search_result
from qnas.search_nas.arch_parameters import split_weight_and_arch_parameters
from qnas.search_nas.search_space import SearchSpaceSpec
from qnas.search_nas.searchable_layer import SearchableQuantumLayer
from qnas.search_nas.tensorcircuit_search_layer import TensorCircuitSearchableQuantumLayer


class ToyQNASRegressor(nn.Module):
    def __init__(self, search_space: SearchSpaceSpec, backend: str, tc_backend: str):
        super().__init__()
        if backend == "torch":
            self.q_layer = SearchableQuantumLayer(search_space)
        elif backend == "tensorcircuit":
            self.q_layer = TensorCircuitSearchableQuantumLayer(search_space, tc_backend=tc_backend)
        else:
            raise ValueError(f"Unknown backend {backend!r}. Use 'torch' or 'tensorcircuit'.")
        self.head = nn.Linear(search_space.n_qubits, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.q_layer(x))


def make_toy_data(num_samples: int, n_qubits: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    x = (torch.rand(num_samples, n_qubits, generator=generator) * 2.0 - 1.0) * math.pi
    y = torch.sin(x[:, 0])
    if n_qubits > 1:
        y = y + 0.5 * torch.cos(x[:, 1])
    if n_qubits > 3:
        y = y - 0.25 * torch.sin(x[:, 2] * x[:, 3])
    y = y.unsqueeze(1)
    return x, y


def parse_args():
    parser = argparse.ArgumentParser(description="Toy fit test for qnas.search_nas.")
    parser.add_argument("--n-qubits", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--num-samples", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--weight-lr", type=float, default=1e-2)
    parser.add_argument("--arch-lr", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", default="experiments/qnas/toy_search/best_arch.json")
    parser.add_argument("--min-loss-ratio", type=float, default=0.95)
    parser.add_argument("--backend", default="torch", choices=["torch", "tensorcircuit"])
    parser.add_argument("--tc-backend", default="pytorch", choices=["pytorch", "tensorflow"])
    return parser.parse_args()


def tensor_to_list(value: torch.Tensor):
    return value.detach().cpu().tolist()


def collect_search_parameters(model: ToyQNASRegressor) -> dict:
    return {
        "alpha_1q": tensor_to_list(model.q_layer.alpha_1q),
        "alpha_2q": tensor_to_list(model.q_layer.alpha_2q),
        "alpha_1q_softmax": tensor_to_list(torch.softmax(model.q_layer.alpha_1q, dim=-1)),
        "alpha_2q_softmax": tensor_to_list(torch.softmax(model.q_layer.alpha_2q, dim=-1)),
        "theta_1q": tensor_to_list(model.q_layer.theta_1q),
        "theta_2q": tensor_to_list(model.q_layer.theta_2q),
        "head_weight": tensor_to_list(model.head.weight),
        "head_bias": tensor_to_list(model.head.bias),
    }


def regression_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    error = pred - target
    mse = error.pow(2).mean()
    mae = error.abs().mean()
    rmse = torch.sqrt(mse)
    ss_res = error.pow(2).sum()
    ss_tot = (target - target.mean()).pow(2).sum().clamp_min(1e-12)
    r2 = 1.0 - ss_res / ss_tot
    return {
        "mse": float(mse.item()),
        "mae": float(mae.item()),
        "rmse": float(rmse.item()),
        "r2": float(r2.item()),
    }


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    search_space = SearchSpaceSpec(n_qubits=args.n_qubits, n_layers=args.n_layers)
    model = ToyQNASRegressor(search_space, args.backend, args.tc_backend)
    x, y = make_toy_data(args.num_samples, args.n_qubits, args.seed)

    weight_params, arch_params = split_weight_and_arch_parameters(model)
    weight_optimizer = torch.optim.Adam(weight_params, lr=args.weight_lr)
    arch_optimizer = torch.optim.Adam(arch_params, lr=args.arch_lr)

    with torch.no_grad():
        initial_loss = F.mse_loss(model(x), y).item()

    for step in range(args.steps):
        indices = torch.randint(0, args.num_samples, (args.batch_size,))
        xb = x[indices]
        yb = y[indices]
        loss = F.mse_loss(model(xb), yb)

        weight_optimizer.zero_grad()
        arch_optimizer.zero_grad()
        loss.backward()
        weight_optimizer.step()
        arch_optimizer.step()

        if step == 0 or (step + 1) % 50 == 0 or step == args.steps - 1:
            print(f"step={step + 1:03d} loss={loss.item():.6f}")

    with torch.no_grad():
        final_pred = model(x)
        final_loss = F.mse_loss(final_pred, y).item()
        final_metrics = regression_metrics(final_pred, y)

    spec = model.q_layer.export_circuit_spec(name="toy_search_arch")
    loss_ratio = final_loss / max(initial_loss, 1e-12)
    save_search_result(
        spec,
        args.out,
        parameters=collect_search_parameters(model),
        training={
            "backend": args.backend,
            "tc_backend": args.tc_backend,
            "seed": args.seed,
            "steps": args.steps,
            "num_samples": args.num_samples,
            "batch_size": args.batch_size,
            "weight_lr": args.weight_lr,
            "arch_lr": args.arch_lr,
            "initial_loss": initial_loss,
            "final_loss": final_loss,
            "loss_ratio": loss_ratio,
            "metrics": final_metrics,
        },
    )

    print(f"initial_loss={initial_loss:.6f}")
    print(f"final_loss={final_loss:.6f}")
    print(f"loss_ratio={loss_ratio:.6f}")
    print(
        "metrics="
        f"mse:{final_metrics['mse']:.6f} "
        f"mae:{final_metrics['mae']:.6f} "
        f"rmse:{final_metrics['rmse']:.6f} "
        f"r2:{final_metrics['r2']:.6f}"
    )
    print(f"exported={args.out}")
    print(f"n_gates={len(spec.gates)}")
    print(f"n_weights={spec.n_weights}")

    if loss_ratio > args.min_loss_ratio:
        raise SystemExit(
            f"Final loss did not improve enough: initial={initial_loss:.6f}, final={final_loss:.6f}."
        )


if __name__ == "__main__":
    main()
