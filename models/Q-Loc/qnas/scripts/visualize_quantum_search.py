from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def resolve_input(path: str | Path) -> Path:
    path = Path(path)
    if path.is_dir():
        checkpoint_dir = path / "checkpoints"
        candidates = [
            checkpoint_dir / "best_quantum_search.json",
            checkpoint_dir / "last_quantum_search.json",
            path / "best_quantum_search.json",
            path / "last_quantum_search.json",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise SystemExit(f"No quantum search json found under {path}")
    return path


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize YOLOv7 QNAS/QNN parameters.")
    parser.add_argument(
        "path",
        help="Path to best_quantum_search.json, last_quantum_search.json, or a run directory.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for png files. Defaults to <json parent>/visualizations.",
    )
    parser.add_argument("--prefix", default="", help="Optional filename prefix.")
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def load_payload(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if "parameters" not in payload:
        raise SystemExit(f"{path} does not contain exported QNN parameters.")
    return payload


def heatmap(ax, values, xlabels, ylabels, title, cbar_label, cmap="viridis"):
    values = np.asarray(values, dtype=float)
    image = ax.imshow(values, aspect="auto", cmap=cmap)
    ax.set_title(title)
    ax.set_xticks(np.arange(len(xlabels)))
    ax.set_xticklabels(xlabels)
    ax.set_yticks(np.arange(len(ylabels)))
    ax.set_yticklabels(ylabels)
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            ax.text(col, row, f"{values[row, col]:.3f}", ha="center", va="center", fontsize=7, color="white")
    cbar = plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel(cbar_label, rotation=-90, va="bottom")


def save_theta_fig(parameters: dict, output_path: Path, dpi: int) -> None:
    theta_1q = np.asarray(parameters["theta_1q"], dtype=float)
    theta_2q = np.asarray(parameters["theta_2q"], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    heatmap(
        axes[0],
        theta_1q,
        [f"q{idx}" for idx in range(theta_1q.shape[1])],
        [f"layer {idx}" for idx in range(theta_1q.shape[0])],
        "1-qubit theta",
        "angle",
        cmap="coolwarm",
    )
    heatmap(
        axes[1],
        theta_2q,
        [f"edge {idx}" for idx in range(theta_2q.shape[1])],
        [f"layer {idx}" for idx in range(theta_2q.shape[0])],
        "2-qubit theta",
        "angle",
        cmap="coolwarm",
    )
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=dpi)
    plt.close(fig)


def save_alpha_fig(parameters: dict, circuit: dict, output_path: Path, dpi: int) -> None:
    alpha_1q = np.asarray(parameters.get("alpha_1q_softmax", parameters["alpha_1q"]), dtype=float)
    alpha_2q = np.asarray(parameters.get("alpha_2q_softmax", parameters["alpha_2q"]), dtype=float)
    meta = circuit.get("meta", {}).get("search_space", {})
    oneq_ops = meta.get("oneq_ops", [f"op{idx}" for idx in range(alpha_1q.shape[-1])])
    twoq_ops = meta.get("twoq_ops", [f"op{idx}" for idx in range(alpha_2q.shape[-1])])

    n_layers = alpha_1q.shape[0]
    fig, axes = plt.subplots(n_layers, 2, figsize=(10, max(3, 2.5 * n_layers)))
    if n_layers == 1:
        axes = np.asarray([axes])
    for layer in range(n_layers):
        heatmap(
            axes[layer, 0],
            alpha_1q[layer],
            oneq_ops,
            [f"q{idx}" for idx in range(alpha_1q.shape[1])],
            f"Layer {layer} 1-qubit op probability",
            "probability",
        )
        heatmap(
            axes[layer, 1],
            alpha_2q[layer],
            twoq_ops,
            [f"edge {idx}" for idx in range(alpha_2q.shape[1])],
            f"Layer {layer} 2-qubit op probability",
            "probability",
        )
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=dpi)
    plt.close(fig)


def gate_label(gate: dict, theta_values: dict[tuple[tuple[int, ...], int], float]) -> str:
    op = gate["op"]
    if gate.get("source") != "weight":
        return op
    key = tuple(gate.get("wires", [])), gate.get("index")
    value = theta_values.get(key)
    if value is None:
        return f"{op}\n#{gate.get('index')}"
    return f"{op}\n{value:.2f}"


def theta_lookup(circuit: dict, parameters: dict) -> dict[tuple[tuple[int, ...], int], float]:
    lookup: dict[tuple[tuple[int, ...], int], float] = {}
    theta_1q = parameters.get("theta_1q", [])
    theta_2q = parameters.get("theta_2q", [])
    oneq_cursor = 0
    twoq_cursor = 0
    n_qubits = int(circuit["n_qubits"])
    edges_per_layer = max(1, len(theta_2q[0]) if theta_2q else n_qubits - 1)
    for gate in circuit["gates"]:
        if gate.get("source") != "weight":
            continue
        wires = tuple(gate.get("wires", []))
        index = int(gate["index"])
        if len(wires) == 1:
            layer = oneq_cursor // n_qubits
            qubit = oneq_cursor % n_qubits
            lookup[(wires, index)] = float(theta_1q[layer][qubit])
            oneq_cursor += 1
        elif len(wires) == 2:
            layer = twoq_cursor // edges_per_layer
            edge = twoq_cursor % edges_per_layer
            lookup[(wires, index)] = float(theta_2q[layer][edge])
            twoq_cursor += 1
    return lookup


def save_circuit_fig(payload: dict, output_path: Path, dpi: int) -> None:
    circuit = payload["circuit"]
    gates = circuit["gates"]
    n_qubits = int(circuit["n_qubits"])
    theta_values = theta_lookup(circuit, payload["parameters"])

    fig_width = max(10, 0.55 * len(gates))
    fig, ax = plt.subplots(figsize=(fig_width, 2 + 0.7 * n_qubits))
    ys = np.arange(n_qubits)[::-1]
    for qubit, y in enumerate(ys):
        ax.hlines(y, -0.5, len(gates) - 0.5, color="#39424e", linewidth=1.4)
        ax.text(-1.0, y, f"q{qubit}", ha="right", va="center", fontsize=10)

    for col, gate in enumerate(gates):
        wires = gate.get("wires", [])
        wire_ys = [ys[wire] for wire in wires]
        color = "#5b8def" if gate.get("source") == "input" else "#f08a5d"
        if len(wire_ys) == 2:
            ax.vlines(col, min(wire_ys), max(wire_ys), color="#6b7280", linewidth=1.2)
        for y in wire_ys:
            ax.text(
                col,
                y,
                gate_label(gate, theta_values),
                ha="center",
                va="center",
                fontsize=7,
                bbox={"boxstyle": "round,pad=0.25", "facecolor": color, "edgecolor": "#20242a", "linewidth": 0.8},
            )

    ax.set_ylim(-0.8, n_qubits - 0.2)
    ax.set_xlim(-1.2, len(gates) - 0.4)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Exported QNN circuit (gate labels include theta when available)")
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    input_path = resolve_input(args.path)
    payload = load_payload(input_path)
    output_dir = Path(args.output_dir) if args.output_dir else input_path.parent / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{args.prefix}_" if args.prefix else ""

    theta_path = output_dir / f"{prefix}qnn_theta_heatmaps.png"
    alpha_path = output_dir / f"{prefix}qnn_alpha_probabilities.png"
    circuit_path = output_dir / f"{prefix}qnn_circuit.png"

    save_theta_fig(payload["parameters"], theta_path, args.dpi)
    save_alpha_fig(payload["parameters"], payload["circuit"], alpha_path, args.dpi)
    save_circuit_fig(payload, circuit_path, args.dpi)

    print(f"input: {input_path}")
    print(f"saved: {theta_path}")
    print(f"saved: {alpha_path}")
    print(f"saved: {circuit_path}")


if __name__ == "__main__":
    main()
