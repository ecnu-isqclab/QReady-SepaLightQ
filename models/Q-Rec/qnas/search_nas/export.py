from __future__ import annotations

from pathlib import Path

from qnas.common.io import save_circuit


def export_search_arch(layer, path: str | Path, *, name: str = "search_arch") -> None:
    if not hasattr(layer, "export_circuit_spec"):
        raise TypeError("layer must provide export_circuit_spec(name=...).")
    save_circuit(layer.export_circuit_spec(name=name), path)

