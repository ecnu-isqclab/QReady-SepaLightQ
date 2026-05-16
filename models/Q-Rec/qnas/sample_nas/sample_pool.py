from __future__ import annotations

import random
from pathlib import Path

from qnas.common.io import save_circuit
from qnas.common.schema import CircuitSpec
from qnas.sample_nas.sampler import sample_layered_circuit


def sample_circuit_pool(
    *,
    out_dir: str | Path,
    num_circuits: int,
    seed: int = 0,
    name_prefix: str = "circ",
    **sampler_kwargs,
) -> list[CircuitSpec]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    specs: list[CircuitSpec] = []
    for idx in range(num_circuits):
        name = f"{name_prefix}_{idx + 1:06d}"
        spec = sample_layered_circuit(name=name, rng=rng, **sampler_kwargs)
        save_circuit(spec, out_dir / f"{name}.json")
        specs.append(spec)
    return specs

