from __future__ import annotations

import torch.nn as nn


def collect_arch_parameters(module: nn.Module) -> list[nn.Parameter]:
    params: list[nn.Parameter] = []
    seen: set[int] = set()
    for child in module.modules():
        if hasattr(child, "arch_parameters"):
            for param in child.arch_parameters():
                if id(param) not in seen:
                    params.append(param)
                    seen.add(id(param))
    return params


def split_weight_and_arch_parameters(module: nn.Module) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    arch_params = collect_arch_parameters(module)
    arch_ids = {id(param) for param in arch_params}
    weight_params = [param for param in module.parameters() if id(param) not in arch_ids]
    return weight_params, arch_params

