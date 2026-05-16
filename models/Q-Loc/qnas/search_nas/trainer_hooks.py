from __future__ import annotations

from qnas.search_nas.arch_parameters import split_weight_and_arch_parameters


def build_search_optimizers(model, *, weight_optimizer_cls, arch_optimizer_cls, weight_kwargs, arch_kwargs):
    weight_params, arch_params = split_weight_and_arch_parameters(model)
    return (
        weight_optimizer_cls(weight_params, **weight_kwargs),
        arch_optimizer_cls(arch_params, **arch_kwargs),
    )
