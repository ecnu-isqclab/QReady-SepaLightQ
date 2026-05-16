import importlib
import inspect

import torch
import torch.nn as nn


def _get_model_class(module, module_name):
    if not hasattr(module, "MODEL_CLASS"):
        raise AttributeError(
            f"{module_name} must define MODEL_CLASS, e.g. MODEL_CLASS = YourModel"
        )

    model_cls = module.MODEL_CLASS
    if not inspect.isclass(model_cls):
        raise TypeError(f"{module_name}.MODEL_CLASS must be a class, got {type(model_cls)!r}")
    if not issubclass(model_cls, nn.Module):
        raise TypeError(f"{module_name}.MODEL_CLASS must be a subclass of torch.nn.Module")
    return model_cls


def _build_init_kwargs(model_cls, candidate_kwargs):
    signature = inspect.signature(model_cls.__init__)
    parameters = signature.parameters
    accepts_var_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )

    if accepts_var_kwargs:
        return candidate_kwargs

    return {
        key: value
        for key, value in candidate_kwargs.items()
        if key in parameters
    }


def build_model(
    module_name,
    anchors_mask,
    num_classes,
    phi="l",
    input_shape=(640, 640),
    pretrained=False,
    **kwargs,
):
    """Build a model from a nets module that exposes MODEL_CLASS.

    Parameters shared by YOLO-style models stay explicit here. Model-specific
    structure options should be passed through kwargs.
    """
    if not module_name:
        raise ValueError("module_name must be a non-empty import path, e.g. 'nets.yolo'")

    module = importlib.import_module(module_name)
    model_cls = _get_model_class(module, module_name)
    candidate_kwargs = {
        "anchors_mask": anchors_mask,
        "num_classes": num_classes,
        "phi": phi,
        "input_shape": input_shape,
        "pretrained": pretrained,
        **kwargs,
    }
    init_kwargs = _build_init_kwargs(model_cls, candidate_kwargs)

    try:
        return model_cls(**init_kwargs)
    except TypeError as exc:
        raise TypeError(
            f"Failed to instantiate {module_name}.{model_cls.__name__}. "
            "Model classes used with utils.model_builder.build_model should accept "
            "the required YOLO construction parameters they need. The builder will "
            "skip unsupported optional parameters such as input_shape for legacy models."
        ) from exc


def load_model_weights(model, weight_path, device="cpu", load_policy="strict"):
    """Load weights into a model and return load statistics.

    load_policy:
        strict        - require the checkpoint to match the model state dict.
        shape_match   - load only keys that exist in the model with identical shape.
        channel_slice - load same-name tensors by cropping each dimension to the
                        target shape. Detection heads are skipped unless their
                        shapes match exactly, because class/output layouts differ.
    """
    if not weight_path:
        raise ValueError("weight_path must be a non-empty path to a checkpoint")

    checkpoint = torch.load(weight_path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Expected checkpoint at {weight_path!r} to be a state dict")

    if load_policy == "strict":
        model.load_state_dict(checkpoint)
        model_keys = set(model.state_dict().keys())
        checkpoint_keys = set(checkpoint.keys())
        return {
            "weight_path": weight_path,
            "load_policy": load_policy,
            "loaded_keys": len(checkpoint_keys),
            "skipped_keys": 0,
            "missing_keys": len(model_keys - checkpoint_keys),
            "unexpected_keys": len(checkpoint_keys - model_keys),
        }

    if load_policy == "shape_match":
        model_dict = model.state_dict()
        matched = {}
        skipped = []

        for key, value in checkpoint.items():
            if key in model_dict and tuple(model_dict[key].shape) == tuple(value.shape):
                matched[key] = value
            else:
                skipped.append(key)

        model_dict.update(matched)
        model.load_state_dict(model_dict)

        return {
            "weight_path": weight_path,
            "load_policy": load_policy,
            "loaded_keys": len(matched),
            "skipped_keys": len(skipped),
            "missing_keys": len(set(model_dict.keys()) - set(matched.keys())),
            "unexpected_keys": len([key for key in skipped if key not in model_dict]),
        }

    if load_policy == "channel_slice":
        model_dict = model.state_dict()
        loaded = {}
        sliced = {}
        skipped = []
        skipped_detection_heads = []

        for key, value in checkpoint.items():
            if key not in model_dict:
                skipped.append(key)
                continue

            target = model_dict[key]
            if tuple(target.shape) == tuple(value.shape):
                loaded[key] = value
                continue

            if key.startswith("yolo_head"):
                skipped_detection_heads.append(key)
                continue

            if target.ndim != value.ndim:
                skipped.append(key)
                continue

            new_value = target.clone()
            slices = tuple(slice(0, min(source_dim, target_dim)) for source_dim, target_dim in zip(value.shape, target.shape))
            new_value[slices] = value[slices].to(device=target.device, dtype=target.dtype)
            sliced[key] = new_value

        model_dict.update(loaded)
        model_dict.update(sliced)
        model.load_state_dict(model_dict)

        loaded_keys = set(loaded.keys()) | set(sliced.keys())
        return {
            "weight_path": weight_path,
            "load_policy": load_policy,
            "loaded_keys": len(loaded_keys),
            "exact_loaded_keys": len(loaded),
            "sliced_loaded_keys": len(sliced),
            "skipped_keys": len(skipped) + len(skipped_detection_heads),
            "skipped_detection_head_keys": len(skipped_detection_heads),
            "missing_keys": len(set(model_dict.keys()) - loaded_keys),
            "unexpected_keys": len([key for key in skipped if key not in model_dict]),
        }

    raise ValueError("load_policy must be one of: 'strict', 'shape_match', 'channel_slice'")


__all__ = ["build_model", "load_model_weights"]
