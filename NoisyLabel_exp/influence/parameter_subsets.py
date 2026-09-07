"""Model-aware parameter-subset resolution for influence computations."""

from __future__ import annotations

from collections import OrderedDict
import re

import torch
from torch import nn


SUBSET_NAMES = (
    "fc",
    "fc_bn",
    "final_block_head",
    "final_stage_head",
    "full",
)


def _select_parameters(
    model: nn.Module,
    predicate,
) -> OrderedDict[str, torch.nn.Parameter]:
    return OrderedDict(
        (name, parameter)
        for name, parameter in model.named_parameters()
        if predicate(name)
    )


def _prefix_predicate(prefixes: tuple[str, ...]):
    return lambda name: any(name.startswith(prefix) for prefix in prefixes)


def _head_prefixes(model: nn.Module) -> tuple[str, ...]:
    if hasattr(model, "fc"):
        return ("fc.",)
    if hasattr(model, "classifier"):
        return ("classifier.",)
    raise ValueError(
        "Could not identify the model head; expected an 'fc' or 'classifier' module"
    )


def _last_parameterized_child(module: nn.Module) -> str:
    children = [
        name
        for name, child in module.named_children()
        if any(True for _ in child.parameters())
    ]
    if not children:
        raise ValueError("Module has no parameterized children")

    numeric = [name for name in children if name.isdigit()]
    if numeric:
        return max(numeric, key=int)
    return children[-1]


def _last_stage_name(model: nn.Module) -> str:
    stages = [
        name
        for name, child in model.named_children()
        if re.fullmatch(r"layer\d+", name) and any(True for _ in child.parameters())
    ]
    if stages:
        return stages[-1]
    if hasattr(model, "features"):
        return "features"
    raise ValueError("Could not identify the final feature stage")


def _final_block_prefixes(model: nn.Module) -> tuple[str, ...]:
    explicit = getattr(model, "influence_final_block_prefixes", None)
    if explicit is not None:
        return tuple(explicit)

    stage_name = _last_stage_name(model)
    stage = getattr(model, stage_name)
    block_name = _last_parameterized_child(stage)
    return (f"{stage_name}.{block_name}.",) + _head_prefixes(model)


def _final_stage_prefixes(model: nn.Module) -> tuple[str, ...]:
    explicit = getattr(model, "influence_final_stage_prefixes", None)
    if explicit is not None:
        return tuple(explicit)

    stage_name = _last_stage_name(model)
    return (f"{stage_name}.",) + _head_prefixes(model)


def _is_bn(name: str) -> bool:
    return ".bn" in f".{name}" or ".downsample.1." in f".{name}" or name.startswith("bn")


def resolve_parameter_subset(
    model: nn.Module,
    subset_name: str,
) -> tuple[OrderedDict[str, torch.nn.Parameter], int]:
    if subset_name not in SUBSET_NAMES:
        raise ValueError(
            f"Unknown parameter subset {subset_name!r}; expected one of "
            f"{list(SUBSET_NAMES)}"
        )

    if subset_name == "full":
        predicate = lambda name: True
    elif subset_name == "fc":
        predicate = _prefix_predicate(_head_prefixes(model))
    elif subset_name == "fc_bn":
        head_prefixes = _head_prefixes(model)
        predicate = lambda name: name.startswith(head_prefixes) or _is_bn(name)
    elif subset_name == "final_block_head":
        predicate = _prefix_predicate(_final_block_prefixes(model))
    else:
        predicate = _prefix_predicate(_final_stage_prefixes(model))

    parameters = _select_parameters(model, predicate)
    if not parameters:
        raise ValueError(f"Parameter subset {subset_name!r} is empty")
    return parameters, sum(parameter.numel() for parameter in parameters.values())
