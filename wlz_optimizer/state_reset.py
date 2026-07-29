"""Pure-local reset planning for mutable and aliased evaluation inputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from .schemas import InputContract


@dataclass(frozen=True)
class StateResetGroup:
    """Inputs that must be restored together from one pristine snapshot."""

    tensor_names: Tuple[str, ...]
    preserve_alias: bool

    def __post_init__(self) -> None:
        if not self.tensor_names:
            raise ValueError("a state reset group must contain at least one tensor")
        if any(not isinstance(name, str) or not name.strip() for name in self.tensor_names):
            raise ValueError("state reset tensor names must be non-empty strings")
        if len(self.tensor_names) != len(set(self.tensor_names)):
            raise ValueError("state reset tensor names must be unique within a group")
        if not isinstance(self.preserve_alias, bool):
            raise ValueError("preserve_alias must be a boolean")
        if self.preserve_alias != (len(self.tensor_names) > 1):
            raise ValueError("preserve_alias must be true exactly for multi-tensor groups")


@dataclass(frozen=True)
class StateResetPlan:
    """Deterministic reset responsibilities; no tensors are allocated or copied here."""

    reset_groups: Tuple[StateResetGroup, ...]
    untouched_tensors: Tuple[str, ...]
    strategy: str = "clone_pristine_per_run"

    def __post_init__(self) -> None:
        if self.strategy != "clone_pristine_per_run":
            raise ValueError(f"unsupported state reset strategy: {self.strategy}")
        reset_names = [name for group in self.reset_groups for name in group.tensor_names]
        all_names = reset_names + list(self.untouched_tensors)
        if len(all_names) != len(set(all_names)):
            raise ValueError("a tensor cannot have multiple state reset responsibilities")


def plan_state_reset(inputs: InputContract) -> StateResetPlan:
    """Plan complete restoration of mutable tensors and their alias groups."""

    alias_members = {
        name
        for alias_group in inputs.alias_groups
        for name in alias_group.tensor_names
    }
    reset_groups = []
    untouched = []

    for alias_group in inputs.alias_groups:
        names = tuple(sorted(alias_group.tensor_names))
        if any(inputs.tensors[name].mutable for name in names):
            reset_groups.append(StateResetGroup(names, preserve_alias=True))
        else:
            untouched.extend(names)

    for name, tensor in inputs.tensors.items():
        if name in alias_members:
            continue
        if tensor.mutable:
            reset_groups.append(StateResetGroup((name,), preserve_alias=False))
        else:
            untouched.append(name)

    return StateResetPlan(
        reset_groups=tuple(sorted(reset_groups, key=lambda group: group.tensor_names)),
        untouched_tensors=tuple(sorted(untouched)),
    )
