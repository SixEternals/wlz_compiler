"""Backend-neutral materialization and fresh-run storage isolation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Protocol, Tuple

from .schemas import EvaluationCase, TensorInputContract


class TensorBackend(Protocol):
    """Minimal allocation surface implemented later by a real tensor backend."""

    def seed(self, seed: int) -> None: ...

    def randn(self, *, shape: Tuple[int, ...], dtype: str) -> Any: ...

    def zeros(self, *, shape: Tuple[int, ...], dtype: str) -> Any: ...

    def full(
        self, *, shape: Tuple[int, ...], dtype: str, fill_value: int | float
    ) -> Any: ...

    def randint(
        self, *, shape: Tuple[int, ...], dtype: str, low: int, high: int
    ) -> Any: ...

    def literal(
        self, *, shape: Tuple[int, ...], dtype: str, values: Tuple[Any, ...]
    ) -> Any: ...

    def clone(self, value: Any) -> Any: ...

    def as_strided(
        self, value: Any, *, shape: Tuple[int, ...], strides: Tuple[int, ...]
    ) -> Any: ...


class UnsupportedInputContractError(ValueError):
    """The contract lacks evidence needed for safe materialization."""


@dataclass(frozen=True)
class ViewSpec:
    """Offset-zero view geometry backed by one materialized storage."""

    storage_key: str
    shape: Tuple[int, ...]
    strides: Tuple[int, ...]


@dataclass
class MaterializedInputs:
    """Inspectable values and their stable first-occurrence source order."""

    tensors: Dict[str, Any]
    scalars: Dict[str, Any]
    order: Tuple[Tuple[str, str], ...]
    seed: int
    input_signature: str
    storages: Dict[str, Any]
    tensor_storage_keys: Dict[str, str]
    view_specs: Dict[str, ViewSpec]


def materialize_inputs(
    case: EvaluationCase, backend: TensorBackend
) -> MaterializedInputs:
    """Materialize each bound source once, following execution argument order."""

    if not isinstance(case, EvaluationCase):
        raise TypeError("case must be an EvaluationCase")
    alias_by_tensor = _validate_alias_groups(case)

    backend.seed(case.seed)
    tensors: Dict[str, Any] = {}
    scalars: Dict[str, Any] = {}
    storages: Dict[str, Any] = {}
    tensor_storage_keys: Dict[str, str] = {}
    view_specs: Dict[str, ViewSpec] = {}
    prepared_views: Dict[str, Any] = {}
    order = []
    seen = set()

    for argument in case.execution.arguments:
        source = (argument.source_kind, argument.source_name)
        if source in seen:
            continue
        seen.add(source)
        order.append(source)
        if argument.source_kind == "tensor":
            name = argument.source_name
            contract = case.inputs.tensors[name]
            alias_names = alias_by_tensor.get(name)
            if alias_names:
                storage_key = "alias:" + "|".join(alias_names)
                if storage_key not in storages:
                    prototype = case.inputs.tensors[alias_names[0]]
                    specs = {
                        member: _view_spec(
                            storage_key, case.inputs.tensors[member]
                        )
                        for member in alias_names
                    }
                    storage_size = max(
                        _required_storage_size(spec) for spec in specs.values()
                    )
                    storages[storage_key] = _materialize_flat_storage(
                        prototype, storage_size, backend
                    )
                    for member, spec in specs.items():
                        tensor_storage_keys[member] = storage_key
                        view_specs[member] = spec
                        prepared_views[member] = backend.as_strided(
                            storages[storage_key],
                            shape=spec.shape,
                            strides=spec.strides,
                        )
                tensors[name] = prepared_views[name]
            elif contract.layout == "strided":
                storage_key = f"tensor:{name}"
                spec = _view_spec(storage_key, contract)
                storages[storage_key] = _materialize_flat_storage(
                    contract, _required_storage_size(spec), backend
                )
                tensor_storage_keys[name] = storage_key
                view_specs[name] = spec
                tensors[name] = backend.as_strided(
                    storages[storage_key], shape=spec.shape, strides=spec.strides
                )
            else:
                storage_key = f"tensor:{name}"
                storages[storage_key] = _materialize_tensor(contract, backend)
                tensor_storage_keys[name] = storage_key
                tensors[name] = storages[storage_key]
        else:
            scalars[argument.source_name] = case.inputs.scalars[argument.source_name]

    return MaterializedInputs(
        tensors=tensors,
        scalars=scalars,
        order=tuple(order),
        seed=case.seed,
        input_signature=case.input_signature(),
        storages=storages,
        tensor_storage_keys=tensor_storage_keys,
        view_specs=view_specs,
    )


def clone_inputs_for_run(
    pristine: MaterializedInputs, backend: TensorBackend
) -> MaterializedInputs:
    """Clone every pristine storage once and rebuild all views for one run."""

    cloned_storages = {
        key: backend.clone(storage) for key, storage in pristine.storages.items()
    }
    tensors = {}
    for name in pristine.tensors:
        storage = cloned_storages[pristine.tensor_storage_keys[name]]
        spec = pristine.view_specs.get(name)
        tensors[name] = (
            backend.as_strided(storage, shape=spec.shape, strides=spec.strides)
            if spec
            else storage
        )
    return MaterializedInputs(
        tensors=tensors,
        scalars=dict(pristine.scalars),
        order=pristine.order,
        seed=pristine.seed,
        input_signature=pristine.input_signature,
        storages=cloned_storages,
        tensor_storage_keys=dict(pristine.tensor_storage_keys),
        view_specs=dict(pristine.view_specs),
    )


def _validate_alias_groups(case: EvaluationCase) -> Dict[str, Tuple[str, ...]]:
    alias_by_tensor = {}
    for group in case.inputs.alias_groups:
        names = tuple(sorted(group.tensor_names))
        contracts = [case.inputs.tensors[name] for name in names]
        if len({contract.dtype for contract in contracts}) != 1 or any(
            contract.initializer != contracts[0].initializer for contract in contracts[1:]
        ):
            raise UnsupportedInputContractError(
                "alias group members must share dtype and initializer: " + ", ".join(names)
            )
        alias_by_tensor.update((name, names) for name in names)
    return alias_by_tensor


def _view_spec(storage_key: str, contract: TensorInputContract) -> ViewSpec:
    shape = tuple(contract.shape)
    strides = (
        tuple(contract.strides)
        if contract.strides is not None
        else _contiguous_strides(shape)
    )
    return ViewSpec(storage_key=storage_key, shape=shape, strides=strides)


def _contiguous_strides(shape: Tuple[int, ...]) -> Tuple[int, ...]:
    strides = []
    running = 1
    for dimension in reversed(shape):
        strides.append(running)
        running *= dimension
    return tuple(reversed(strides))


def _required_storage_size(spec: ViewSpec) -> int:
    if any(dimension == 0 for dimension in spec.shape):
        return 0
    return 1 + sum(
        (dimension - 1) * stride
        for dimension, stride in zip(spec.shape, spec.strides)
    )


def _materialize_flat_storage(
    contract: TensorInputContract, size: int, backend: TensorBackend
) -> Any:
    return _materialize_initializer(contract, shape=(size,), backend=backend)


def _materialize_tensor(contract: TensorInputContract, backend: TensorBackend) -> Any:
    return _materialize_initializer(contract, shape=tuple(contract.shape), backend=backend)


def _materialize_initializer(
    contract: TensorInputContract, *, shape: Tuple[int, ...], backend: TensorBackend
) -> Any:
    common = {"shape": shape, "dtype": contract.dtype}
    initializer = contract.initializer
    if initializer.kind == "randn":
        return backend.randn(**common)
    if initializer.kind == "zeros":
        return backend.zeros(**common)
    if initializer.kind == "full":
        return backend.full(**common, fill_value=initializer.parameters["fill_value"])
    if initializer.kind == "randint":
        return backend.randint(
            **common,
            low=initializer.parameters["low"],
            high=initializer.parameters["high"],
        )
    if initializer.kind == "literal":
        return backend.literal(
            **common,
            values=tuple(initializer.parameters["values"]),
        )
    raise UnsupportedInputContractError(
        f"unsupported tensor initializer: {initializer.kind}"
    )
