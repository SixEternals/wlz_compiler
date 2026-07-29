"""Optional Torch implementation of the B2 tensor backend protocol."""

from __future__ import annotations

import importlib
from typing import Any, Tuple


class TorchTensorBackend:
    """Materialize and clone tensors on one explicitly selected Torch device."""

    def __init__(self, device: str = "cpu", torch_module: Any = None) -> None:
        self.torch = torch_module or importlib.import_module("torch")
        self.device = self.torch.device(device)
        if self.device.type == "cuda" and not self.torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")

    def seed(self, seed: int) -> None:
        self.torch.manual_seed(seed)
        if self.device.type == "cuda":
            self.torch.cuda.manual_seed_all(seed)

    def randn(self, *, shape: Tuple[int, ...], dtype: str) -> Any:
        return self.torch.randn(shape, device=self.device, dtype=self._dtype(dtype))

    def zeros(self, *, shape: Tuple[int, ...], dtype: str) -> Any:
        return self.torch.zeros(shape, device=self.device, dtype=self._dtype(dtype))

    def full(
        self,
        *,
        shape: Tuple[int, ...],
        dtype: str,
        fill_value: int | float,
    ) -> Any:
        return self.torch.full(
            shape, fill_value, device=self.device, dtype=self._dtype(dtype)
        )

    def randint(
        self,
        *,
        shape: Tuple[int, ...],
        dtype: str,
        low: int,
        high: int,
    ) -> Any:
        return self.torch.randint(
            low, high, shape, device=self.device, dtype=self._dtype(dtype)
        )

    def literal(
        self, *, shape: Tuple[int, ...], dtype: str, values: Tuple[Any, ...]
    ) -> Any:
        return self.torch.tensor(
            values, device=self.device, dtype=self._dtype(dtype)
        ).reshape(shape)

    def clone(self, value: Any) -> Any:
        return value.clone()

    def as_strided(
        self,
        value: Any,
        *,
        shape: Tuple[int, ...],
        strides: Tuple[int, ...],
    ) -> Any:
        return self.torch.as_strided(value, size=shape, stride=strides)

    def synchronize(self) -> None:
        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)

    def _dtype(self, dtype: str) -> Any:
        if not isinstance(dtype, str) or not dtype.strip():
            raise ValueError("dtype must be a non-empty string")
        name = dtype.rsplit(".", 1)[-1]
        value = getattr(self.torch, name, None)
        if value is None or not isinstance(value, self.torch.dtype):
            raise ValueError(f"unsupported torch dtype: {dtype}")
        return value
