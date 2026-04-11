from __future__ import annotations

from collections.abc import Sequence

import torch

TensorLike = torch.Tensor | Sequence[float] | None


def ensure_tensor(value: TensorLike, dim: int, device: torch.device | str) -> torch.Tensor:
    device = torch.device(device)
    if value is None:
        return torch.zeros(dim, dtype=torch.float32, device=device)
    if isinstance(value, torch.Tensor):
        tensor = value.detach().to(device=device, dtype=torch.float32).flatten()
    else:
        tensor = torch.tensor(list(value), dtype=torch.float32, device=device).flatten()
    if tensor.numel() == dim:
        return tensor
    if tensor.numel() > dim:
        return tensor[:dim]
    padded = torch.zeros(dim, dtype=torch.float32, device=device)
    if tensor.numel() > 0:
        padded[: tensor.numel()] = tensor
    return padded


def clone_tensor_dict(values: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.detach().clone() for key, value in values.items()}
