"""Small distributed helpers used by low-VRAM staged inference."""

from collections.abc import Mapping

import torch
import torch.distributed as dist


_DTYPES = (
    torch.bool,
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
)


def _collective_device(group=None) -> torch.device:
    backend = dist.get_backend(group)
    if backend == "nccl":
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def broadcast_variable_tensor(tensor, src, group=None):
    """Broadcast an optional, variable-shaped tensor without large object payloads."""
    rank = dist.get_rank()
    device = _collective_device(group)
    if rank == src:
        present = tensor is not None
        ndim = tensor.ndim if present else 0
        dtype_index = _DTYPES.index(tensor.dtype) if present else 0
    else:
        present = False
        ndim = dtype_index = 0

    header = torch.tensor([int(present), ndim, dtype_index], dtype=torch.int64, device=device)
    dist.broadcast(header, src=src, group=group)
    if not bool(header[0].item()):
        return None

    ndim = int(header[1].item())
    dtype = _DTYPES[int(header[2].item())]
    if rank == src:
        shape = torch.tensor(tensor.shape, dtype=torch.int64, device=device)
    else:
        shape = torch.empty(ndim, dtype=torch.int64, device=device)
    dist.broadcast(shape, src=src, group=group)

    if rank == src:
        payload = tensor.to(device=device).contiguous()
    else:
        payload = torch.empty(tuple(shape.tolist()), dtype=dtype, device=device)
    dist.broadcast(payload, src=src, group=group)
    return payload


def broadcast_tensor_mapping(tensors, keys, src, group=None):
    """Broadcast a fixed-schema mapping of optional tensors in deterministic order."""
    tensors = tensors or {}
    if not isinstance(tensors, Mapping):
        raise TypeError("tensors must be a mapping or None")
    return {
        key: broadcast_variable_tensor(tensors.get(key), src=src, group=group)
        for key in keys
    }
