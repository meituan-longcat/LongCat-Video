import torch
import torch.distributed as dist

from ..context_parallel import context_parallel_util


def all_to_all(tensor, scatter_idx, gather_idx, group=None, gather=True):
    """Perform all-to-all communication on a tensor.

    Args:
        tensor (torch.Tensor): Input tensor for all-to-all communication
        scatter_idx (int): Dimension to scatter, will split along this dimension and then scatter to all processes
        gather_idx (int): Dimension to gather, will gather from all processes and then concatenate along this dimension
        group (ProcessGroup, optional): Process group to use for communication

    Returns:
        torch.Tensor
    """
    if not dist.is_initialized():
        return tensor

    world_size = dist.get_world_size(group)
    ulysses_rank = context_parallel_util.get_cp_rank()
    if world_size == 1:
        return tensor

    if scatter_idx == gather_idx:
        raise ValueError("scatter_idx and gather_idx must be different")

    def chunk_tensor(tensor, scatter_idx):
        t_shape = list(tensor.shape)
        if t_shape[scatter_idx] % world_size != 0:
            raise ValueError(f"Dimension {scatter_idx} must be divisible by world size {world_size}")
        chunk_size = t_shape[scatter_idx] // world_size
        new_shape = list()
        for i in range(len(t_shape)):
            if i != scatter_idx:
                new_shape.append(t_shape[i])
            else:
                new_shape.extend([world_size, chunk_size])
        tensor = tensor.reshape(*new_shape)
        # move scatter_idx to front
        tensor = tensor.permute(scatter_idx, *[i for i in range(len(new_shape)) if i != scatter_idx]).contiguous()
        return tensor

    # chunk tensor for all_to_all
    tensor = chunk_tensor(tensor, scatter_idx)

    # Perform all2all
    output = torch.empty_like(tensor)
    dist.all_to_all_single(output, tensor, group=group)

    # output: e.g., [world_size, B, chunked_H, chunked_S, D] if scatter_idx == 1, gather_idx == 2 -> [B, chunked_H, S, D]
    def reorder_tensor(tensor, gather_idx):
        t_shape = list(tensor.shape)
        world_size = t_shape[0]
        # insert front to gather_idx + 1
        permute_idx = list()
        for i in range(1, len(t_shape)):
            if i != gather_idx + 1:
                permute_idx.append(i)
            else:
                permute_idx.extend([0, i])
        tensor = tensor.permute(*permute_idx).contiguous() # permute(1,2,0,3) W B CH CS D -> B CH W CS D

        # reshape tensor
        new_shape = list()
        if gather:
            for i in range(1, len(t_shape)): # B CH CS D
                if i != gather_idx + 1:
                    new_shape.append(t_shape[i])
                else:
                    new_shape.append(world_size * t_shape[i]) # B CH W*CS D

            tensor = tensor.reshape(*new_shape)
        else:
            tensor = tensor[:,ulysses_rank] # W B CS CH D -> B CS W CH D

        return tensor

    output = reorder_tensor(output, gather_idx)

    return output


@torch.compiler.disable
def ulysses_a2a_in(query, key, value):
    if context_parallel_util.get_cp_size() == 1:
        return query, key, value

    # [B, H, S/N, D] -> [B, H/N, S, D]
    query = all_to_all(query, scatter_idx=1, gather_idx=2, group=context_parallel_util.get_cp_group())
    key = all_to_all(key, scatter_idx=1, gather_idx=2, group=context_parallel_util.get_cp_group())
    value = all_to_all(value, scatter_idx=1, gather_idx=2, group=context_parallel_util.get_cp_group())
    return query, key, value


@torch.compiler.disable
def ulysses_a2a_out(output):
    if context_parallel_util.get_cp_size() == 1:
        return output

    # [B, H/N, S, D] -> [B, H, S/N, D]
    output = all_to_all(output, scatter_idx=2, gather_idx=1, group=context_parallel_util.get_cp_group())
    return output


def ulysses_wrapper(func):
    def wrapper(self, query, key, value, shape):
        # Apply ulysses_a2a_in before the function call, gather sequence and split head
        query, key, value = ulysses_a2a_in(query, key, value)
        output = func(self, query, key, value, shape)        
        output = ulysses_a2a_out(output)
        return output

    return wrapper
