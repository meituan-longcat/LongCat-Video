import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

from einops import rearrange


dp_size = dp_group = cp_group = cp_stream = dp_ranks = cp_ranks = dp_rank = None
cp_size: int = 1
cp_rank: int = 0


def init_context_parallel(context_parallel_size: int = 1,
                          global_rank: int = 0,
                          world_size: int = 1,):

    global dp_size, cp_size, dp_group, cp_group, dp_ranks, cp_ranks, dp_rank, cp_rank

    if world_size % context_parallel_size != 0:
        raise RuntimeError(f'world_size {world_size} must be multiple of context_parallel_size {context_parallel_size}')

    cp_size = context_parallel_size
    dp_size = world_size//context_parallel_size
    print(f'[rank {global_rank}] init_device_mesh [dp_size x cp_size]: [{dp_size} x {cp_size}]')

    mesh_2d = init_device_mesh("cuda", (dp_size, cp_size), mesh_dim_names=("dp", "cp"))
    print(f'[rank {global_rank}] mesh_2d: {mesh_2d}')

    dp_group = mesh_2d.get_group(mesh_dim="dp")
    cp_group = mesh_2d.get_group(mesh_dim="cp")
    dp_ranks = torch.distributed.get_process_group_ranks(dp_group)
    cp_ranks = torch.distributed.get_process_group_ranks(cp_group)
    dp_rank = dist.get_rank(group=dp_group)
    cp_rank = dist.get_rank(group=cp_group)

    curr_global_rank = torch.distributed.get_rank()
    print(f'[rank {curr_global_rank}] [dp_rank, cp_rank]: [{dp_rank}, {cp_rank}],  dp_ranks: {dp_ranks}, cp_ranks: {cp_ranks}')


def get_cp_size():
    global cp_size
    return cp_size


def get_dp_size():
    global dp_size
    return dp_size


def get_cp_stream():
    global cp_stream
    if cp_stream == None:
        cp_stream = torch.cuda.Stream()
    return cp_stream


def get_dp_group():
    global dp_group
    return dp_group


def get_cp_group():
    global cp_group
    return cp_group


def get_dp_rank():
    global dp_rank
    return dp_rank


def get_cp_rank():
    global cp_rank
    return cp_rank


def get_cp_rank_list():
    global cp_ranks
    if cp_ranks == None:
        cp_ranks = torch.distributed.get_process_group_ranks(cp_group)
    return cp_ranks


def cp_broadcast(tensor, cp_index=0):
    global dp_group
    global cp_group
    cp_ranks = get_cp_rank_list()
    torch.distributed.broadcast(tensor, cp_ranks[cp_index], group=cp_group)


def split_tensor_in_cp_2d(input, dim_hw, split_hw):

    global cp_size
    
    dim_h, dim_w = dim_hw
    split_h, split_w = split_hw
    
    assert cp_size == split_h * split_w

    seq_size_h = input.shape[dim_h]
    seq_size_w = input.shape[dim_w]

    if seq_size_h % split_h != 0:
        raise RuntimeError(f'seq_size_h {seq_size_h} in dim_h {dim_h} must be multiple of split_h {split_h}!!!')
    if seq_size_w % split_w != 0:
        raise RuntimeError(f'seq_size_w {seq_size_w} in dim_w {dim_w} must be multiple of split_w {split_w}!!!')

    split_seq_size_h = seq_size_h // split_h
    split_seq_size_w = seq_size_w // split_w

    tensor_splits_h = input.split(split_seq_size_h, dim=dim_h)
    tensor_splits = []
    for tensor_split_h in tensor_splits_h:
        tensor_splits_hw = tensor_split_h.split(split_seq_size_w, dim=dim_w)
        tensor_splits.extend(tensor_splits_hw)

    cp_rank = get_cp_rank()

    split_tensor = tensor_splits[cp_rank]

    return split_tensor


class GatherFunction2D(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, process_group, seq_dim_hw, shape, split_hw):
        ctx.cp_group = process_group
        ctx.seq_dim_hw = seq_dim_hw
        ctx.split_hw = split_hw
        ctx.shape = shape
        ctx.cp_size = get_cp_size()

        T, H, W = shape
        dim_h, dim_w = seq_dim_hw
        split_h, split_w = split_hw
        assert H % split_h == 0, W % split_w == 0
        assert T * (H // split_h) * (W // split_w) == input.shape[1]
        input = rearrange(input, "B (T H W) C -> B T H W C", T=T, H=H // split_h, W=W // split_w)

        with torch.no_grad():
            input = input.contiguous()
            output_tensors = [torch.zeros_like(input) for _ in range(ctx.cp_size)]
            dist.all_gather(output_tensors, input, group=ctx.cp_group)
            output_tensors_hs = []
            assert ctx.cp_size % split_w == 0
            for i in range(0, ctx.cp_size // split_w):
                output_tensors_hs.append(
                    torch.cat(output_tensors[i * split_w : (i + 1) * split_w], dim=dim_w)
                )
            output_tensor = torch.cat(output_tensors_hs, dim=dim_h)

        output_tensor = rearrange(output_tensor, "B T H W C -> B (T H W) C")

        return output_tensor

    @staticmethod
    def backward(ctx, grad_output):
        T, H, W = ctx.shape
        with torch.no_grad():
            grad_output = grad_output * ctx.cp_size
            grad_output = rearrange(grad_output, "B (T H W) C -> B T H W C", T=T, H=H, W=W)
            grad_input = split_tensor_in_cp_2d(grad_output, ctx.seq_dim_hw, ctx.split_hw)
            grad_input = rearrange(grad_input, "B T H W C -> B (T H W) C")
            
        return grad_input, None, None, None, None


class SplitFunction2D(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, process_group, seq_dim_hw, split_hw):
        ctx.cp_group = process_group
        ctx.seq_dim_hw = seq_dim_hw
        ctx.split_hw = split_hw
        ctx.cp_size = get_cp_size()
        output_tensor = split_tensor_in_cp_2d(input, ctx.seq_dim_hw, split_hw)

        return output_tensor

    @staticmethod
    def backward(ctx, grad_output):
        with torch.no_grad():
            grad_output = grad_output / ctx.cp_size
            output_tensors = [torch.zeros_like(grad_output) for _ in range(ctx.cp_size)]
            dist.all_gather(output_tensors, grad_output, group=ctx.cp_group)
            split_h, split_w = ctx.split_hw
            dim_h, dim_w = ctx.seq_dim_hw
            output_tensors_hs = []
            assert ctx.cp_size % split_w == 0
            for i in range(0, ctx.cp_size // split_w):
                output_tensors_hs.append(
                    torch.cat(output_tensors[i * split_w : (i + 1) * split_w], dim=dim_w)
                )
            grad_input = torch.cat(output_tensors_hs, dim=dim_h)

        return grad_input, None, None, None


def gather_cp_2d(input, shape, split_hw):
    cp_process_group = get_cp_group()
    output_tensor = GatherFunction2D.apply(input, cp_process_group, (2, 3), shape, split_hw)

    return output_tensor


def split_cp_2d(input, seq_dim_hw, split_hw):
    cp_process_group = get_cp_group()
    output_tensor = SplitFunction2D.apply(input, cp_process_group, seq_dim_hw, split_hw)

    return output_tensor


def get_optimal_split(size):
    factors = []
    for i in range(1, int(size**0.5) + 1):
        if size % i == 0:
            factors.append([i, size // i])
    return min(factors, key=lambda x: abs(x[0] - x[1]))