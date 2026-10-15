# References:
# https://github.com/facebookresearch/fairseq/blob/main/fairseq/modules/rotary_positional_embedding.py

import numpy as np

import torch
import torch.nn as nn

from einops import rearrange, repeat

from ...context_parallel import context_parallel_util


def broadcat(tensors, dim=-1):
    num_tensors = len(tensors)
    shape_lens = set(list(map(lambda t: len(t.shape), tensors)))
    assert len(shape_lens) == 1, "tensors must all have the same number of dimensions"
    shape_len = list(shape_lens)[0]
    dim = (dim + shape_len) if dim < 0 else dim
    dims = list(zip(*map(lambda t: list(t.shape), tensors)))
    expandable_dims = [(i, val) for i, val in enumerate(dims) if i != dim]
    assert all(
        [*map(lambda t: len(set(t[1])) <= 2, expandable_dims)]
    ), "invalid dimensions for broadcastable concatentation"
    max_dims = list(map(lambda t: (t[0], max(t[1])), expandable_dims))
    expanded_dims = list(map(lambda t: (t[0], (t[1],) * num_tensors), max_dims))
    expanded_dims.insert(dim, (dim, dims[dim]))
    expandable_shapes = list(zip(*map(lambda t: t[1], expanded_dims)))
    tensors = list(map(lambda t: t[0].expand(*t[1]), zip(tensors, expandable_shapes)))
    return torch.cat(tensors, dim=dim)


def rotate_half(x):
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, "... d r -> ... (d r)")


class RotaryPositionalEmbedding(nn.Module):

    _gpu_cos_sin_cache = {}

    def __init__(self,
                 head_dim,
                 cp_split_hw=None
                 ):
        """Rotary positional embedding for 3D
        Reference : https://blog.eleuther.ai/rotary-embeddings/
        Paper: https://arxiv.org/pdf/2104.09864.pdf
        Args:
            dim: Dimension of embedding
            base: Base value for exponential
        """
        super().__init__()
        self.head_dim = head_dim
        assert self.head_dim % 8 == 0, 'Dim must be a multiply of 8 for 3D RoPE.'
        self.cp_split_hw = cp_split_hw
        # We take the assumption that the longest side of grid will not larger than 512, i.e, 512 * 8 = 4098 input pixels
        self.base = 10000
        self.freqs_dict = {}

    def clear_gpu_cache(self):
        self._gpu_cos_sin_cache.clear()

    def register_grid_size(self, grid_size, key_name, frame_index=None, num_ref_latents=None):
        
        if key_name not in self.freqs_dict:
            self.freqs_dict.update({
                key_name: self.precompute_freqs_cis_3d(grid_size, frame_index, num_ref_latents)
            })

    def precompute_freqs_cis_3d(self, grid_size, frame_index=None, num_ref_latents=None):
        num_frames, height, width = grid_size     
        dim_t = self.head_dim - 4 * (self.head_dim // 6)
        dim_h = 2 * (self.head_dim // 6)
        dim_w = 2 * (self.head_dim // 6)
        freqs_t = 1.0 / (self.base ** (torch.arange(0, dim_t, 2)[: (dim_t // 2)].float() / dim_t))
        freqs_h = 1.0 / (self.base ** (torch.arange(0, dim_h, 2)[: (dim_h // 2)].float() / dim_h))
        freqs_w = 1.0 / (self.base ** (torch.arange(0, dim_w, 2)[: (dim_w // 2)].float() / dim_w))
        if frame_index is not None and num_ref_latents is not None:
            grid_t = torch.concat([torch.tensor([frame_index], dtype=torch.float32), torch.arange(0, num_frames-num_ref_latents, dtype=torch.float32)], dim=0)
        else:
            grid_t = np.linspace(0, num_frames, num_frames, endpoint=False, dtype=np.float32)
            grid_t = torch.from_numpy(grid_t).float()
        grid_h = np.linspace(0, height, height, endpoint=False, dtype=np.float32)
        grid_w = np.linspace(0, width, width, endpoint=False, dtype=np.float32)
        grid_h = torch.from_numpy(grid_h).float()
        grid_w = torch.from_numpy(grid_w).float()
        freqs_t = torch.einsum("..., f -> ... f", grid_t, freqs_t)
        freqs_h = torch.einsum("..., f -> ... f", grid_h, freqs_h)
        freqs_w = torch.einsum("..., f -> ... f", grid_w, freqs_w)
        freqs_t = repeat(freqs_t, "... n -> ... (n r)", r=2)
        freqs_h = repeat(freqs_h, "... n -> ... (n r)", r=2)
        freqs_w = repeat(freqs_w, "... n -> ... (n r)", r=2)
        freqs = broadcat((freqs_t[:, None, None, :], freqs_h[None, :, None, :], freqs_w[None, None, :, :]), dim=-1)
        # (T H W D)
        freqs = rearrange(freqs, "T H W D -> (T H W) D")
        if self.cp_split_hw[0] * self.cp_split_hw[1] > 1:
            with torch.no_grad():
                freqs = rearrange(freqs, "(T H W) D -> T H W D", T=num_frames, H=height, W=width)
                freqs = context_parallel_util.split_cp_2d(freqs, seq_dim_hw=(1, 2), split_hw=self.cp_split_hw)
                freqs = rearrange(freqs, "T H W D -> (T H W) D")

        return freqs

    def forward(self, q, k, grid_size, frame_index=None, num_ref_latents=None, token_offset=0):
        """3D RoPE.

        Args:
            query: [B, head, seq, head_dim]
            key: [B, head, seq, head_dim]
        Returns:
            query and key with the same shape as input.
        """
        key_name = '.'.join([str(i) for i in grid_size]) + f"-{str(frame_index)}-{str(num_ref_latents)}"
        if key_name not in self.freqs_dict:
            self.register_grid_size(grid_size, key_name, frame_index, num_ref_latents)

        q_, k_ = q.float(), k.float()
        cache_key = (
            self.head_dim,
            tuple(self.cp_split_hw or ()),
            key_name,
            q.device.type,
            q.device.index,
        )
        cached = self._gpu_cos_sin_cache.get(cache_key)
        if cached is None:
            # Keep the original FP32 path exactly; only reuse its immutable result.
            freqs_cis = self.freqs_dict[key_name].to(q.device)
            freqs_cis = freqs_cis.float().to(q.device)
            cos, sin = freqs_cis.cos(), freqs_cis.sin()
            cos = rearrange(cos, 'n d -> 1 1 n d')
            sin = rearrange(sin, 'n d -> 1 1 n d')
            cached = (cos, sin)
            self._gpu_cos_sin_cache[cache_key] = cached
        cos, sin = cached
        token_count = q.shape[-2]
        cos = cos[:, :, token_offset:token_offset + token_count]
        sin = sin[:, :, token_offset:token_offset + token_count]
        assert cos.shape[-2] == token_count
        q_ = (q_ * cos) + (rotate_half(q_) * sin)
        k_ = (k_ * cos) + (rotate_half(k_) * sin)

        return q_.type_as(q), k_.type_as(k)


class RotaryPositionalEmbedding1D(nn.Module):

    def __init__(self,
                 head_dim
                 ):
        """Rotary positional embedding for 1D
        Args:
            dim: Dimension of embedding
            base: Base value for exponential
        """
        super().__init__()
        self.head_dim = head_dim
        self.base = 10000

    def precompute_freqs_cis_1d(self, pos_indices):

        freqs = 1.0 / (self.base ** (torch.arange(0, self.head_dim, 2)[: (self.head_dim // 2)].float() / self.head_dim))

        freqs = freqs.to(pos_indices.device)
        freqs = torch.einsum("..., f -> ... f", pos_indices.float(), freqs)
        freqs = repeat(freqs, "... n -> ... (n r)", r=2)

        return freqs

    def forward(self, x, pos_indices):
        """1D RoPE.

        Args:
            query (torch.tensor): [B, head, seq, head_dim]
            pos_indices (torch.tensor): [seq,]
        Returns:
            query with the same shape as input.
        """
        freqs_cis = self.precompute_freqs_cis_1d(pos_indices)

        x_ = x.float()

        freqs_cis = freqs_cis.float().to(x.device)
        cos, sin = freqs_cis.cos(), freqs_cis.sin()
        cos, sin = rearrange(cos, 'n d -> 1 1 n d'), rearrange(sin, 'n d -> 1 1 n d')
        x_ = (x_ * cos) + (rotate_half(x_) * sin)

        return x_.type_as(x)
    
