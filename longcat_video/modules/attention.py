from typing import List, Optional

import torch
import torch.nn as nn
import xformers.ops

from einops import rearrange

from .rope_3d import RotaryPositionalEmbedding
from .blocks import RMSNorm_FP32
from ..block_sparse_attention.bsa_interface import flash_attn_bsa_3d
from ..context_parallel.ulysses_wrapper import ulysses_wrapper


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        enable_flashattn3: bool = False,
        enable_flashattn2: bool = False,
        enable_xformers: bool = False,
        enable_bsa: bool = False,
        bsa_params: dict = None,
        cp_split_hw: Optional[List[int]] = None
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.enable_flashattn3 = enable_flashattn3
        self.enable_flashattn2 = enable_flashattn2
        self.enable_xformers = enable_xformers
        self.enable_bsa = enable_bsa
        self.bsa_params = bsa_params
        self.cp_split_hw = cp_split_hw

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.q_norm = RMSNorm_FP32(self.head_dim, eps=1e-6)
        self.k_norm = RMSNorm_FP32(self.head_dim, eps=1e-6)
        self.proj = nn.Linear(dim, dim)

        self.rope_3d = RotaryPositionalEmbedding(
            self.head_dim,
            cp_split_hw=cp_split_hw
        )

    @ulysses_wrapper
    def _process_attn(self, q, k, v, shape):
        """
            function wrapper to do attention with q, k, v
        """

        B, H, SQ, D = q.shape
        _, _, SKV, _ = k.shape

        if self.enable_bsa and shape[0] > 1: # bsa will not be used in image training / sampling
            assert self.bsa_params is not None
            _, H, W = shape
            assert H % self.cp_split_hw[0] == 0, W % self.cp_split_hw[1] == 0
            H, W = H // self.cp_split_hw[0], W // self.cp_split_hw[1]
            Tq = SQ // (H * W)
            Tk = SKV // (H * W)
            latent_shape_q = (Tq, H, W)
            latent_shape_k = (Tk, H, W)
            x = flash_attn_bsa_3d(q, k, v, latent_shape_q, latent_shape_k, **self.bsa_params)
        elif self.enable_flashattn3:
            from flash_attn_interface import flash_attn_func
            q = rearrange(q, "B H S D -> B S H D").contiguous()
            k = rearrange(k, "B H S D -> B S H D").contiguous()
            v = rearrange(v, "B H S D -> B S H D").contiguous()
            x, *_ = flash_attn_func(
                q,
                k,
                v,
                softmax_scale=self.scale,
            )
            x = rearrange(x, "B S H D -> B H S D")
        elif self.enable_flashattn2:
            from flash_attn import flash_attn_func
            q = rearrange(q, "B H S D -> B S H D")
            k = rearrange(k, "B H S D -> B S H D")
            v = rearrange(v, "B H S D -> B S H D")
            x = flash_attn_func(
                q,
                k,
                v,
                dropout_p=0.0,
                softmax_scale=self.scale,
            )
            x = rearrange(x, "B S H D -> B H S D")
        elif self.enable_xformers:
            # Input tensors must be in format ``[B, M, H, K]``, where B is the batch size, M \
            # the sequence length, H the number of heads, and K the embeding size per head
            q = rearrange(q, "B H M K -> B M H K")
            k = rearrange(k, "B H M K -> B M H K")
            v = rearrange(v, "B H M K -> B M H K")
            x = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None, op=None,)
            x = rearrange(x, "B M H K -> B H M K")
        else:
            raise RuntimeError("Unsupported attention operations.")

        return x

    def forward(self, x: torch.Tensor, shape=None, num_cond_latents=None, return_kv=False) -> torch.Tensor:
        """
        """
        B, N, C = x.shape
        qkv = self.qkv(x)

        qkv_shape = (B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.view(qkv_shape).permute((2, 0, 3, 1, 4)) # [3, B, H, N, D]
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if return_kv:
            k_cache, v_cache = k.clone(), v.clone()

        q, k = self.rope_3d(q, k, shape)

        # cond mode
        if num_cond_latents is not None and num_cond_latents > 0:
            num_cond_latents_thw = num_cond_latents * (N // shape[0])
            # process the condition tokens
            q_cond = q[:, :, :num_cond_latents_thw].contiguous()
            k_cond = k[:, :, :num_cond_latents_thw].contiguous()
            v_cond = v[:, :, :num_cond_latents_thw].contiguous()
            x_cond = self._process_attn(q_cond, k_cond, v_cond, shape)
            # process the noise tokens
            q_noise = q[:, :, num_cond_latents_thw:].contiguous()
            x_noise = self._process_attn(q_noise, k, v, shape)
            # merge x_cond and x_noise
            x = torch.cat([x_cond, x_noise], dim=2).contiguous()
        else:
            x = self._process_attn(q, k, v, shape)

        x_output_shape = (B, N, C)
        x = x.transpose(1, 2) # [B, H, N, D] --> [B, N, H, D]
        x = x.reshape(x_output_shape) # [B, N, H, D] --> [B, N, C]
        x = self.proj(x)

        if return_kv:
            return x, (k_cache, v_cache)
        else:
            return x

    def forward_with_kv_cache(self, x: torch.Tensor, shape=None, num_cond_latents=None, kv_cache=None) -> torch.Tensor:
        """
        """
        B, N, C = x.shape
        qkv = self.qkv(x)
        
        qkv_shape = (B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.view(qkv_shape).permute((2, 0, 3, 1, 4)) # [3, B, H, N, D]
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        T, H, W = shape
        k_cache, v_cache = kv_cache
        assert k_cache.shape[0] == v_cache.shape[0] and k_cache.shape[0] in [1, B]
        if k_cache.shape[0] == 1:
            k_cache = k_cache.repeat(B, 1, 1, 1)
            v_cache = v_cache.repeat(B, 1, 1, 1)
        
        if num_cond_latents is not None and num_cond_latents > 0:
            k_full = torch.cat([k_cache, k], dim=2).contiguous()
            v_full = torch.cat([v_cache, v], dim=2).contiguous()
            q_padding = torch.cat([torch.empty_like(k_cache), q], dim=2).contiguous()
            q_padding, k_full = self.rope_3d(q_padding, k_full, (T + num_cond_latents, H, W))
            q = q_padding[:, :, -N:].contiguous()
            
        x = self._process_attn(q, k_full, v_full, shape)
        
        x_output_shape = (B, N, C)
        x = x.transpose(1, 2) # [B, H, N, D] --> [B, N, H, D]
        x = x.reshape(x_output_shape) # [B, N, H, D] --> [B, N, C]
        x = self.proj(x)

        return x


class MultiHeadCrossAttention(nn.Module):
    def __init__(
            self,
            dim,
            num_heads,
            enable_flashattn3=False,
            enable_flashattn2=False,
            enable_xformers=False,
        ):
        super(MultiHeadCrossAttention, self).__init__()
        assert dim % num_heads == 0, "d_model must be divisible by num_heads"

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q_linear = nn.Linear(dim, dim)
        self.kv_linear = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)

        self.q_norm = RMSNorm_FP32(self.head_dim, eps=1e-6)
        self.k_norm = RMSNorm_FP32(self.head_dim, eps=1e-6)

        self.enable_flashattn3 = enable_flashattn3
        self.enable_flashattn2 = enable_flashattn2
        self.enable_xformers = enable_xformers

    def _process_cross_attn(self, x, cond, kv_seqlen):
        B, N, C = x.shape
        assert C == self.dim and cond.shape[2] == self.dim

        q = self.q_linear(x).view(1, -1, self.num_heads, self.head_dim)
        kv = self.kv_linear(cond).view(1, -1, 2, self.num_heads, self.head_dim)
        k, v = kv.unbind(2)

        q, k = self.q_norm(q), self.k_norm(k)

        if self.enable_flashattn3:
            from flash_attn_interface import flash_attn_varlen_func
            x = flash_attn_varlen_func(
                q=q[0],
                k=k[0],
                v=v[0],
                cu_seqlens_q=torch.tensor([0] + [N] * B, device=q.device).cumsum(0).to(torch.int32),
                cu_seqlens_k=torch.tensor([0] + kv_seqlen, device=q.device).cumsum(0).to(torch.int32),
                max_seqlen_q=N,
                max_seqlen_k=max(kv_seqlen),
            )[0]
        elif self.enable_flashattn2:
            from flash_attn import flash_attn_varlen_func
            x = flash_attn_varlen_func(
                q=q[0],
                k=k[0],
                v=v[0],
                cu_seqlens_q=torch.tensor([0] + [N] * B, device=q.device).cumsum(0).to(torch.int32),
                cu_seqlens_k=torch.tensor([0] + kv_seqlen, device=q.device).cumsum(0).to(torch.int32),
                max_seqlen_q=N,
                max_seqlen_k=max(kv_seqlen),
            )
        elif self.enable_xformers:
            attn_bias = xformers.ops.fmha.attn_bias.BlockDiagonalMask.from_seqlens([N] * B, kv_seqlen)
            x = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        else:
            raise RuntimeError("Unsupported attention operations.")


        x = x.view(B, -1, C)
        x = self.proj(x)
        return x

    def forward(self, x, cond, kv_seqlen, num_cond_latents=None, shape=None):
        """
            x: [B, N, C]
            cond: [B, M, C]
        """
        if num_cond_latents is None or num_cond_latents == 0:
            return self._process_cross_attn(x, cond, kv_seqlen)
        else:
            B, N, C = x.shape
            if num_cond_latents is not None and num_cond_latents > 0:
                assert shape is not None, "SHOULD pass in the shape"
                num_cond_latents_thw = num_cond_latents * (N // shape[0])
                x_noise = x[:, num_cond_latents_thw:] # [B, N_noise, C]
                output_noise = self._process_cross_attn(x_noise, cond, kv_seqlen) # [B, N_noise, C]
                output = torch.cat([
                    torch.zeros((B, num_cond_latents_thw, C), dtype=output_noise.dtype, device=output_noise.device),
                    output_noise
                ], dim=1).contiguous()
            else:
                raise NotImplementedError
                
            return output