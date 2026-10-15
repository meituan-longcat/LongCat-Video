from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.amp as amp

import numpy as np 
from einops import rearrange

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin

from safetensors.torch import load_file

from ..lora_utils import create_lora_network
from ...context_parallel import context_parallel_util
from ..attention import MultiHeadCrossAttention
from ..blocks import TimestepEmbedder, CaptionEmbedder, PatchEmbed3D, FeedForwardSwiGLU, FinalLayer_FP32, LayerNorm_FP32, modulate_fp32

from .attention import Attention, SingleStreamAttention
from .blocks import AudioProjModel


class LongCatAvatarSingleStreamBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: int,
        adaln_tembed_dim: int,
        enable_flashattn3: bool = False,
        enable_flashattn2: bool = False,
        enable_xformers: bool = False,
        enable_bsa: bool = False,
        bsa_params=None,
        cp_split_hw=None,
        # avatar config
        output_dim=768,
        audio_prenorm=True,
        class_range=24,
        class_interval=4
    ):
        super().__init__()

        self.hidden_size = hidden_size

        # scale and gate modulation
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(adaln_tembed_dim, 6 * hidden_size, bias=True)
        )
        self.audio_adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(adaln_tembed_dim, 3 * hidden_size, bias=True)
        )

        self.mod_norm_attn = LayerNorm_FP32(hidden_size, eps=1e-6, elementwise_affine=False)
        self.mod_norm_ffn  = LayerNorm_FP32(hidden_size, eps=1e-6, elementwise_affine=False)
        self.pre_crs_attn_norm = LayerNorm_FP32(hidden_size, eps=1e-6, elementwise_affine=True)

        self.pre_video_crs_attn_norm = LayerNorm_FP32(hidden_size, eps=1e-6, elementwise_affine=True)
        self.pre_audio_crs_attn_norm = LayerNorm_FP32(output_dim, eps=1e-6, elementwise_affine=True) if audio_prenorm else nn.Identity()
        
        self.attn = Attention(
            dim=hidden_size,
            num_heads=num_heads,
            enable_flashattn3=enable_flashattn3,
            enable_flashattn2=enable_flashattn2,
            enable_xformers=enable_xformers,
            enable_bsa=enable_bsa,
            bsa_params=bsa_params,
            cp_split_hw=cp_split_hw
        )
        self.cross_attn = MultiHeadCrossAttention(
            dim=hidden_size,
            num_heads=num_heads,
            enable_flashattn3=enable_flashattn3,
            enable_flashattn2=enable_flashattn2,
            enable_xformers=enable_xformers
        )

        self.audio_cross_attn = SingleStreamAttention(
                dim=hidden_size,
                encoder_hidden_states_dim=output_dim,
                num_heads=num_heads,
                qk_norm=True,
                qkv_bias=True,
                class_range=class_range,
                class_interval=class_interval,
                cp_split_hw=cp_split_hw,
                enable_flashattn3=enable_flashattn3,
                enable_flashattn2=enable_flashattn2,
                enable_xformers=enable_xformers
            )

        self.ffn = FeedForwardSwiGLU(dim=hidden_size, hidden_dim=int(hidden_size * mlp_ratio))

    def forward(
        self, 
        x, 
        y, 
        t, 
        y_seqlen, 
        latent_shape, 
        num_cond_latents=None, 
        return_kv=False, 
        kv_cache=None, 
        skip_crs_attn=False,
        # avatar related params
        num_ref_latents=None,
        audio_hidden_states=None, 
        ref_img_index=None,
        mask_frame_range=None,
        token_ref_target_masks=None,
        human_num=None,
    ):
        """
            x: [B, N, C]
            y: [1, N_valid_tokens, C]
            t: [B, T, C_t]
            y_seqlen: [B]; type of a list
            latent_shape: latent shape of a single item
        """
        x_dtype = x.dtype

        B, N, C = x.shape
        T, _, _ = latent_shape # S != T*H*W in case of CP split on H*W.

        # compute modulation params in fp32
        with amp.autocast(device_type='cuda', dtype=torch.float32):
            shift_msa, scale_msa, gate_msa, \
            shift_mlp, scale_mlp, gate_mlp = \
                self.adaLN_modulation(t).unsqueeze(2).chunk(6, dim=-1) # [B, T, 1, C]

        # self attn with modulation
        x_m = modulate_fp32(self.mod_norm_attn, x.view(B, T, -1, C), shift_msa, scale_msa).view(B, N, C)

        if kv_cache is not None:
            kv_cache = (kv_cache[0].to(x.device), kv_cache[1].to(x.device))
            attn_outputs = self.attn.forward_with_kv_cache(x_m, shape=latent_shape, num_cond_latents=num_cond_latents, kv_cache=kv_cache, num_ref_latents=num_ref_latents, \
                                                            ref_img_index=ref_img_index, mask_frame_range=mask_frame_range, ref_target_masks=token_ref_target_masks)
        else:
            attn_outputs = self.attn(x_m, shape=latent_shape, num_cond_latents=num_cond_latents, return_kv=return_kv, num_ref_latents=num_ref_latents, \
                                                            ref_img_index=ref_img_index, mask_frame_range=mask_frame_range, ref_target_masks=token_ref_target_masks)
        
        if return_kv:
            x_s, kv_cache, x_ref_attn_map = attn_outputs
        else:
            x_s, x_ref_attn_map = attn_outputs

        with amp.autocast(device_type='cuda', dtype=torch.float32):
            x = x + (gate_msa * x_s.view(B, -1, N//T, C)).view(B, -1, C) # [B, N, C]
        x = x.to(x_dtype)

        # text cross attn
        if not skip_crs_attn:
            if kv_cache is not None:
                num_cond_latents = None
            x = x + self.cross_attn(self.pre_crs_attn_norm(x), y, y_seqlen, num_cond_latents=num_cond_latents, shape=latent_shape)
        
        # audio cross attn
        if not skip_crs_attn:
            if kv_cache is not None:
                num_cond_latents = 0

            with amp.autocast(device_type='cuda', dtype=torch.float32):  
                audio_shift_mca, audio_scale_mca, audio_gate_mca = \
                        self.audio_adaLN_modulation(t[:, num_cond_latents:]).unsqueeze(2).chunk(3, dim=-1) # [B, T, 1, C]

            audio_output_cond, audio_output_noise = self.audio_cross_attn(self.pre_video_crs_attn_norm(x), self.pre_audio_crs_attn_norm(audio_hidden_states), \
                                                                            shape=latent_shape, num_cond_latents=num_cond_latents, x_ref_attn_map=x_ref_attn_map, human_num=human_num)

            with amp.autocast(device_type='cuda', dtype=torch.float32):  
                audio_output_noise = modulate_fp32(self.mod_norm_attn, audio_output_noise.view(B, T-num_cond_latents, -1, C), audio_shift_mca, audio_scale_mca).view(B, -1, C)
                audio_add_x = (audio_gate_mca * audio_output_noise.view(B, T-num_cond_latents, -1, C)).view(B, -1, C) # [B, N, C]
                if audio_output_cond is not None:
                    audio_add_x = torch.cat([audio_output_cond, audio_add_x], dim=1).contiguous()
            x = x + audio_add_x
            x = x.to(x_dtype)

        # ffn with modulation
        x_m = modulate_fp32(self.mod_norm_ffn, x.view(B, -1, N//T, C), shift_mlp, scale_mlp).view(B, -1, C)
        x_s = self.ffn(x_m)
        with amp.autocast(device_type='cuda', dtype=torch.float32):
            x = x + (gate_mlp * x_s.view(B, -1, N//T, C)).view(B, -1, C) # [B, N, C]
        x = x.to(x_dtype)

        if return_kv:
            return x, kv_cache
        else:
            return x


class LongCatVideoAvatarTransformer3DModel(
    ModelMixin, ConfigMixin
):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        in_channels: int = 16,
        out_channels: int = 16,
        hidden_size: int = 4096,
        depth: int = 48,
        num_heads: int = 32,
        caption_channels: int = 4096,
        mlp_ratio: int = 4,
        adaln_tembed_dim: int = 512,
        frequency_embedding_size: int = 256,
        # default params
        patch_size: Tuple[int] = (1, 2, 2),
        # attention config
        enable_flashattn3: bool = False,
        enable_flashattn2: bool = False,
        enable_xformers: bool = False,
        enable_bsa: bool = False,
        bsa_params: dict = None,
        cp_split_hw: Optional[List[int]] = None,
        text_tokens_zero_pad: bool = False,
        # avatar config
        audio_window: int = 5,
        audio_block: int = 12,
        audio_channel: int = 768,
        intermediate_dim: int = 512,
        output_dim: int = 768,
        context_tokens: int = 32,
        vae_scale: int = 4, 
        audio_prenorm: bool = False,
        class_range: int = 24,
        class_interval: int = 4
    ) -> None:
        super().__init__()

        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.cp_split_hw = cp_split_hw
        self.vae_scale = vae_scale
        self.audio_window = audio_window

        self.x_embedder = PatchEmbed3D(patch_size, in_channels, hidden_size)
        self.t_embedder = TimestepEmbedder(t_embed_dim=adaln_tembed_dim, frequency_embedding_size=frequency_embedding_size)
        self.y_embedder = CaptionEmbedder(
            in_channels=caption_channels,
            hidden_size=hidden_size,
        )

        self.blocks = nn.ModuleList(
            [
                LongCatAvatarSingleStreamBlock(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    adaln_tembed_dim=adaln_tembed_dim,
                    enable_flashattn3=enable_flashattn3,
                    enable_flashattn2=enable_flashattn2,
                    enable_xformers=enable_xformers,
                    enable_bsa=enable_bsa,
                    bsa_params=bsa_params,
                    cp_split_hw=cp_split_hw,
                    output_dim=output_dim,
                    audio_prenorm=audio_prenorm,
                    class_range=class_range,
                    class_interval=class_interval
                )
                for i in range(depth)
            ]
        )

        self.audio_proj = AudioProjModel(
                    seq_len=audio_window,
                    seq_len_vf=audio_window+vae_scale-1,
                    blocks=audio_block,
                    channels=audio_channel,
                    intermediate_dim=intermediate_dim,
                    output_dim=output_dim,
                    context_tokens=context_tokens
                )

        self.final_layer = FinalLayer_FP32(
            hidden_size,
            np.prod(self.patch_size),
            out_channels,
            adaln_tembed_dim,
        )

        self.gradient_checkpointing = False
        self.text_tokens_zero_pad = text_tokens_zero_pad

        self.lora_dict = {}
        self.active_loras = []
        self.lora_mix_config = None
        self._lora_mix_active = None
    
    def load_lora(self, lora_path, lora_key, multiplier=1.0, lora_network_dim=128, lora_network_alpha=64):
        lora_network_state_dict_loaded = load_file(lora_path, device="cpu")
        lora_network = create_lora_network(
            transformer=self,
            lora_network_state_dict_loaded=lora_network_state_dict_loaded,
            multiplier=multiplier,
            network_dim=lora_network_dim,
            network_alpha=lora_network_alpha,
        )
        
        lora_network.load_state_dict(lora_network_state_dict_loaded, strict=True)
        
        self.lora_dict[lora_key] = lora_network

    def merge_lora(self, lora_path, lora_key="merged", multiplier=0.8, lora_network_dim=128, lora_network_alpha=64):
        if lora_key in self.lora_dict:
            raise KeyError(f"LoRA key already loaded: {lora_key}")
        self.load_lora(
            lora_path,
            lora_key,
            multiplier=multiplier,
            lora_network_dim=lora_network_dim,
            lora_network_alpha=lora_network_alpha,
        )
        lora_network = self.lora_dict.pop(lora_key)

        targets = []
        for lora in lora_network.loras:
            module_name = lora.lora_name.replace("lora___lorahyphen___", "").replace("___lorahyphen___", ".")
            module = self._get_module_by_name(module_name)
            if module.__class__.__name__ != "Linear":
                raise TypeError(f"Static LoRA target must be Linear, got {module.__class__.__name__}: {module_name}")
            targets.append((module, lora))
        if not targets:
            raise RuntimeError(f"No LoRA modules found in {lora_path}")

        with torch.no_grad():
            for module, lora in targets:
                origin_dtype = module.weight.dtype
                origin_device = module.weight.device
                module.to(device=origin_device, dtype=torch.float32)
                lora.to(device=origin_device, dtype=torch.float32)

                down = lora.lora_down.weight
                if hasattr(lora.lora_up, "blocks"):
                    down_chunks = torch.chunk(down, len(lora.lora_up.blocks), dim=0)
                    delta = torch.cat([
                        torch.mm(block.weight, down_chunks[index])
                        for index, block in enumerate(lora.lora_up.blocks)
                    ], dim=0)
                else:
                    delta = torch.mm(lora.lora_up.weight, down)

                scale = lora.multiplier * float(lora.alpha_scale.float().item())
                module.weight.data += scale * delta
                module.to(device=origin_device, dtype=origin_dtype)

        return len(targets)

    def enable_lora_mix(self, dmd_key, opsd_key, mix_alpha, mix_steps):
        if not 0.0 <= mix_alpha <= 1.0:
            raise ValueError(f"mix_alpha must be in [0, 1], got {mix_alpha}")
        missing_keys = [key for key in (dmd_key, opsd_key) if key not in self.lora_dict]
        if missing_keys:
            raise KeyError(f"LoRA keys not loaded: {missing_keys}")

        self.enable_loras([dmd_key, opsd_key])
        self.lora_mix_config = {
            "dmd_key": dmd_key,
            "opsd_key": opsd_key,
            "mix_alpha": mix_alpha,
            "mix_steps": mix_steps,
        }
        self._lora_mix_active = None
        self.set_lora_mix_step(0)

    def set_lora_mix_step(self, step_index):
        if self.lora_mix_config is None:
            return

        is_mixed = step_index < self.lora_mix_config["mix_steps"]
        if is_mixed == self._lora_mix_active:
            return

        dmd_network = self.lora_dict[self.lora_mix_config["dmd_key"]]
        opsd_network = self.lora_dict[self.lora_mix_config["opsd_key"]]
        alpha = self.lora_mix_config["mix_alpha"]
        dmd_multiplier = alpha if is_mixed else 0.0
        opsd_multiplier = 1.0 - alpha if is_mixed else 1.0

        dmd_network.set_multiplier(dmd_multiplier)
        dmd_network.set_use_lora(dmd_multiplier != 0.0)
        opsd_network.set_multiplier(opsd_multiplier)
        opsd_network.set_use_lora(opsd_multiplier != 0.0)
        self._lora_mix_active = is_mixed

    def enable_loras(self, lora_key_list=[]):
        self.disable_all_loras()
    
        module_loras = {}  # {module_name: [lora1, lora2, ...]}
        model_device = next(self.parameters()).device
        model_dtype = next(self.parameters()).dtype
        
        for lora_key in lora_key_list:
            if lora_key in self.lora_dict:
                for lora in self.lora_dict[lora_key].loras:
                    lora.to(model_device, dtype=model_dtype, non_blocking=True)
                    module_name = lora.lora_name.replace("lora___lorahyphen___", "").replace("___lorahyphen___", ".")
                    if module_name not in module_loras:
                        module_loras[module_name] = []
                    module_loras[module_name].append(lora)
                self.active_loras.append(lora_key)
    
        for module_name, loras in module_loras.items():
            module = self._get_module_by_name(module_name)
            if not hasattr(module, 'org_forward'):
                module.org_forward = module.forward
            module.forward = self._create_multi_lora_forward(module, loras)
    
    def _create_multi_lora_forward(self, module, loras):
        def multi_lora_forward(x, *args, **kwargs):
            weight_dtype = x.dtype
            org_output = module.org_forward(x, *args, **kwargs)
            
            total_lora_output = 0
            for lora in loras:
                if lora.use_lora:
                    lx = lora.lora_down(x.to(lora.lora_down.weight.dtype))
                    lx = lora.lora_up(lx)
                    lora_output = lx.to(weight_dtype) * lora.multiplier * lora.alpha_scale
                    total_lora_output += lora_output
            
            return org_output + total_lora_output
        
        return multi_lora_forward
    
    def _get_module_by_name(self, module_name):
        try:
            module = self
            for part in module_name.split('.'):
                module = getattr(module, part)
            return module
        except AttributeError as e:
            raise ValueError(f"Cannot find module: {module_name}, error: {e}")
    
    def disable_all_loras(self):
        for name, module in self.named_modules():
            if hasattr(module, 'org_forward'):
                module.forward = module.org_forward
                delattr(module, 'org_forward')
        
        for lora_key, lora_network in self.lora_dict.items():
            for lora in lora_network.loras:
                lora.to("cpu")
        
        self.active_loras.clear()

    def enable_bsa(self,):
        for block in self.blocks:
            block.attn.enable_bsa = True
    
    def disable_bsa(self,):
        for block in self.blocks:
            block.attn.enable_bsa = False    

    def forward(
        self, 
        hidden_states, 
        timestep, 
        encoder_hidden_states, 
        encoder_attention_mask=None, 
        num_cond_latents=0,
        return_kv=False, 
        kv_cache_dict={},
        skip_crs_attn=False, 
        offload_kv_cache=False,
        # avatar related params
        audio_embs=None,
        num_ref_latents=None,
        ref_img_index=None, 
        mask_frame_range=None,
        ref_target_masks=None
    ):

        B, _, T, H, W = hidden_states.shape

        N_t = T // self.patch_size[0]
        N_h = H // self.patch_size[1]
        N_w = W // self.patch_size[2]

        assert self.patch_size[0]==1, "Currently, 3D x_embedder should not compress the temporal dimension."

        # expand the shape of timestep from [B] to [B, T]
        if len(timestep.shape) == 1:
            timestep = timestep.unsqueeze(1).expand(-1, N_t) # [B, T]

        dtype = self.x_embedder.proj.weight.dtype
        hidden_states = hidden_states.to(dtype)
        timestep = timestep.to(dtype)
        encoder_hidden_states = encoder_hidden_states.to(dtype)

        hidden_states = self.x_embedder(hidden_states)  # [B, N, C]

        with amp.autocast(device_type='cuda', dtype=torch.float32):
            t = self.t_embedder(timestep.float().flatten(), dtype=torch.float32).reshape(B, N_t, -1)  # [B, T, C_t]

        encoder_hidden_states = self.y_embedder(encoder_hidden_states)  # [B, 1, N_token, C]

        # get audio token
        audio_cond = audio_embs.to(device=hidden_states.device, dtype=hidden_states.dtype)
        first_frame_audio_emb_s = audio_cond[:, :1, ...] # [B, 1, W, S, C_a]

        latter_frame_audio_emb = audio_cond[:, 1:, ...] # [B, T-1, W, S, C_a]
        latter_frame_audio_emb = rearrange(latter_frame_audio_emb, "b (n_t n) w s c -> b n_t n w s c", n=self.vae_scale)
        middle_index = self.audio_window // 2
        latter_first_frame_audio_emb = latter_frame_audio_emb[:, :, :1, :middle_index+1, ...]
        latter_first_frame_audio_emb = rearrange(latter_first_frame_audio_emb, "b n_t n w s c -> b n_t (n w) s c")

        latter_last_frame_audio_emb = latter_frame_audio_emb[:, :, -1:, middle_index:, ...]
        latter_last_frame_audio_emb = rearrange(latter_last_frame_audio_emb, "b n_t n w s c -> b n_t (n w) s c")

        latter_middle_frame_audio_emb = latter_frame_audio_emb[:, :, 1:-1, middle_index:middle_index+1, ...]
        latter_middle_frame_audio_emb = rearrange(latter_middle_frame_audio_emb, "b n_t n w s c -> b n_t (n w) s c")
        latter_frame_audio_emb_s = torch.concat([latter_first_frame_audio_emb, latter_middle_frame_audio_emb, latter_last_frame_audio_emb], dim=2) # [B, (T-1)//vae_scale, W-1+vae_scale, S, C_a]
        audio_hidden_states = self.audio_proj(first_frame_audio_emb_s, latter_frame_audio_emb_s) # B T 32 768
        
        if num_ref_latents is not None and num_ref_latents > 0:
            audio_start_ref = audio_hidden_states[:, [0], :, :] # padding
            audio_hidden_states = torch.cat([audio_start_ref, audio_hidden_states], dim=1).contiguous()
        audio_hidden_states = audio_hidden_states[:, -N_t:]

        human_num = None
        if ref_target_masks is not None:
            # multitalk
            human_num = len(audio_hidden_states)
            audio_hidden_states = torch.concat(audio_hidden_states.split(1), dim=2) # B T 32 768 --> # 1 T B*32 768
            audio_hidden_states = audio_hidden_states.squeeze(0)
        else:
            audio_hidden_states = rearrange(audio_hidden_states, "b t n c -> (b t) n c")
        
        # convert ref_target_masks to token_ref_target_masks
        # 计算target mask 时会把ref image token 进行 gather，因此这里不需要CP
        token_ref_target_masks = None
        if ref_target_masks is not None:
            ref_target_masks = ref_target_masks.unsqueeze(0).to(torch.float32) # [1, B, H, W]; cast for interpolation
            token_ref_target_masks = nn.functional.interpolate(ref_target_masks, size=(N_h, N_w), mode='nearest') # [1, B, N_h, N_w]
            token_ref_target_masks = token_ref_target_masks.squeeze(0) # [B, N_h, N_w]
            token_ref_target_masks = (token_ref_target_masks > 0)
            token_ref_target_masks = token_ref_target_masks.view(token_ref_target_masks.shape[0], -1) # [B, N_h, N_w] --> [B, N_h * N_w]
            token_ref_target_masks = token_ref_target_masks.to(dtype)
        
        if self.text_tokens_zero_pad and encoder_attention_mask is not None:
            encoder_hidden_states = encoder_hidden_states * encoder_attention_mask[:, None, :, None]
            encoder_attention_mask = (encoder_attention_mask * 0 + 1).to(encoder_attention_mask.dtype)

        if encoder_attention_mask is not None:
            encoder_attention_mask = encoder_attention_mask.squeeze(1).squeeze(1)
            encoder_hidden_states = encoder_hidden_states.squeeze(1).masked_select(encoder_attention_mask.unsqueeze(-1) != 0).view(1, -1, hidden_states.shape[-1]) # [1, N_valid_tokens, C]
            y_seqlens = encoder_attention_mask.sum(dim=1).tolist() # [B]
        else:
            y_seqlens = [encoder_hidden_states.shape[2]] * encoder_hidden_states.shape[0]
            encoder_hidden_states = encoder_hidden_states.squeeze(1).view(1, -1, hidden_states.shape[-1])

        if self.cp_split_hw[0] * self.cp_split_hw[1] > 1:
            hidden_states = rearrange(hidden_states, "B (T H W) C -> B T H W C", T=N_t, H=N_h, W=N_w)
            hidden_states = context_parallel_util.split_cp_2d(hidden_states, seq_dim_hw=(2, 3), split_hw=self.cp_split_hw)
            hidden_states = rearrange(hidden_states, "B T H W C -> B (T H W) C")

        # blocks
        kv_cache_dict_ret = {}
        for i, block in enumerate(self.blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                block_outputs = self._gradient_checkpointing_func(
                    block, hidden_states, encoder_hidden_states, t, y_seqlens,
                    (N_t, N_h, N_w), num_cond_latents, return_kv, kv_cache_dict.get(i, None), skip_crs_attn, num_ref_latents, audio_hidden_states, ref_img_index, mask_frame_range, token_ref_target_masks, human_num
                )
            else:
                block_outputs = block(
                    hidden_states, encoder_hidden_states, t, y_seqlens,
                    (N_t, N_h, N_w), num_cond_latents, return_kv, kv_cache_dict.get(i, None), skip_crs_attn, num_ref_latents, audio_hidden_states, ref_img_index, mask_frame_range, token_ref_target_masks, human_num
                )
            
            if return_kv:
                hidden_states, kv_cache = block_outputs
                if offload_kv_cache:
                    kv_cache_dict_ret[i] = (kv_cache[0].cpu(), kv_cache[1].cpu())
                else:
                    kv_cache_dict_ret[i] = (kv_cache[0].contiguous(), kv_cache[1].contiguous())
            else:
                hidden_states = block_outputs

        hidden_states = self.final_layer(hidden_states, t, (N_t, N_h, N_w))  # [B, N, C=T_p*H_p*W_p*C_out]

        if self.cp_split_hw[0] * self.cp_split_hw[1] > 1:
            hidden_states = context_parallel_util.gather_cp_2d(hidden_states, shape=(N_t, N_h, N_w), split_hw=self.cp_split_hw)

        hidden_states = self.unpatchify(hidden_states, N_t, N_h, N_w)  # [B, C_out, H, W]

        # cast to float32 for better accuracy
        hidden_states = hidden_states.to(torch.float32)

        if return_kv:
            return hidden_states, kv_cache_dict_ret
        else:
            return hidden_states
    

    def unpatchify(self, x, N_t, N_h, N_w):
        """
        Args:
            x (torch.Tensor): of shape [B, N, C]

        Return:
            x (torch.Tensor): of shape [B, C_out, T, H, W]
        """
        T_p, H_p, W_p = self.patch_size
        x = rearrange(
            x,
            "B (N_t N_h N_w) (T_p H_p W_p C_out) -> B C_out (N_t T_p) (N_h H_p) (N_w W_p)",
            N_t=N_t,
            N_h=N_h,
            N_w=N_w,
            T_p=T_p,
            H_p=H_p,
            W_p=W_p,
            C_out=self.out_channels,
        )
        return x
