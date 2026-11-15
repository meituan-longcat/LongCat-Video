import os
from typing import Any, Dict, List, Optional, Union, Literal

import gc
import math
import torch
import loguru
import numpy as np
from einops import rearrange
import torch.nn.functional as F
from tqdm import tqdm 
from PIL import Image
from diffusers.video_processor import VideoProcessor
from diffusers.image_processor import PipelineImageInput
from transformers import AutoTokenizer, UMT5EncoderModel

from longcat_video.modules.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from longcat_video.modules.autoencoder_kl_wan import AutoencoderKLWan
from longcat_video.modules.avatar.longcat_video_dit_avatar import LongCatVideoAvatarTransformer3DModel
from longcat_video.context_parallel import context_parallel_util
from longcat_video.utils.bukcet_config import get_bucket_config

import ftfy
import regex as re
import html

# -------- avatar related --------
import scipy.signal as ss
import pyloudnorm as pyln
from diffusers.image_processor import is_valid_image, is_valid_image_imagelist
import warnings


def torch_gc():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.retrieve_latents
def retrieve_latents(
    encoder_output: torch.Tensor, generator: Optional[torch.Generator] = None, sample_mode: str = "sample"
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")

def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def prompt_clean(text):
    text = whitespace_clean(basic_clean(text))
    return text

def preprocess_video(self, video, height: Optional[int] = None, width: Optional[int] = None, resize_mode: Optional[str] = 'crop') -> torch.Tensor:
    r"""
    hack diffusers.video_processor.VideoProcessor to support the parameter of resize_mode 
    """
    if isinstance(video, list) and isinstance(video[0], np.ndarray) and video[0].ndim == 5:
        warnings.warn(
            "Passing `video` as a list of 5d np.ndarray is deprecated."
            "Please concatenate the list along the batch dimension and pass it as a single 5d np.ndarray",
            FutureWarning,
        )
        video = np.concatenate(video, axis=0)
    if isinstance(video, list) and isinstance(video[0], torch.Tensor) and video[0].ndim == 5:
        warnings.warn(
            "Passing `video` as a list of 5d torch.Tensor is deprecated."
            "Please concatenate the list along the batch dimension and pass it as a single 5d torch.Tensor",
            FutureWarning,
        )
        video = torch.cat(video, axis=0)

    # ensure the input is a list of videos:
    # - if it is a batch of videos (5d torch.Tensor or np.ndarray), it is converted to a list of videos (a list of 4d torch.Tensor or np.ndarray)
    # - if it is a single video, it is converted to a list of one video.
    if isinstance(video, (np.ndarray, torch.Tensor)) and video.ndim == 5:
        video = list(video)
    elif isinstance(video, list) and is_valid_image(video[0]) or is_valid_image_imagelist(video):
        video = [video]
    elif isinstance(video, list) and is_valid_image_imagelist(video[0]):
        video = video
    else:
        raise ValueError(
            "Input is in incorrect format. Currently, we only support numpy.ndarray, torch.Tensor, PIL.Image.Image"
        )

    video = torch.stack([self.preprocess(img, height=height, width=width, resize_mode=resize_mode) for img in video], dim=0)
    video = video.permute(0, 2, 1, 3, 4)

    return video

class LongCatVideoAvatarPipeline:
    r"""
    Pipeline for text-to-video generation using LongCatVideo.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    """

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        text_encoder: UMT5EncoderModel,
        vae: AutoencoderKLWan,
        scheduler: FlowMatchEulerDiscreteScheduler,
        dit: LongCatVideoAvatarTransformer3DModel,
        audio_encoder = None,
        audio_feature_extractor = None,
        model_type="avatar-v1.0",
    ):
        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.scheduler = scheduler
        self.dit = dit 
        self.device = "cuda"

        self.vae_scale_factor_temporal = self.vae.config.scale_factor_temporal if getattr(self, "vae", None) else 4
        self.vae_scale_factor_spatial = self.vae.config.scale_factor_spatial if getattr(self, "vae", None) else 8 
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)
        self.video_processor.preprocess_video = preprocess_video

        self._num_timesteps = 1000
        self._num_distill_sample_steps = 50

        self.audio_encoder = audio_encoder
        self.audio_feature_extractor = audio_feature_extractor

        self.model_type = model_type
        self.use_vcond = True
        if self.model_type == "avatar-v1.5":
            self.use_vcond = False
            self._num_distill_sample_steps = 8

    @staticmethod
    def _cp_tile_bounds(rank, split_h, split_w, height, width, overlap):
        row, col = divmod(rank, split_w)
        core_h, core_w = height // split_h, width // split_w
        return (
            max(0, row * core_h - overlap),
            min(height, (row + 1) * core_h + overlap),
            max(0, col * core_w - overlap),
            min(width, (col + 1) * core_w + overlap),
        )

    def _decode_video_latents(self, latents):
        cp_size = context_parallel_util.get_cp_size()
        if cp_size == 1:
            return self.vae.decode(latents, return_dict=False)[0]

        cp_rank = context_parallel_util.get_cp_rank()
        cp_group = context_parallel_util.get_cp_group()
        cp_ranks = context_parallel_util.get_cp_rank_list()
        split_h, split_w = context_parallel_util.get_optimal_split(cp_size)
        height, width = latents.shape[-2:]
        if height % split_h or width % split_w:
            raise ValueError(
                f"VAE tile shape {(height, width)} must be divisible by CP split {(split_h, split_w)}"
            )

        scale = self.vae.spatial_compression_ratio
        overlap = max(
            1,
            (self.vae.tile_sample_min_height - self.vae.tile_sample_stride_height) // scale,
        )
        top, bottom, left, right = self._cp_tile_bounds(
            cp_rank, split_h, split_w, height, width, overlap
        )
        local_tile = self.vae.decode(
            latents[..., top:bottom, left:right].contiguous(), return_dict=False
        )[0].contiguous()

        if cp_rank == 0:
            tiles = [local_tile]
            for tile_rank in range(1, cp_size):
                tile_top, tile_bottom, tile_left, tile_right = self._cp_tile_bounds(
                    tile_rank, split_h, split_w, height, width, overlap
                )
                tile = torch.empty(
                    (*local_tile.shape[:3], (tile_bottom - tile_top) * scale, (tile_right - tile_left) * scale),
                    dtype=local_tile.dtype,
                    device=local_tile.device,
                )
                torch.distributed.recv(tile, src=cp_ranks[tile_rank])
                tiles.append(tile)

            output_shape = (*local_tile.shape[:3], height * scale, width * scale)
            output_sum = torch.zeros(output_shape, dtype=torch.float32, device=local_tile.device)
            weight_sum = torch.zeros((1, 1, 1, height * scale, width * scale), dtype=torch.float32, device=local_tile.device)
            overlap_pixels = overlap * scale

            for tile_rank, tile in enumerate(tiles):
                row, col = divmod(tile_rank, split_w)
                tile_top, tile_bottom, tile_left, tile_right = self._cp_tile_bounds(
                    tile_rank, split_h, split_w, height, width, overlap
                )
                tile_h, tile_w = tile.shape[-2:]
                weight_h = torch.ones(tile_h, dtype=torch.float32, device=tile.device)
                weight_w = torch.ones(tile_w, dtype=torch.float32, device=tile.device)
                blend_h = min(2 * overlap_pixels, tile_h)
                blend_w = min(2 * overlap_pixels, tile_w)
                if row > 0:
                    weight_h[:blend_h] = torch.linspace(0, 1, blend_h, device=tile.device)
                if row + 1 < split_h:
                    weight_h[-blend_h:] *= torch.linspace(1, 0, blend_h, device=tile.device)
                if col > 0:
                    weight_w[:blend_w] = torch.linspace(0, 1, blend_w, device=tile.device)
                if col + 1 < split_w:
                    weight_w[-blend_w:] *= torch.linspace(1, 0, blend_w, device=tile.device)
                weight = weight_h[:, None] * weight_w[None, :]
                sample_top, sample_bottom = tile_top * scale, tile_bottom * scale
                sample_left, sample_right = tile_left * scale, tile_right * scale
                output_sum[..., sample_top:sample_bottom, sample_left:sample_right] += tile.float() * weight
                weight_sum[..., sample_top:sample_bottom, sample_left:sample_right] += weight

            output_video = (output_sum / weight_sum.clamp_min_(1e-6)).to(local_tile.dtype)
        else:
            torch.distributed.send(local_tile, dst=cp_ranks[0])
            output_video = torch.empty(
                (*local_tile.shape[:3], height * scale, width * scale),
                dtype=local_tile.dtype,
                device=local_tile.device,
            )

        torch.distributed.broadcast(output_video, src=cp_ranks[0], group=cp_group)
        return output_video

    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 512,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt = [prompt_clean(u) for u in prompt]
        batch_size = len(prompt)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask

        prompt_embeds = self.text_encoder(text_input_ids.to(device), mask.to(device)).last_hidden_state
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        mask = mask.to(device=device)
        mask = torch.cat([mask]*num_videos_per_prompt, dim=0)

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, 1, seq_len, -1)

        return prompt_embeds, mask

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        do_classifier_free_guidance: bool = True,
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 512,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                Number of videos that should be generated per prompt. torch device to place the resulting embeddings on
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            device: (`torch.device`, *optional*):
                torch device
            dtype: (`torch.dtype`, *optional*):
                torch dtype
        """

        prompt_cache_key = (
            tuple([prompt] if isinstance(prompt, str) else prompt),
            None if negative_prompt is None else tuple(
                [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
            ),
            do_classifier_free_guidance,
            num_videos_per_prompt,
            max_sequence_length,
            str(device),
            dtype,
        )
        if getattr(self, "_prompt_cache_key", None) == prompt_cache_key:
            return self._prompt_cache

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        prompt_embeds, prompt_attention_mask = self._get_t5_prompt_embeds(
            prompt=prompt,
            num_videos_per_prompt=num_videos_per_prompt,
            max_sequence_length=max_sequence_length,
            device=device,
            dtype=dtype,
        )

        if do_classifier_free_guidance:
            negative_prompt = negative_prompt or ""
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt

            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )

            negative_prompt_embeds, negative_prompt_attention_mask = self._get_t5_prompt_embeds(
                prompt=negative_prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )
        else:
            negative_prompt_embeds = None
            negative_prompt_attention_mask = None
            
        result = (
            prompt_embeds,
            prompt_attention_mask,
            negative_prompt_embeds,
            negative_prompt_attention_mask,
        )
        self._prompt_cache_key = prompt_cache_key
        self._prompt_cache = result
        return result

    def check_inputs(
        self,
        prompt,
        negative_prompt,
        height,
        width,
        scale_factor_spatial
    ):
        # Check height and width divisibility
        if height % scale_factor_spatial != 0 or width % scale_factor_spatial != 0:
            raise ValueError(f"`height and width` have to be divisible by {scale_factor_spatial} but are {height} and {width}.")

        # Check prompt validity
        if prompt is None:
            raise ValueError("Cannot leave `prompt` undefined.")
        
        if prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt has to be of type str or list` but is {type(prompt)}")
        
        # Check negative prompt validity
        if negative_prompt is not None and (not isinstance(negative_prompt, str) and not isinstance(negative_prompt, list)):
            raise ValueError(f"`negative_prompt has to be of type str or list` but is {type(negative_prompt)}")
        
    def prepare_latents(
        self,
        image: Optional[torch.Tensor] = None,
        video: Optional[torch.Tensor] = None,
        batch_size: int = 1,
        num_channels_latents: int = 16,
        height: int = 480,
        width: int = 832,
        num_frames: int = 93,
        num_cond_frames: int = 0,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        num_cond_frames_added: int = 0,
        need_encode: bool = True
    ) -> torch.Tensor:
        if (image is not None) and (video is not None):
            raise ValueError("Cannot provide both `image and video` at the same time. Please provide only one.")
        if latents is not None:
            latents = latents.to(device=device, dtype=dtype)
        else:
            num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
            shape = (
                batch_size,
                num_channels_latents,
                num_latent_frames,
                int(height) // self.vae_scale_factor_spatial,
                int(width) // self.vae_scale_factor_spatial,
            )
            if isinstance(generator, list) and len(generator) != batch_size:
                raise ValueError(
                    f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                    f" size of {batch_size}. Make sure the batch size matches the length of the generators."
                )

            # Generate random noise with shape latent_shape
            latents = torch.randn(shape, generator=generator, device=device, dtype=dtype)

        if image is not None or video is not None:
            if isinstance(generator, list):
                if len(generator) != batch_size:
                    raise ValueError(
                        f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                        f" size of {batch_size}. Make sure the batch size matches the length of the generators."
                    )
            condition_data = image if image is not None else video
            num_cond_latents = 1 + (num_cond_frames - 1) // self.vae_scale_factor_temporal

            if need_encode:
                is_image = image is not None
                cond_latents = []
                for i in range(batch_size):
                    gen = generator[i] if isinstance(generator, list) else generator
                    if is_image:
                        encoded_input = condition_data[i].unsqueeze(0).unsqueeze(2)
                    else:
                        encoded_input = condition_data[i][:, -(num_cond_frames-num_cond_frames_added):].unsqueeze(0)
                    if num_cond_frames_added > 0:
                        pad_front = encoded_input[:, :, 0:1].repeat(1, 1, num_cond_frames_added, 1, 1)
                        encoded_input = torch.cat([pad_front, encoded_input], dim=2)
                    assert encoded_input.shape[2] == num_cond_frames
                    latent = retrieve_latents(self.vae.encode(encoded_input), gen, sample_mode="argmax")
                    cond_latents.append(latent)

                cond_latents = torch.cat(cond_latents, dim=0).to(dtype)
                cond_latents = self.normalize_latents(cond_latents)
            else:
                cond_latents = condition_data[:, :, -num_cond_latents:]
            
            latents[:, :, :num_cond_latents] = cond_latents

        return latents

    @property
    def text_guidance_scale(self):
        return self._text_guidance_scale
    
    @property
    def audio_guidance_scale(self):
        return self._audio_guidance_scale

    @property
    def do_classifier_free_guidance(self):
        return self._text_guidance_scale > 1.0 or self._audio_guidance_scale > 1.0

    @property
    def num_timesteps(self):
        return self._num_timesteps
    
    @property
    def num_distill_sample_steps(self):
        return self._num_distill_sample_steps
    
    @property
    def current_timestep(self):
        return self._current_timestep

    @property
    def interrupt(self):
        return self._interrupt

    @property
    def attention_kwargs(self):
        return self._attention_kwargs
    
    def get_timesteps_sigmas(self, sampling_steps: int, use_distill: bool=False):
        if use_distill:            
            if self.model_type == "avatar-v1.5":
                distill_indices = torch.arange(1, self.num_distill_sample_steps + 1, dtype=torch.float32)
                distill_indices = (distill_indices * (self.num_timesteps // self.num_distill_sample_steps)).round().long()
                distill_indices = self.num_timesteps - distill_indices
                sigmas = torch.flip(torch.linspace(0, 1, self.num_timesteps), [0])
                sigmas = torch.flip(sigmas[distill_indices], [0]).float()
            else:
                distill_indices = torch.arange(1, self.num_distill_sample_steps + 1, dtype=torch.float32)
                distill_indices = (distill_indices * (self.num_timesteps // self.num_distill_sample_steps)).round().long()
                inference_indices = np.linspace(0, self.num_distill_sample_steps, num=sampling_steps, endpoint=False)
                inference_indices = np.floor(inference_indices).astype(np.int64)
            
                sigmas = torch.flip(distill_indices, [0])[inference_indices].float() / self.num_timesteps
        else:
            sigmas = torch.linspace(1, 0.001, sampling_steps)
        sigmas = sigmas.to(torch.float32)
        return sigmas

    def _update_kv_cache_dict(self, kv_cache_dict):
        self.kv_cache_dict = kv_cache_dict

    def _cache_clean_latents(self, cond_latents, model_max_length, offload_kv_cache, device, dtype, audio_embs, num_cond_latents, num_ref_latents, ref_img_index):
        timestep = torch.zeros(cond_latents.shape[0], cond_latents.shape[2]).to(device=device, dtype=dtype)
        # make null prompt tensor(skip_crs_attn=True, so tensors below will not be actually used)
        empty_embeds = torch.zeros([cond_latents.shape[0], 1, model_max_length, self.text_encoder.config.d_model], device=device, dtype=dtype)
        _, kv_cache_dict = self.dit(
            hidden_states=cond_latents, 
            timestep=timestep, 
            encoder_hidden_states=empty_embeds,
            num_cond_latents=num_cond_latents,
            return_kv=True, 
            skip_crs_attn=True, 
            offload_kv_cache=offload_kv_cache,
            audio_embs=audio_embs,
            num_ref_latents=num_ref_latents,
            ref_img_index=ref_img_index
        )
        self._update_kv_cache_dict(kv_cache_dict)
    
    def _get_kv_cache_dict(self):
        return self.kv_cache_dict
    
    def _clear_cache(self):
        self.kv_cache_dict = None
        gc.collect()
        torch.cuda.empty_cache()

    def get_condition_shape(self, condition, resolution, scale_factor_spatial=32):
        bucket_config = get_bucket_config(resolution, scale_factor_spatial=scale_factor_spatial)

        obj = condition[0] if isinstance(condition, list) and condition else condition
        try:
            height = getattr(obj, "height")
            width = getattr(obj, "width")
        except AttributeError:
            raise ValueError("Unsupported condition type")

        ratio = height / width
        # Find the closest bucket
        closest_bucket = sorted(list(bucket_config.keys()), key=lambda x: abs(float(x) - ratio))[0]
        target_h, target_w = bucket_config[closest_bucket][0]
        return target_h, target_w
    
    def optimized_scale(self, positive_flat, negative_flat):
        """ from CFG-zero paper
        """
        # Calculate dot production
        dot_product = torch.sum(positive_flat * negative_flat, dim=1, keepdim=True)
        # Squared norm of uncondition
        squared_norm = torch.sum(negative_flat ** 2, dim=1, keepdim=True) + 1e-8
        # st_star = v_condˆT * v_uncond / ||v_uncond||ˆ2
        st_star = dot_product / squared_norm
        return st_star
    
    def normalize_latents(self, latents):
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            latents.device, latents.dtype
        )
        return (latents - latents_mean) * latents_std

    def denormalize_latents(self, latents):
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            latents.device, latents.dtype
        )
        return latents / latents_std + latents_mean

    def _loudness_norm(self, audio_array, sr=16000, lufs=-23, threshold=100):
        meter = pyln.Meter(sr)
        loudness = meter.integrated_loudness(audio_array)
        if abs(loudness) > threshold:
            return audio_array
        normalized_audio = pyln.normalize.loudness(audio_array, loudness, lufs)
        return normalized_audio

    def _add_noise_floor(self, audio, noise_db=-45):
        noise_amp = 10 ** (noise_db / 20)
        noise = np.random.randn(len(audio)) * noise_amp
        return audio + noise

    def _smooth_transients(self, audio, sr=16000):
        b, a = ss.butter(3, 3000 / (sr/2))
        return ss.lfilter(b, a, audio)
    
    def _resize_and_centercrop_tensor(self, mask: torch.Tensor, target_h: int, target_w: int, resize_mode: str = 'crop'):
        """
        mask: Tensor, shape [3, H, W], dtype=float, device=gpu/cpu
        return: [3, target_h, target_w]
        """

        if resize_mode == 'default':
            mask_resized = F.interpolate(
                mask.unsqueeze(0),  # [1, 3, H, W]
                size=(new_h, new_w),
                mode="bilinear",
                align_corners=False
            ).squeeze(0)
            return mask_resized

        elif resize_mode == 'crop':
            _, H, W = mask.shape
            ratio = target_w / target_h # 1
            src_ratio = W / H # > 1

            if ratio > src_ratio:
                new_w = target_w
                new_h = int(H * target_w / W)
            else:
                new_h = target_h
                new_w = int(W * target_h / H)

            mask_resized = F.interpolate(
                mask.unsqueeze(0),  # [1, 3, H, W]
                size=(new_h, new_w),
                mode="bilinear",
                align_corners=False
            ).squeeze(0)

            top = (new_h - target_h) // 2
            left = (new_w - target_w) // 2

            mask_resized_cropped = mask_resized[:, top:top + target_h, left:left + target_w]
            return mask_resized_cropped
        
        else:
            raise NotImplementedError(f"Not supported resize_mode {resize_mode}")

    def get_audio_embedding(self, speech_array, fps=32, device='cpu', sample_rate=16000,
                            model_type="avatar-v1.0", use_loudness_norm=True):
        if model_type == "avatar-v1.0":
            return self.get_audio_embedding_wav2vec2(speech_array, fps, device, sample_rate)
        if model_type == "avatar-v1.5":
            return self.get_audio_embedding_whisper(
                speech_array, fps, device, sample_rate, use_loudness_norm=use_loudness_norm
            )

    @torch.no_grad()
    def get_audio_embedding_whisper(self, speech_array, fps=25, device='cpu', sample_rate=16000,
                                    use_loudness_norm=True):
        """使用 Whisper encoder 提取音频特征。
        Args:
            speech_array: 原始音频波形 (numpy array, 单声道, sample_rate=16000)
            fps:          目标视频帧率
            device:       推理设备
            sample_rate:  音频采样率
        Returns:
            audio_emb: [T, 5, D]，T = int(audio_duration * fps)
        """
        def linear_interpolation_fps(features, input_fps, output_fps, output_len=None):
            """将音频特征从 input_fps 插值到 output_fps。
            Args:
                features:   [B, T, D]
                input_fps:  源帧率
                output_fps: 目标帧率
                output_len: 若指定则直接用该长度，否则按帧率比换算
            Returns:
                [B, output_len, D]
            """
            features = features.transpose(1, 2)   # [B, D, T]
            if output_len is None:
                output_len = int(features.shape[2] / float(input_fps) * output_fps)
            output_features = F.interpolate(features, size=output_len, align_corners=True, mode='linear')
            return output_features.transpose(1, 2)
        # ---- 常量 ----
        MEL_CHUNK      = 750 * 640   # feature extractor 滑窗大小（样本数）
        ENC_CHUNK      = 3000        # encoder 滑窗大小（mel 帧数）
        ENC_FPS        = 50          # encoder 输出帧率

        # ---- 时长 / 帧数 ----
        audio_duration = len(speech_array) / sample_rate
        video_length   = int(audio_duration * fps)

        # ---- 音频预处理 ----
        if use_loudness_norm:
            speech_array = self._loudness_norm(speech_array, sample_rate)

        # ---- Whisper feature extractor：wav → mel spectrogram ----
        mel_chunks = []
        for i in range(0, len(speech_array), MEL_CHUNK):
            mel = self.audio_feature_extractor(
                speech_array[i: i + MEL_CHUNK],
                sampling_rate=sample_rate,
                return_tensors="pt",
            ).input_features                          # [1, mel_bins, T_mel]
            mel_chunks.append(mel)
        audio_features = torch.cat(mel_chunks, dim=-1)   # [1, mel_bins, T_mel_total]
        audio_features = audio_features.to(self.audio_encoder.dtype)

        # ---- Whisper encoder：mel → hidden states ----
        enc_chunks = []
        for i in range(0, audio_features.shape[-1], ENC_CHUNK):
            chunk_hs = self.audio_encoder.encoder(
                audio_features[:, :, i: i + ENC_CHUNK].to(device),
                output_hidden_states=True,
            ).hidden_states                           # tuple: (n_layers+1,) x [1, T_enc, D]
            enc_chunks.append(torch.stack(chunk_hs, dim=2))  # [1, T_enc, n_layers, D]
        audio_prompts = torch.cat(enc_chunks, dim=1)         # [1, T_enc_total, n_layers, D]
        audio_prompts = audio_prompts[:, :video_length * 2]  # 截取有效帧

        # ---- 按层分组 + 插值到目标帧数 ----
        feat0 = linear_interpolation_fps(audio_prompts[:, :,  0: 8].mean(dim=2), ENC_FPS, fps, video_length)
        feat1 = linear_interpolation_fps(audio_prompts[:, :,  8:16].mean(dim=2), ENC_FPS, fps, video_length)
        feat2 = linear_interpolation_fps(audio_prompts[:, :, 16:24].mean(dim=2), ENC_FPS, fps, video_length)
        feat3 = linear_interpolation_fps(audio_prompts[:, :, 24:32].mean(dim=2), ENC_FPS, fps, video_length)
        feat4 = linear_interpolation_fps(audio_prompts[:, :, 32],                ENC_FPS, fps, video_length)
        audio_emb = torch.stack([feat0, feat1, feat2, feat3, feat4], dim=2)[0]   # [T, 5, D]
        return audio_emb
    
    @torch.no_grad()
    def get_audio_embedding_wav2vec2(self, speech_array, fps=32, device='cpu', sample_rate=16000):
            
        audio_duration = len(speech_array) / sample_rate
        video_length = audio_duration * fps

        # speech preprocess
        speech_array = self._loudness_norm(speech_array, sample_rate)
        speech_array = self._add_noise_floor(speech_array)
        speech_array = self._smooth_transients(speech_array)

        # wav2vec_feature_extractor
        audio_feature = np.squeeze(
            self.audio_feature_extractor(speech_array, sampling_rate=sample_rate).input_values
        )
        audio_feature = torch.from_numpy(audio_feature).float().to(device=device)
        audio_feature = audio_feature.unsqueeze(0)

        # audio embedding
        embeddings = self.audio_encoder(audio_feature, seq_len=int(video_length), output_hidden_states=True)

        audio_emb = torch.stack(embeddings.hidden_states[1:], dim=1).squeeze(0)
        audio_emb = rearrange(audio_emb, "b s d -> s b d").contiguous() # T, 12, 768
        return audio_emb

    @torch.no_grad()
    def generate_at2v(
        self,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        height: int = 480,
        width: int = 832,
        num_frames: int = 93,
        num_inference_steps: int = 50,
        use_distill: bool = False,
        text_guidance_scale: float = 4.0,
        audio_guidance_scale: float = 4.0,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "np",
        attention_kwargs: Optional[Dict[str, Any]] = None,
        max_sequence_length: int = 512,
        # avatar related params
        audio_emb: torch.Tensor = None
    ):
        r"""
        Generates video frames from text prompt using diffusion process.

        Args:
            prompt (`str or List[str]`):
                Text prompt(s) for video content generation.
            negative_prompt (`str or List[str]`, *optional*):
                Negative prompt(s) for content exclusion. If not provided, uses empty string.
            height (`int`, *optional*, defaults to 480):
                Height of each video frame. Must be divisible by 16.
            width (`int`, *optional*, defaults to 832):
                Width of each video frame. Must be divisible by 16.
            num_frames (`int`, *optional*, defaults to 93):
                Number of frames to generate for the video. Should satisfy (num_frames - 1) % vae_scale_factor_temporal == 0.
            num_inference_steps (`int`, *optional*, defaults to 50):
                Number of diffusion sampling steps. Higher values improve quality but slow generation.
            use_distill (`bool`, *optional*, defaults to False):
                Whether to use distillation sampling schedule.
            text_guidance_scale (`float`, *optional*, defaults to 4.0):
                Classifier-free guidance scale. Controls prompt adherence vs. creativity.
            audio_guidance_scale (`float`, *optional*, defaults to 4.0):
                Classifier-free guidance scale. Controls audio adherence. Larger values may lead to exaggerated mouth.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                Number of videos to generate per prompt.
            generator (`torch.Generator or List[torch.Generator]`, *optional*):
                Random seed generator(s) for noise generation.
            latents (`torch.Tensor`, *optional*):
                Precomputed latent tensor. If not provided, random latents are generated.
            output_type (`str`, *optional*, defaults to "np"):
                Output format type. "np" for numpy array, "latent" for latent tensor.
            attention_kwargs (`Dict[str, Any]`, *optional*):
                Additional attention parameters for the model.
            max_sequence_length (`int`, *optional*, defaults to 512):
                Maximum sequence length for text encoding.
            audio_emb (`torch.Tensor`):
                Audio embedding to driven the lip movements and body motions of character.

        Returns:
            np.ndarray or torch.Tensor:
                Generated video frames. If output_type is "np", returns numpy array of shape (B, N, H, W, C).
                If output_type is "latent", returns latent tensor.
        """

        # 1. Check inputs. Raise error if not correct
        scale_factor_spatial = self.vae_scale_factor_spatial * 2
        if self.dit.cp_split_hw is not None:
            scale_factor_spatial *= max(self.dit.cp_split_hw)
        self.check_inputs(
            prompt,
            negative_prompt,
            height,
            width,
            scale_factor_spatial
        )

        if num_frames % self.vae_scale_factor_temporal != 1:
            loguru.logger.warning(
                f"`num_frames - 1` has to be divisible by {self.vae_scale_factor_temporal}. Rounding to the nearest number."
            )
            num_frames = num_frames // self.vae_scale_factor_temporal * self.vae_scale_factor_temporal + 1
        num_frames = max(num_frames, 1)

        self._text_guidance_scale = text_guidance_scale
        self._audio_guidance_scale = audio_guidance_scale
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        device = self.device

        # 2. Define call parameters
        if isinstance(prompt, str):
            batch_size = 1
        else:
            batch_size = len(prompt)


        # 3. Encode inputs
        dit_dtype = self.dit.dtype

        if context_parallel_util.get_cp_rank() == 0:
            (
                prompt_embeds, 
                prompt_attention_mask, 
                negative_prompt_embeds, 
                negative_prompt_attention_mask,
            ) = self.encode_prompt(
                prompt=prompt,
                negative_prompt=negative_prompt,
                do_classifier_free_guidance=self.do_classifier_free_guidance,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                dtype=dit_dtype,
                device=device,
            )
            if context_parallel_util.get_cp_size() > 1:
                context_parallel_util.cp_broadcast(prompt_embeds)
                context_parallel_util.cp_broadcast(prompt_attention_mask)
                if self.do_classifier_free_guidance:
                    context_parallel_util.cp_broadcast(negative_prompt_embeds)
                    context_parallel_util.cp_broadcast(negative_prompt_attention_mask)
        elif context_parallel_util.get_cp_size() > 1:
            caption_channels = self.text_encoder.config.d_model
            prompt_embeds = torch.zeros([batch_size, 1, max_sequence_length, caption_channels], dtype=dit_dtype, device=device)
            prompt_attention_mask = torch.zeros([batch_size, max_sequence_length], dtype=torch.int64, device=device)
            context_parallel_util.cp_broadcast(prompt_embeds)
            context_parallel_util.cp_broadcast(prompt_attention_mask)
            if self.do_classifier_free_guidance:
                negative_prompt_embeds = torch.zeros([batch_size, 1, max_sequence_length, caption_channels], dtype=dit_dtype, device=device)
                negative_prompt_attention_mask = torch.zeros([batch_size, max_sequence_length], dtype=torch.int64, device=device)
                context_parallel_util.cp_broadcast(negative_prompt_embeds)
                context_parallel_util.cp_broadcast(negative_prompt_attention_mask)

        audio_cond_embs = torch.cat([audio_emb] * num_videos_per_prompt, dim=0)
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            prompt_attention_mask = torch.cat([negative_prompt_attention_mask, prompt_attention_mask], dim=0)
            audio_unond_embs = torch.zeros_like(audio_cond_embs)
            audio_cond_embs = torch.cat([audio_cond_embs, audio_cond_embs], dim=0)

        # 4. Prepare timesteps
        sigmas = self.get_timesteps_sigmas(num_inference_steps, use_distill=use_distill)
        self.scheduler.set_timesteps(num_inference_steps, sigmas=sigmas, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.dit.config.in_channels
            
        latents = self.prepare_latents(
            batch_size=batch_size * num_videos_per_prompt,
            num_channels_latents=num_channels_latents,
            height=height,
            width=width,
            num_frames=num_frames,
            dtype=torch.float32,
            device=device,
            generator=generator,
            latents=latents,
        )
        if context_parallel_util.get_cp_size() > 1:
            context_parallel_util.cp_broadcast(latents)
        
        # 6. Denoising loop
        if context_parallel_util.get_cp_size() > 1:
            torch.distributed.barrier(group=context_parallel_util.get_cp_group())

        with tqdm(total=len(timesteps), desc="Denoising") as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue
                if self.dit.lora_mix_config is not None:
                    self.dit.set_lora_mix_step(i)

                self._current_timestep = t

                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                latent_model_input = latent_model_input.to(dit_dtype)

                timestep = t.expand(latent_model_input.shape[0]).to(dit_dtype)

                noise_pred = self.dit(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    encoder_attention_mask=prompt_attention_mask,
                    audio_embs=audio_cond_embs
                )

                if self.do_classifier_free_guidance:
                    timestep_uncond = t.expand(latents.shape[0]).to(dit_dtype)
                    noise_pred_uncond = self.dit(
                        hidden_states=latents,
                        timestep=timestep_uncond,
                        encoder_hidden_states=negative_prompt_embeds,
                        encoder_attention_mask=negative_prompt_attention_mask,
                        audio_embs=audio_unond_embs
                    )

                    noise_pred_uncond_text, noise_pred_cond = noise_pred.chunk(2)

                    noise_pred = noise_pred_uncond + text_guidance_scale * (noise_pred_cond - noise_pred_uncond_text) + audio_guidance_scale * (noise_pred_uncond_text - noise_pred_uncond)

                # negate for scheduler compatibility
                noise_pred = -noise_pred

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                # call the callback, if provided
                if i == len(timesteps) - 1 or (i + 1) % self.scheduler.order == 0:
                    progress_bar.update()

        self._current_timestep = None

        if output_type == 'latent':
            return latents
        
        if output_type == 'both':
            latents_ = latents.clone()

        latents = latents.to(self.vae.dtype)
        latents = self.denormalize_latents(latents)
        output_video = self._decode_video_latents(latents)
        output_video = self.video_processor.postprocess_video(output_video)

        if output_type == 'both':
            return (output_video, latents_)
        else:
            return output_video
    

    @torch.no_grad()
    def generate_ai2v(
        self,
        image: PipelineImageInput,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        resolution: Literal["480p", "720p"] = "480p",
        num_frames: int = 93,
        num_inference_steps: int = 50,
        use_distill: bool = False,
        text_guidance_scale: float = 4.0,
        audio_guidance_scale: float = 4.0,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "np",
        attention_kwargs: Optional[Dict[str, Any]] = None,
        max_sequence_length: int = 512,
        # avatar related params
        audio_emb: torch.Tensor = None,
        ref_target_masks: torch.Tensor = None,
        resize_mode: Optional[str] = "crop", # "default" / "crop"
    ):
        r"""
        Generates video frames from an input image and text prompt using diffusion process.

        Args:
            image (`PipelineImageInput`):
                Input image for video generation.
            prompt (`str or List[str]`, *optional*):
                Text prompt(s) for video content generation.
            negative_prompt (`str or List[str]`, *optional*):
                Negative prompt(s) for content exclusion. If not provided, uses empty string.
            resolution (`Literal["480p", "720p"]`, *optional*, defaults to "480p"):
                Target video resolution. Determines output frame size.
            num_frames (`int`, *optional*, defaults to 93):
                Number of frames to generate for the video. Should satisfy (num_frames - 1) % vae_scale_factor_temporal == 0.
            num_inference_steps (`int`, *optional*, defaults to 50):
                Number of diffusion sampling steps. Higher values improve quality but slow generation.
            use_distill (`bool`, *optional*, defaults to False):
                Whether to use distillation sampling schedule.
            text_guidance_scale (`float`, *optional*, defaults to 4.0):
                Classifier-free guidance scale. Controls prompt adherence vs. creativity.
            audio_guidance_scale (`float`, *optional*, defaults to 4.0):
                Classifier-free guidance scale. Controls audio adherence. Larger values may lead to exaggerated mouth.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                Number of videos to generate per prompt.
            generator (`torch.Generator or List[torch.Generator]`, *optional*):
                Random seed generator(s) for noise generation.
            latents (`torch.Tensor`, *optional*):
                Precomputed latent tensor. If not provided, random latents are generated.
            output_type (`str`, *optional*, defaults to "np"):
                Output format type. "np" for numpy array, "latent" for latent tensor.
            attention_kwargs (`Dict[str, Any]`, *optional*):
                Additional attention parameters for the model.
            max_sequence_length (`int`, *optional*, defaults to 512):
                Maximum sequence length for text encoding.
            audio_emb (`torch.Tensor`):
                Audio embedding to driven the lip movements and body motions of character.
            ref_target_masks(`torch.Tensor`, *optional*, defaults to None):
                Mask used in dual-speaker audio-driven mode.
            resize_mode(`str`, *optional*):
                Output format type. "default" for resize, "crop" for shorter-length resize and centercrop.

        Returns:
            np.ndarray or torch.Tensor:
                Generated video frames. If output_type is "np", returns numpy array of shape (B, N, H, W, C).
                If output_type is "latent", returns latent tensor.
        """

        # 1. Check inputs. Raise error if not correct
        scale_factor_spatial = self.vae_scale_factor_spatial * 2
        if self.dit.cp_split_hw is not None:
            scale_factor_spatial *= max(self.dit.cp_split_hw)
        height, width = self.get_condition_shape(image, resolution, scale_factor_spatial=scale_factor_spatial)
        self.check_inputs(
            prompt,
            negative_prompt,
            height,
            width,
            scale_factor_spatial
        )
        assert resize_mode in ['default', 'crop'], f"Unsupported resize_mode {resize_mode}, and you can only choose from [default, crop]"

        if num_frames % self.vae_scale_factor_temporal != 1:
            loguru.logger.warning(
                f"`num_frames - 1` has to be divisible by {self.vae_scale_factor_temporal}. Rounding to the nearest number."
            )
            num_frames = num_frames // self.vae_scale_factor_temporal * self.vae_scale_factor_temporal + 1
        num_frames = max(num_frames, 1)


        self._text_guidance_scale = text_guidance_scale
        self._audio_guidance_scale = audio_guidance_scale
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        device = self.device

        # 2. Define call parameters
        if isinstance(prompt, str):
            batch_size = 1
        else:
            batch_size = len(prompt)


        # 3. Encode inputs
        dit_dtype = self.dit.dtype

        if context_parallel_util.get_cp_rank() == 0:
            (
                prompt_embeds, 
                prompt_attention_mask, 
                negative_prompt_embeds, 
                negative_prompt_attention_mask,
            ) = self.encode_prompt(
                prompt=prompt,
                negative_prompt=negative_prompt,
                do_classifier_free_guidance=self.do_classifier_free_guidance,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                dtype=dit_dtype,
                device=device,
            )
            if context_parallel_util.get_cp_size() > 1:
                context_parallel_util.cp_broadcast(prompt_embeds)
                context_parallel_util.cp_broadcast(prompt_attention_mask)
                if self.do_classifier_free_guidance:
                    context_parallel_util.cp_broadcast(negative_prompt_embeds)
                    context_parallel_util.cp_broadcast(negative_prompt_attention_mask)
        elif context_parallel_util.get_cp_size() > 1:
            caption_channels = self.text_encoder.config.d_model
            prompt_embeds = torch.zeros([batch_size, 1, max_sequence_length, caption_channels], dtype=dit_dtype, device=device)
            prompt_attention_mask = torch.zeros([batch_size, max_sequence_length], dtype=torch.int64, device=device)
            context_parallel_util.cp_broadcast(prompt_embeds)
            context_parallel_util.cp_broadcast(prompt_attention_mask)
            if self.do_classifier_free_guidance:
                negative_prompt_embeds = torch.zeros([batch_size, 1, max_sequence_length, caption_channels], dtype=dit_dtype, device=device)
                negative_prompt_attention_mask = torch.zeros([batch_size, max_sequence_length], dtype=torch.int64, device=device)
                context_parallel_util.cp_broadcast(negative_prompt_embeds)
                context_parallel_util.cp_broadcast(negative_prompt_attention_mask)

        audio_cond_embs = torch.cat([audio_emb] * num_videos_per_prompt, dim=0)
        audio_num = audio_cond_embs.shape[0]
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            prompt_attention_mask = torch.cat([negative_prompt_attention_mask, prompt_attention_mask], dim=0)
            audio_unond_embs = torch.zeros_like(audio_cond_embs)
            audio_cond_embs = torch.cat([audio_cond_embs, audio_cond_embs], dim=0)
        
        # 4. Prepare timesteps
        sigmas = self.get_timesteps_sigmas(num_inference_steps, use_distill=use_distill)
        self.scheduler.set_timesteps(num_inference_steps, sigmas=sigmas, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        image = self.video_processor.preprocess(image, height=height, width=width, resize_mode=resize_mode)
        image = image.to(device=device, dtype=prompt_embeds.dtype)

        num_channels_latents = self.dit.config.in_channels
            
        latents = self.prepare_latents(
            image=image, 
            batch_size=batch_size * num_videos_per_prompt,
            num_channels_latents=num_channels_latents,
            height=height,
            width=width,
            num_frames=num_frames,
            num_cond_frames=1, 
            dtype=torch.float32,
            device=device,
            generator=generator,
            latents=latents,
        )
        if context_parallel_util.get_cp_size() > 1:
            context_parallel_util.cp_broadcast(latents)

        # 6. Prepare ref_target_masks to latent size
        if ref_target_masks is not None:
            ref_target_masks = self._resize_and_centercrop_tensor(ref_target_masks, height, width, resize_mode)

        # 7. Denoising loop
        if context_parallel_util.get_cp_size() > 1:
            torch.distributed.barrier(group=context_parallel_util.get_cp_group())


        with tqdm(total=len(timesteps), desc="Denoising") as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue
                if self.dit.lora_mix_config is not None:
                    self.dit.set_lora_mix_step(i)

                self._current_timestep = t

                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                latent_model_input = latent_model_input.to(dit_dtype)

                timestep = t.expand(latent_model_input.shape[0]).to(dit_dtype)
                timestep = timestep.unsqueeze(-1).repeat(1, latent_model_input.shape[2])
                timestep[:, :1] = 0

                if self.do_classifier_free_guidance and ref_target_masks is not None:
                    # Multitalk mode with CFG
                    noise_pred_uncond_text = self.dit(
                            hidden_states=latent_model_input[:1],
                            timestep=timestep[:1],
                            encoder_hidden_states=prompt_embeds[:1],
                            encoder_attention_mask=prompt_attention_mask[:1],
                            num_cond_latents=1,
                            audio_embs=audio_cond_embs[:audio_num],
                            ref_target_masks=ref_target_masks
                        )
                    noise_pred_cond = self.dit(
                            hidden_states=latent_model_input[1:],
                            timestep=timestep[1:],
                            encoder_hidden_states=prompt_embeds[1:],
                            encoder_attention_mask=prompt_attention_mask[1:],
                            num_cond_latents=1,
                            audio_embs=audio_cond_embs[audio_num:2*audio_num],
                            ref_target_masks=ref_target_masks
                        )
                    noise_pred = torch.cat([noise_pred_uncond_text, noise_pred_cond])
                else:
                    # Singletalk mode
                    # Multitalk mode without CFG (e.g. use_distill with scale=1.0)
                    noise_pred = self.dit(
                        hidden_states=latent_model_input,
                        timestep=timestep,
                        encoder_hidden_states=prompt_embeds,
                        encoder_attention_mask=prompt_attention_mask,
                        num_cond_latents=1,
                        audio_embs=audio_cond_embs,
                        ref_target_masks=ref_target_masks
                    )

                if self.do_classifier_free_guidance:
                    timestep_uncond = t.expand(latents.shape[0]).to(dit_dtype)
                    timestep_uncond = timestep_uncond.unsqueeze(-1).repeat(1, latent_model_input.shape[2])
                    timestep_uncond[:, :1] = 0

                    noise_pred_uncond = self.dit(
                        hidden_states=latents,
                        timestep=timestep_uncond,
                        encoder_hidden_states=negative_prompt_embeds,
                        encoder_attention_mask=negative_prompt_attention_mask,
                        num_cond_latents=1,
                        audio_embs=audio_unond_embs,
                        ref_target_masks=ref_target_masks
                    )

                    noise_pred_uncond_text, noise_pred_cond = noise_pred.chunk(2)
                    
                    noise_pred = noise_pred_uncond + text_guidance_scale * (noise_pred_cond - noise_pred_uncond_text) + audio_guidance_scale * (noise_pred_uncond_text - noise_pred_uncond)

                # negate for scheduler compatibility
                noise_pred = -noise_pred

                # compute the previous noisy sample x_t -> x_t-1
                latents[:, :, 1:] = self.scheduler.step(noise_pred[:, :, 1:], t, latents[:, :, 1:], return_dict=False)[0]

                # call the callback, if provided
                if i == len(timesteps) - 1 or (i + 1) % self.scheduler.order == 0:
                    progress_bar.update()


        self._current_timestep = None

        if output_type == 'latent':
            return latents
        
        if output_type == 'both':
            latents_ = latents.clone()

        latents = latents.to(self.vae.dtype)
        latents = self.denormalize_latents(latents)
        output_video = self._decode_video_latents(latents)
        output_video = self.video_processor.postprocess_video(output_video)

        if output_type == 'both':
            return (output_video, latents_)
        else: 
            return output_video


    @torch.no_grad()
    def generate_avc(
        self,
        video: List[Image.Image],
        video_latent: torch.Tensor,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        height: int = 480,
        width: int = 832,
        num_frames: int = 93,
        num_cond_frames: int = 13,
        num_inference_steps: int = 50,
        use_distill: bool = False,
        text_guidance_scale: float = 4.0,
        audio_guidance_scale: float = 4.0,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "np",
        attention_kwargs: Optional[Dict[str, Any]] = None,
        max_sequence_length: int = 512,
        use_kv_cache=True,
        offload_kv_cache=False,
        enhance_hf=True,
        # avatar related params
        audio_emb: torch.Tensor = None,
        ref_latent: torch.Tensor = None,
        ref_img_index: int = None,
        mask_frame_range: int = None,
        ref_target_masks: torch.Tensor = None,
        resize_mode: Optional[str] = "crop", # "default" / "crop"
    ):
        r"""
        Generates video frames from a source video and text prompt using diffusion process with spatio-temporal conditioning.

        Args:
            video (`List[Image.Image]`):
                Input video frames for conditioning.
            prompt (`str or List[str]`, *optional*):
                Text prompt(s) for video content generation.
            negative_prompt (`str or List[str]`, *optional*):
                Negative prompt(s) for content exclusion. If not provided, uses empty string.
            num_frames (`int`, *optional*, defaults to 93):
                Number of frames to generate for the video. Should satisfy (num_frames - 1) % vae_scale_factor_temporal == 0.
            num_cond_frames (`int`, *optional*, defaults to 13):
                Number of conditioning frames from the input video.
            num_inference_steps (`int`, *optional*, defaults to 50):
                Number of diffusion sampling steps. Higher values improve quality but slow generation.
            use_distill (`bool`, *optional*, defaults to False):
                Whether to use distillation sampling schedule.
            text_guidance_scale (`float`, *optional*, defaults to 4.0):
                Classifier-free guidance scale. Controls prompt adherence vs. creativity.
            audio_guidance_scale (`float`, *optional*, defaults to 4.0):
                Classifier-free guidance scale. Controls audio adherence. Larger values may lead to exaggerated mouth.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                Number of videos to generate per prompt.
            generator (`torch.Generator or List[torch.Generator]`, *optional*):
                Random seed generator(s) for noise generation.
            latents (`torch.Tensor`, *optional*):
                Precomputed latent tensor. If not provided, random latents are generated.
            output_type (`str`, *optional*, defaults to "np"):
                Output format type. "np" for numpy array, "latent" for latent tensor.
            attention_kwargs (`Dict[str, Any]`, *optional*):
                Additional attention parameters for the model.
            max_sequence_length (`int`, *optional*, defaults to 512):
                Maximum sequence length for text encoding.
            use_kv_cache (`bool`, *optional*, defaults to True):
                Whether to use key-value cache for faster inference.
            offload_kv_cache (`bool`, *optional*, defaults to False):
                Whether to offload key-value cache to CPU to save VRAM.
            enhance_hf (`bool`, *optional*, defaults to True):
                Whether to use enhanced high-frequency denoising schedule.
            audio_emb (`torch.Tensor`):
                Audio embedding to driven the lip movements and body motions of character.
            ref_latent (`torch.Tensor`):
                The latent of reference anchor image when generate long video.
            ref_img_index (`int`, *optional*, defaults to 10)
                The insertion position of the reference image relative to the noisy latent along the temporal dimension.
            mask_frame_range (`int`, *optional*, defaults to 0)
                The attention masking range for the reference image.
            ref_target_masks(`torch.Tensor`, *optional*, defaults to None):
                Mask used in dual-speaker audio-driven mode.
            resize_mode(`str`, *optional*):
                Output format type. "default" for resize, "crop" for shorter-length resize and centercrop.

        Returns:
            np.ndarray or torch.Tensor:
                Generated video frames. If output_type is "np", returns numpy array of shape (B, N, H, W, C).
                If output_type is "latent", returns latent tensor.
        """

        # 1. Check inputs. Raise error if not correct
        assert not (use_distill and enhance_hf), "use_distill and enhance_hf cannot both be True"
        scale_factor_spatial = self.vae_scale_factor_spatial * 2
        if self.dit.cp_split_hw is not None:
            scale_factor_spatial *= max(self.dit.cp_split_hw)
        
        self.check_inputs(
            prompt,
            negative_prompt,
            height,
            width,
            scale_factor_spatial
        )
        assert resize_mode in ['default', 'crop'], f"Unsupported resize_mode {resize_mode}, and you can choose from [default, crop]"
        
        if num_frames % self.vae_scale_factor_temporal != 1:
            loguru.logger.warning(
                f"`num_frames - 1` has to be divisible by {self.vae_scale_factor_temporal}. Rounding to the nearest number."
            )
            num_frames = num_frames // self.vae_scale_factor_temporal * self.vae_scale_factor_temporal + 1
        num_frames = max(num_frames, 1)

        self._text_guidance_scale = text_guidance_scale
        self._audio_guidance_scale = audio_guidance_scale
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        device = self.device

        # 2. Define call parameters
        if isinstance(prompt, str):
            batch_size = 1
        else:
            batch_size = len(prompt)

        # 3. Encode inputs
        dit_dtype = self.dit.dtype

        if context_parallel_util.get_cp_rank() == 0:
            (
                prompt_embeds, 
                prompt_attention_mask, 
                negative_prompt_embeds, 
                negative_prompt_attention_mask,
            ) = self.encode_prompt(
                prompt=prompt,
                negative_prompt=negative_prompt,
                do_classifier_free_guidance=self.do_classifier_free_guidance,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                dtype=dit_dtype,
                device=device,
            )
            if context_parallel_util.get_cp_size() > 1:
                context_parallel_util.cp_broadcast(prompt_embeds)
                context_parallel_util.cp_broadcast(prompt_attention_mask)
                if self.do_classifier_free_guidance:
                    context_parallel_util.cp_broadcast(negative_prompt_embeds)
                    context_parallel_util.cp_broadcast(negative_prompt_attention_mask)
        elif context_parallel_util.get_cp_size() > 1:
            caption_channels = self.text_encoder.config.d_model
            prompt_embeds = torch.zeros([batch_size, 1, max_sequence_length, caption_channels], dtype=dit_dtype, device=device)
            prompt_attention_mask = torch.zeros([batch_size, max_sequence_length], dtype=torch.int64, device=device)
            context_parallel_util.cp_broadcast(prompt_embeds)
            context_parallel_util.cp_broadcast(prompt_attention_mask)
            if self.do_classifier_free_guidance:
                negative_prompt_embeds = torch.zeros([batch_size, 1, max_sequence_length, caption_channels], dtype=dit_dtype, device=device)
                negative_prompt_attention_mask = torch.zeros([batch_size, max_sequence_length], dtype=torch.int64, device=device)
                context_parallel_util.cp_broadcast(negative_prompt_embeds)
                context_parallel_util.cp_broadcast(negative_prompt_attention_mask)

        audio_cond_embs = torch.cat([audio_emb] * num_videos_per_prompt, dim=0)
        audio_num = audio_cond_embs.shape[0]
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            prompt_attention_mask = torch.cat([negative_prompt_attention_mask, prompt_attention_mask], dim=0)
            audio_unond_embs = torch.zeros_like(audio_cond_embs)
            audio_cond_embs = torch.cat([audio_cond_embs, audio_cond_embs], dim=0)

        # 4. Prepare timesteps
        sigmas = self.get_timesteps_sigmas(num_inference_steps, use_distill=use_distill)
        self.scheduler.set_timesteps(num_inference_steps, sigmas=sigmas, device=device)
        timesteps = self.scheduler.timesteps

        if enhance_hf:
            tail_uniform_start = 500
            tail_uniform_end = 0
            num_tail_uniform_steps = 10
            timesteps_uniform_tail = list(np.linspace(tail_uniform_start, tail_uniform_end, num_tail_uniform_steps, dtype=np.float32, endpoint=(tail_uniform_end != 0)))
            timesteps_uniform_tail = [torch.tensor(t, device=device).unsqueeze(0) for t in timesteps_uniform_tail]
            filtered_timesteps = [timestep.unsqueeze(0) for timestep in timesteps if timestep > tail_uniform_start]
            timesteps = torch.cat(filtered_timesteps + timesteps_uniform_tail)
            self.scheduler.timesteps = timesteps
            self.scheduler.sigmas = torch.cat([timesteps / 1000, torch.zeros(1, device=timesteps.device)])

        # 5. Prepare latent variables
        video = self.video_processor.preprocess_video(self.video_processor, video, height=height, width=width, resize_mode=resize_mode)
        video = video.to(device=device, dtype=prompt_embeds.dtype) 
        cond_videos = video[:, :, -num_cond_frames:]
        cond_videos_latents = retrieve_latents(self.vae.encode(cond_videos), generator, sample_mode="argmax")
        cond_videos_latents = self.normalize_latents(cond_videos_latents)


        num_channels_latents = self.dit.config.in_channels
        latents = self.prepare_latents(
            video=video_latent,
            batch_size=batch_size * num_videos_per_prompt,
            num_channels_latents=num_channels_latents,
            height=height,
            width=width,
            num_frames=num_frames,
            num_cond_frames=num_cond_frames,
            dtype=dit_dtype,
            device=device,
            generator=generator,
            latents=latents,
            need_encode=False
        )

        num_cond_latents = 1 + (num_cond_frames - 1) // self.vae_scale_factor_temporal
        if not self.use_vcond:
            latents[:, :, :num_cond_latents] = cond_videos_latents

        if context_parallel_util.get_cp_size() > 1:
            context_parallel_util.cp_broadcast(latents)
        
        # 6. Prepare ref_target_masks from source size to latent size
        if ref_target_masks is not None:
            ref_target_masks = self._resize_and_centercrop_tensor(ref_target_masks, height, width, resize_mode)

        # 7. Add reference image
        if ref_latent is not None:
            num_cond_latents += 1
            num_ref_latents = 1
            latents = torch.cat([ref_latent, latents], dim=2)

        # 8. Denoising loop
        if context_parallel_util.get_cp_size() > 1:
            torch.distributed.barrier(group=context_parallel_util.get_cp_group())


        if use_kv_cache:
            cond_latents = latents[:, :, :num_cond_latents]
            self._cache_clean_latents(cond_latents, max_sequence_length, offload_kv_cache=offload_kv_cache, device=self.device, dtype=dit_dtype, \
                audio_embs=audio_emb, num_cond_latents=num_cond_latents, num_ref_latents=num_ref_latents, ref_img_index=ref_img_index)
            kv_cache_dict = self._get_kv_cache_dict()
            latents = latents[:, :, num_cond_latents:]
        else:
            kv_cache_dict = {}

        with tqdm(total=len(timesteps), desc="Denoising") as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue
                if self.dit.lora_mix_config is not None:
                    self.dit.set_lora_mix_step(i)

                self._current_timestep = t

                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                latent_model_input = latent_model_input.to(dit_dtype)

                timestep = t.expand(latent_model_input.shape[0]).to(dit_dtype)
                timestep = timestep.unsqueeze(-1).repeat(1, latent_model_input.shape[2])
                if not use_kv_cache:
                    timestep[:, :num_cond_latents] = 0
                
                if self.do_classifier_free_guidance and ref_target_masks is not None:
                    # Multitalk mode with CFG
                    noise_pred_uncond_text = self.dit(
                        hidden_states=latent_model_input[:1],
                        timestep=timestep[:1],
                        encoder_hidden_states=prompt_embeds[:1],
                        encoder_attention_mask=prompt_attention_mask[:1],
                        num_cond_latents=num_cond_latents, 
                        kv_cache_dict=kv_cache_dict,
                        audio_embs=audio_cond_embs[:audio_num], 
                        num_ref_latents=num_ref_latents, 
                        ref_img_index=ref_img_index,
                        mask_frame_range=mask_frame_range,
                        ref_target_masks=ref_target_masks,
                    )
                    noise_pred_cond = self.dit(
                        hidden_states=latent_model_input[1:], 
                        timestep=timestep[1:],
                        encoder_hidden_states=prompt_embeds[1:], 
                        encoder_attention_mask=prompt_attention_mask[1:],
                        num_cond_latents=num_cond_latents,
                        kv_cache_dict=kv_cache_dict,
                        audio_embs=audio_cond_embs[audio_num:2*audio_num], 
                        num_ref_latents=num_ref_latents, 
                        ref_img_index=ref_img_index,
                        mask_frame_range=mask_frame_range,
                        ref_target_masks=ref_target_masks,
                    )
                    noise_pred = torch.cat([noise_pred_uncond_text, noise_pred_cond])
                else:
                    # Singletalk mode
                    # Multitalk mode without CFG (e.g. use_distill with scale=1.0)
                    noise_pred = self.dit(
                        hidden_states=latent_model_input, 
                        timestep=timestep,
                        encoder_hidden_states=prompt_embeds,
                        encoder_attention_mask=prompt_attention_mask, 
                        num_cond_latents=num_cond_latents, 
                        kv_cache_dict=kv_cache_dict,
                        audio_embs=audio_cond_embs, 
                        num_ref_latents=num_ref_latents, 
                        ref_img_index=ref_img_index,
                        mask_frame_range=mask_frame_range,
                        ref_target_masks=ref_target_masks
                    )

                if self.do_classifier_free_guidance:
                    timestep_uncond = t.expand(latents.shape[0]).to(dit_dtype)
                    timestep_uncond = timestep_uncond.unsqueeze(-1).repeat(1, latent_model_input.shape[2])
                    if not use_kv_cache:
                        timestep_uncond[:, :num_cond_latents] = 0

                    noise_pred_uncond = self.dit(
                        hidden_states=latents,
                        timestep=timestep_uncond,
                        encoder_hidden_states=negative_prompt_embeds,
                        encoder_attention_mask=negative_prompt_attention_mask,
                        num_cond_latents=num_cond_latents,
                        kv_cache_dict=kv_cache_dict,
                        audio_embs=audio_unond_embs,
                        num_ref_latents=num_ref_latents, 
                        ref_img_index=ref_img_index,
                        mask_frame_range=mask_frame_range,
                        ref_target_masks=ref_target_masks
                    )

                    noise_pred_uncond_text, noise_pred_cond = noise_pred.chunk(2)
                    
                    noise_pred = noise_pred_uncond + text_guidance_scale * (noise_pred_cond - noise_pred_uncond_text) + audio_guidance_scale * (noise_pred_uncond_text - noise_pred_uncond)
                
                # negate for scheduler compatibility
                noise_pred = -noise_pred

                # compute the previous noisy sample x_t -> x_t-1
                if use_kv_cache:
                    latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                else:
                    latents[:, :, num_cond_latents:] = self.scheduler.step(noise_pred[:, :, num_cond_latents:], t, latents[:, :, num_cond_latents:], return_dict=False)[0]

                # call the callback, if provided
                if i == len(timesteps) - 1 or (i + 1) % self.scheduler.order == 0:
                    progress_bar.update()
            
            if use_kv_cache:
                latents = torch.cat([cond_latents, latents], dim=2)
            
            if ref_latent is not None:
                latents = latents[:, :, num_ref_latents:]
                num_cond_latents -= 1

            latents[:, :, :num_cond_latents] = cond_videos_latents


        self._current_timestep = None

        if output_type == 'latent':
            return latents
        
        if output_type == 'both':
            latents_ = latents.clone()

        latents = latents.to(self.vae.dtype)
        latents = self.denormalize_latents(latents)
        output_video = self._decode_video_latents(latents)
        output_video = self.video_processor.postprocess_video(output_video)

        if output_type == 'both':
            return (output_video, latents_)
        else: 
            return output_video
    

    

    def to(self, device: str | torch.device):
        """
        Move pipeline to specified device.

        Args:
            device: Target device string

        Returns:
            Self
        """
        self.device = device
        if self.dit is not None:
            self.dit = self.dit.to(device, non_blocking=True)
            if hasattr(self.dit, 'lora_dict') and self.dit.lora_dict:
                for lora_key, lora_network in self.dit.lora_dict.items():
                    for lora in lora_network.loras:
                        lora.to(device, non_blocking=True)
        if self.text_encoder is not None:
            self.text_encoder = self.text_encoder.to(device, non_blocking=True)
        if self.vae is not None:
            self.vae = self.vae.to(device, non_blocking=True)
        return self
