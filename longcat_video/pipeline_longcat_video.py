from typing import Any, Dict, List, Optional, Union, Literal

import gc
import math
import torch
import loguru
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm 
from PIL import Image
from diffusers.video_processor import VideoProcessor
from diffusers.image_processor import PipelineImageInput
from transformers import AutoTokenizer, UMT5EncoderModel

from longcat_video.modules.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from longcat_video.modules.autoencoder_kl_wan import AutoencoderKLWan
from longcat_video.modules.longcat_video_dit import LongCatVideoTransformer3DModel
from longcat_video.context_parallel import context_parallel_util
from longcat_video.utils.bukcet_config import get_bucket_config

import ftfy
import regex as re
import html


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

class LongCatVideoPipeline:
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
        dit: LongCatVideoTransformer3DModel,
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

        self._num_timesteps = 1000
        self._num_distill_sample_steps = 50

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
            
        return prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask

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
                latent = retrieve_latents(self.vae.encode(encoded_input), gen)
                cond_latents.append(latent)

            cond_latents = torch.cat(cond_latents, dim=0).to(dtype)
            cond_latents = self.normalize_latents(cond_latents)

            num_cond_latents = 1 + (num_cond_frames - 1) // self.vae_scale_factor_temporal
            latents[:, :, :num_cond_latents] = cond_latents

        return latents

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1.0

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

    def _cache_clean_latents(self, cond_latents, model_max_length, offload_kv_cache, device, dtype):
        timestep = torch.zeros(cond_latents.shape[0], cond_latents.shape[2]).to(device=device, dtype=dtype)
        # make null prompt tensor(skip_crs_attn=True, so tensors below will not be actually used)
        empty_embeds = torch.zeros([cond_latents.shape[0], 1, model_max_length, self.text_encoder.config.d_model], device=device, dtype=dtype)
        _, kv_cache_dict = self.dit(
            hidden_states=cond_latents, 
            timestep=timestep, 
            encoder_hidden_states=empty_embeds,
            return_kv=True, 
            skip_crs_attn=True, 
            offload_kv_cache=offload_kv_cache
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

    @torch.no_grad()
    def generate_t2v(
        self,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        height: int = 480,
        width: int = 832,
        num_frames: int = 93,
        num_inference_steps: int = 50,
        use_distill: bool = False,
        guidance_scale: float = 4.0,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "np",
        attention_kwargs: Optional[Dict[str, Any]] = None,
        max_sequence_length: int = 512,
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
            guidance_scale (`float`, *optional*, defaults to 4.0):
                Classifier-free guidance scale. Controls prompt adherence vs. creativity.
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
            scale_factor_spatial,
        )

        if num_frames % self.vae_scale_factor_temporal != 1:
            loguru.logger.warning(
                f"`num_frames - 1` has to be divisible by {self.vae_scale_factor_temporal}. Rounding to the nearest number."
            )
            num_frames = num_frames // self.vae_scale_factor_temporal * self.vae_scale_factor_temporal + 1
        num_frames = max(num_frames, 1)

        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        device = self.device

        # 2. Define call parameters
        if isinstance(prompt, str):
            batch_size = 1
        else:
            batch_size = len(prompt)

        # 3. Encode input prompt
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
        
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            prompt_attention_mask = torch.cat([negative_prompt_attention_mask, prompt_attention_mask], dim=0)

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

                self._current_timestep = t

                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                latent_model_input = latent_model_input.to(dit_dtype)

                timestep = t.expand(latent_model_input.shape[0]).to(dit_dtype)

                noise_pred = self.dit(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    encoder_attention_mask=prompt_attention_mask,
                )

                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)

                    B = noise_pred_cond.shape[0]
                    positive = noise_pred_cond.reshape(B, -1)
                    negative = noise_pred_uncond.reshape(B, -1)
                    # Calculate the optimized scale
                    st_star = self.optimized_scale(positive, negative)
                    # Reshape for broadcasting
                    st_star = st_star.view(B, 1, 1, 1)
                    # print(f'step i: {i} --> scale: {st_star}')

                    noise_pred = noise_pred_uncond * st_star + guidance_scale * (noise_pred_cond - noise_pred_uncond * st_star) 

                # negate for scheduler compatibility
                noise_pred = -noise_pred

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                # call the callback, if provided
                if i == len(timesteps) - 1 or (i + 1) % self.scheduler.order == 0:
                    progress_bar.update()

        self._current_timestep = None

        if not output_type == "latent":
            latents = latents.to(self.vae.dtype)
            latents = self.denormalize_latents(latents)
            output_video = self.vae.decode(latents, return_dict=False)[0]
            output_video = self.video_processor.postprocess_video(output_video, output_type=output_type)
        else:
            output_video = latents

        return output_video 
    

    @torch.no_grad()
    def generate_i2v(
        self,
        image: PipelineImageInput,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        resolution: Literal["480p", "720p"] = "480p",
        num_frames: int = 93,
        num_inference_steps: int = 50,
        use_distill: bool = False,
        guidance_scale: float = 4.0,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "np",
        attention_kwargs: Optional[Dict[str, Any]] = None,
        max_sequence_length: int = 512,
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
            guidance_scale (`float`, *optional*, defaults to 4.0):
                Classifier-free guidance scale. Controls prompt adherence vs. creativity.
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

        if num_frames % self.vae_scale_factor_temporal != 1:
            loguru.logger.warning(
                f"`num_frames - 1` has to be divisible by {self.vae_scale_factor_temporal}. Rounding to the nearest number."
            )
            num_frames = num_frames // self.vae_scale_factor_temporal * self.vae_scale_factor_temporal + 1
        num_frames = max(num_frames, 1)

        has_neg_prompt = negative_prompt is not None 
        do_true_cfg = guidance_scale > 1 and has_neg_prompt

        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        device = self.device

        # 2. Define call parameters
        if isinstance(prompt, str):
            batch_size = 1
        else:
            batch_size = len(prompt)


        # 3. Encode input prompt
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

        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            prompt_attention_mask = torch.cat([negative_prompt_attention_mask, prompt_attention_mask], dim=0)
        
        # 4. Prepare timesteps
        sigmas = self.get_timesteps_sigmas(num_inference_steps, use_distill=use_distill)
        self.scheduler.set_timesteps(num_inference_steps, sigmas=sigmas, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        image = self.video_processor.preprocess(image, height=height, width=width)
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

        # 6. Denoising loop
        if context_parallel_util.get_cp_size() > 1:
            torch.distributed.barrier(group=context_parallel_util.get_cp_group())

        with tqdm(total=len(timesteps), desc="Denoising") as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t

                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                latent_model_input = latent_model_input.to(dit_dtype)

                timestep = t.expand(latent_model_input.shape[0]).to(dit_dtype)
                timestep = timestep.unsqueeze(-1).repeat(1, latent_model_input.shape[2])
                timestep[:, :1] = 0

                noise_pred = self.dit(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    encoder_attention_mask=prompt_attention_mask,
                    num_cond_latents=1,
                )

                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                    
                    B = noise_pred_cond.shape[0]
                    positive = noise_pred_cond.reshape(B, -1)
                    negative = noise_pred_uncond.reshape(B, -1)
                    # Calculate the optimized scale
                    st_star = self.optimized_scale(positive, negative)
                    # Reshape for broadcasting
                    st_star = st_star.view(B, 1, 1, 1)
                    # print(f'step i: {i} --> scale: {st_star}')

                    noise_pred = noise_pred_uncond * st_star + guidance_scale * (noise_pred_cond - noise_pred_uncond * st_star)

                # negate for scheduler compatibility
                noise_pred = -noise_pred

                # compute the previous noisy sample x_t -> x_t-1
                latents[:, :, 1:] = self.scheduler.step(noise_pred[:, :, 1:], t, latents[:, :, 1:], return_dict=False)[0]

                # call the callback, if provided
                if i == len(timesteps) - 1 or (i + 1) % self.scheduler.order == 0:
                    progress_bar.update()

        self._current_timestep = None

        if not output_type == "latent":
            latents = latents.to(self.vae.dtype)
            latents = self.denormalize_latents(latents)
            output_video = self.vae.decode(latents, return_dict=False)[0]
            output_video = self.video_processor.postprocess_video(output_video, output_type=output_type)
        else:
            output_video = latents

        return output_video 


    @torch.no_grad()
    def generate_vc(
        self,
        video: List[Image.Image],
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        resolution: Literal["480p", "720p"] = "480p",
        num_frames: int = 93,
        num_cond_frames: int = 13,
        num_inference_steps: int = 50,
        use_distill: bool = False,
        guidance_scale: float = 4.0,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "np",
        attention_kwargs: Optional[Dict[str, Any]] = None,
        max_sequence_length: int = 512,
        use_kv_cache=True,
        offload_kv_cache=False,
        enhance_hf=True,
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
            resolution (`Literal["480p", "720p"]`, *optional*, defaults to "480p"):
                Target video resolution. Determines output frame size.
            num_frames (`int`, *optional*, defaults to 93):
                Number of frames to generate for the video. Should satisfy (num_frames - 1) % vae_scale_factor_temporal == 0.
            num_cond_frames (`int`, *optional*, defaults to 13):
                Number of conditioning frames from the input video.
            num_inference_steps (`int`, *optional*, defaults to 50):
                Number of diffusion sampling steps. Higher values improve quality but slow generation.
            use_distill (`bool`, *optional*, defaults to False):
                Whether to use distillation sampling schedule.
            guidance_scale (`float`, *optional*, defaults to 4.0):
                Classifier-free guidance scale. Controls prompt adherence vs. creativity.
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
        height, width = self.get_condition_shape(video, resolution, scale_factor_spatial=scale_factor_spatial)
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

        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        device = self.device

        # 2. Define call parameters
        if isinstance(prompt, str):
            batch_size = 1
        else:
            batch_size = len(prompt)

        # 3. Encode input prompt
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

        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            prompt_attention_mask = torch.cat([negative_prompt_attention_mask, prompt_attention_mask], dim=0)

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
        video = self.video_processor.preprocess_video(video, height=height, width=width)
        video = video.to(device=device, dtype=prompt_embeds.dtype)

        num_channels_latents = self.dit.config.in_channels
        latents = self.prepare_latents(
            video=video,
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
        )
        if context_parallel_util.get_cp_size() > 1:
            context_parallel_util.cp_broadcast(latents)

        num_cond_latents = 1 + (num_cond_frames - 1) // self.vae_scale_factor_temporal
        
        # 6. Denoising loop
        if context_parallel_util.get_cp_size() > 1:
            torch.distributed.barrier(group=context_parallel_util.get_cp_group())

        if use_kv_cache:
            cond_latents = latents[:, :, :num_cond_latents]
            self._cache_clean_latents(cond_latents, max_sequence_length, offload_kv_cache=offload_kv_cache, device=self.device, dtype=dit_dtype)
            kv_cache_dict = self._get_kv_cache_dict()
            latents = latents[:, :, num_cond_latents:]
        else:
            kv_cache_dict = {}

        with tqdm(total=len(timesteps), desc="Denoising") as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t

                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                latent_model_input = latent_model_input.to(dit_dtype)

                timestep = t.expand(latent_model_input.shape[0]).to(dit_dtype)
                timestep = timestep.unsqueeze(-1).repeat(1, latent_model_input.shape[2])
                if not use_kv_cache:
                    timestep[:, :num_cond_latents] = 0

                noise_pred = self.dit(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    encoder_attention_mask=prompt_attention_mask,
                    num_cond_latents=num_cond_latents,
                    kv_cache_dict=kv_cache_dict
                )

                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                    
                    B = noise_pred_cond.shape[0]
                    positive = noise_pred_cond.reshape(B, -1)
                    negative = noise_pred_uncond.reshape(B, -1)
                    # Calculate the optimized scale
                    st_star = self.optimized_scale(positive, negative)
                    # Reshape for broadcasting
                    st_star = st_star.view(B, 1, 1, 1)
                    # print(f'step i: {i} --> scale: {st_star}')

                    noise_pred = noise_pred_uncond * st_star + guidance_scale * (noise_pred_cond - noise_pred_uncond * st_star)
                
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

        self._current_timestep = None

        if not output_type == "latent":
            latents = latents.to(self.vae.dtype)
            latents = self.denormalize_latents(latents)
            output_video = self.vae.decode(latents, return_dict=False)[0]
            output_video = self.video_processor.postprocess_video(output_video, output_type=output_type)
        else:
            output_video = latents
        
        return output_video
    

    @torch.no_grad()
    def generate_refine(
        self,
        image: Optional[PipelineImageInput] = None,
        video: Optional[List[Image.Image]] = None,
        prompt: Union[str, List[str]] = None,
        stage1_video: Optional[str] = None,
        num_cond_frames: int = 0,
        num_inference_steps: int = 50,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "np",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        max_sequence_length: int = 512,
        t_thresh = 0.5,
        spatial_refine_only = False,
    ):
        r"""
        Generates super-resolution video frames from a low-resolution input video, image, and text prompt using diffusion process.

        Args:
            image (`PipelineImageInput`, *optional*):
                Input image for conditioning. Cannot be provided together with `video`.
            video (`List[Image.Image]`, *optional*):
                Input video frames for conditioning. Cannot be provided together with `image`.
            prompt (`str or List[str]`, *optional*):
                Text prompt(s) for video content generation.
            stage1_video (`str or np.ndarray`):
                Low-resolution input video for super-resolution generation.
            num_cond_frames (`int`, *optional*, defaults to 0):
                Number of conditioning frames from the input video or image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                Number of diffusion sampling steps. Higher values improve quality but slow generation.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                Number of videos to generate per prompt.
            generator (`torch.Generator or List[torch.Generator]`, *optional*):
                Random seed generator(s) for noise generation.
            latents (`torch.Tensor`, *optional*):
                Precomputed latent tensor. If not provided, random latents are generated.
            output_type (`str`, *optional*, defaults to "np"):
                Output format type. "np" for numpy array, "latent" for latent tensor.
            return_dict (`bool`, *optional*, defaults to True):
                Whether to return output as a dictionary.
            attention_kwargs (`Dict[str, Any]`, *optional*):
                Additional attention parameters for the model.
            max_sequence_length (`int`, *optional*, defaults to 512):
                Maximum sequence length for text encoding.
            t_thresh (`float`, *optional*, defaults to 0.5):
                Threshold for timestep scheduling in the denoising process.
            spatial_refine_only (`bool`, *optional*, defaults to False):
                If True, only perform spatial super-resolution (increase resolution, keep frame count unchanged).
                If False, perform both spatial and temporal super-resolution (increase resolution and double the frame count).

        Returns:
            np.ndarray or torch.Tensor:
                Generated super-resolution video frames. If output_type is "np", returns numpy array of shape (B, N, H, W, C).
                If output_type is "latent", returns latent tensor.
        """

        # 1. Check inputs. Raise error if not correct
        if (image is not None) and (video is not None):
            raise ValueError("Cannot provide both `image and video` at the same time. Please provide only one.")
        
        scale_factor_spatial = self.vae_scale_factor_spatial * 2 * 4
        if self.dit.cp_split_hw is not None:
            scale_factor_spatial *= max(self.dit.cp_split_hw)
        height, width = self.get_condition_shape(stage1_video, "720p", scale_factor_spatial=scale_factor_spatial)
        self.check_inputs(
            prompt,
            None,
            height,
            width,
            scale_factor_spatial
        )

        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        device = self.device

        # 2. Define call parameters
        if isinstance(prompt, str):
            batch_size = 1
        else:
            batch_size = len(prompt)

        # 3. Encode input prompt
        dit_dtype = self.dit.dtype

        if context_parallel_util.get_cp_rank() == 0:
            (
                prompt_embeds, 
                prompt_attention_mask, 
                _, 
                _,
            ) = self.encode_prompt(
                prompt=prompt,
                do_classifier_free_guidance=False,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                dtype=dit_dtype,
                device=device,
            )
            if context_parallel_util.get_cp_size() > 1:
                context_parallel_util.cp_broadcast(prompt_embeds)
                context_parallel_util.cp_broadcast(prompt_attention_mask)
        elif context_parallel_util.get_cp_size() > 1:
            caption_channels = self.text_encoder.config.d_model
            prompt_embeds = torch.zeros([batch_size, 1, max_sequence_length, caption_channels], dtype=dit_dtype, device=device)
            prompt_attention_mask = torch.zeros([batch_size, max_sequence_length], dtype=torch.int64, device=device)
            context_parallel_util.cp_broadcast(prompt_embeds)
            context_parallel_util.cp_broadcast(prompt_attention_mask)
        
        # 4. Prepare timesteps
        sigmas = self.get_timesteps_sigmas(num_inference_steps)
        self.scheduler.set_timesteps(num_inference_steps, sigmas=sigmas, device=device)
        timesteps = self.scheduler.timesteps

        if t_thresh:
            t_thresh_tensor = torch.tensor(t_thresh * 1000, dtype=timesteps.dtype, device=timesteps.device)
            timesteps = torch.cat([t_thresh_tensor.unsqueeze(0), timesteps[timesteps < t_thresh_tensor]])
            self.scheduler.timesteps = timesteps
            self.scheduler.sigmas = torch.cat([timesteps / 1000, torch.zeros(1, device=timesteps.device)])
        
        # 5. Prepare latent variables
        num_frame = len(stage1_video)
        new_frame_size = num_frame if spatial_refine_only else 2 * num_frame
        stage1_video = torch.from_numpy(np.array(stage1_video)).permute(0, 3, 1, 2)
        stage1_video = stage1_video.to(device=device, dtype=prompt_embeds.dtype)

        video_DOWN = F.interpolate(stage1_video, size=(height, width), mode='bilinear', align_corners=True)
        video_DOWN = video_DOWN.permute(1, 0, 2, 3).unsqueeze(0) # [frame, C, H, W] -> [1, C, frame, H, W]
        video_DOWN = video_DOWN / 255.0

        video_UP = F.interpolate(video_DOWN, size=(new_frame_size, height, width), mode='trilinear', align_corners=True) # [B, C, frame, H, W]
        video_UP = video_UP * 2 - 1

        # do padding
        bsa_latent_granularity = 4
        num_noise_frames = video_UP.shape[2] - num_cond_frames

        num_cond_latents = 0
        num_cond_frames_added = 0
        if num_cond_frames > 0:
            num_cond_latents = 1 + math.ceil((num_cond_frames - 1) / self.vae_scale_factor_temporal)
            num_cond_latents = math.ceil(num_cond_latents / bsa_latent_granularity) * bsa_latent_granularity
            num_cond_frames_added = 1 + (num_cond_latents - 1) * self.vae_scale_factor_temporal - num_cond_frames
            num_cond_frames = num_cond_frames + num_cond_frames_added

        num_noise_latents = math.ceil(num_noise_frames / self.vae_scale_factor_temporal)
        num_noise_latents = math.ceil(num_noise_latents / bsa_latent_granularity) * bsa_latent_granularity
        num_noise_frames_added = num_noise_latents * self.vae_scale_factor_temporal - num_noise_frames
        
        pad_front = video_UP[:, :, 0:1].repeat(1, 1, num_cond_frames_added, 1, 1)
        pad_back = video_UP[:, :, -1:].repeat(1, 1, num_noise_frames_added, 1, 1)
        video_UP = torch.cat([pad_front, video_UP, pad_back], dim=2)
            
        latent_up = retrieve_latents(self.vae.encode(video_UP))
        latent_up = self.normalize_latents(latent_up)
        latent_up = (1 - t_thresh) * latent_up + t_thresh * torch.randn_like(latent_up).contiguous()
        del video_DOWN, video_UP, stage1_video
        torch_gc()

        num_channels_latents = self.dit.config.in_channels

        if image is not None:
            image = self.video_processor.preprocess(image, height=height, width=width)
            image = image.to(device=device, dtype=prompt_embeds.dtype)
        if video is not None:
            video = self.video_processor.preprocess_video(video, height=height, width=width)
            video = video.to(device=device, dtype=prompt_embeds.dtype)
        
        latents = self.prepare_latents(
            image=image,
            video=video,
            batch_size=batch_size * num_videos_per_prompt,
            num_channels_latents=num_channels_latents,
            height=height,
            width=width,
            num_cond_frames=num_cond_frames,
            dtype=torch.float32,
            device=device,
            generator=generator,
            latents=latent_up,
            num_cond_frames_added=num_cond_frames_added,
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

                self._current_timestep = t

                latent_model_input = latents.to(dit_dtype)

                timestep = t.expand(latent_model_input.shape[0]).to(dit_dtype)
                timestep = timestep.unsqueeze(-1).repeat(1, latent_model_input.shape[2])
                timestep[:, :num_cond_latents] = 0

                noise_pred_cond = self.dit(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    encoder_attention_mask=prompt_attention_mask,
                    num_cond_latents=num_cond_latents,
                )

                noise_pred = noise_pred_cond

                # negate for scheduler compatibility
                noise_pred = -noise_pred

                # compute the previous noisy sample x_t -> x_t-1
                latents[:, :, num_cond_latents:] = self.scheduler.step(noise_pred[:, :, num_cond_latents:], t, latents[:, :, num_cond_latents:], return_dict=False)[0]

                # call the callback, if provided
                if i == len(timesteps) - 1 or (i + 1) % self.scheduler.order == 0:
                    progress_bar.update()

        self._current_timestep = None

        if not output_type == "latent":
            latents = latents.to(self.vae.dtype)
            latents = self.denormalize_latents(latents)
            output_video = self.vae.decode(latents, return_dict=False)[0]
            output_video = self.video_processor.postprocess_video(output_video, output_type=output_type)
            output_video = output_video[:, num_cond_frames_added: new_frame_size+num_cond_frames_added]
        else:
            output_video = latents

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
    