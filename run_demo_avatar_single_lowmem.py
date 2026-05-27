"""
Low-memory Avatar 1.5 inference script for single-GPU (24GB VRAM, 32GB RAM).
Uses sequential model loading/offloading to avoid OOM.
"""
import os

# Persistent kernel cache — skip autotuning on subsequent runs
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", os.path.join(os.path.dirname(__file__), ".cache/torch_inductor"))
os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE", "1")
import json
import time
import math
import random
import argparse
import datetime
import PIL.Image
import numpy as np
from pathlib import Path

import torch
import torch.distributed as dist

from transformers import AutoTokenizer, UMT5EncoderModel
from diffusers.utils import load_image

from longcat_video.pipeline_longcat_video_avatar import LongCatVideoAvatarPipeline
from longcat_video.modules.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from longcat_video.modules.autoencoder_kl_wan import AutoencoderKLWan
from longcat_video.modules.avatar.longcat_video_dit_avatar import LongCatVideoAvatarTransformer3DModel
from longcat_video.modules.quantization import load_quantized_dit
from longcat_video.context_parallel import context_parallel_util

import librosa
from longcat_video.audio_process import get_audio_encoder, get_audio_feature_extractor
from longcat_video.audio_process.torch_utils import save_video_ffmpeg
from audio_separator.separator import Separator
import gc


def torch_gc():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


def generate_random_uid():
    timestamp_part = str(int(time.time()))[-6:]
    random_part = str(random.randint(100000, 999999))
    return timestamp_part + random_part


def extract_vocal_from_speech(source_path, target_path, vocal_separator, audio_output_dir_temp):
    outputs = vocal_separator.separate(source_path)
    if len(outputs) <= 0:
        print("Audio separate failed. Using raw audio.")
        return None
    default_vocal_path = audio_output_dir_temp / "vocals" / outputs[0]
    default_vocal_path = default_vocal_path.resolve().as_posix()
    cmd = f"mv '{default_vocal_path}' '{target_path}'"
    os.system(cmd)
    return target_path


def generate(args):
    checkpoint_dir = args.checkpoint_dir
    input_json = args.input_json
    stage_1 = args.stage_1
    resolution = args.resolution
    num_segments = max(1, args.num_segments)
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Fixed params for v1.5 distill
    num_inference_steps = args.num_steps_override if args.num_steps_override > 0 else 8
    text_guidance_scale = 1.0
    audio_guidance_scale = 1.0
    save_fps = 25
    audio_stride = 1
    num_frames = 93
    num_cond_frames = 13

    if resolution == '480p':
        height, width = 480, 832
    elif resolution == '720p':
        height, width = 768, 1280

    # Load input
    with open(input_json, 'r', encoding='utf-8') as f:
        input_data = json.load(f)
    prompt = input_data['prompt']
    negative_prompt = None  # Not used in distill mode
    raw_speech_path = input_data['cond_audio']['person1']

    # Distributed setup (single GPU)
    rank = int(os.environ.get('RANK', 0))
    num_gpus = torch.cuda.device_count()
    local_rank = rank % num_gpus
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=3600))
    global_rank = dist.get_rank()
    num_processes = dist.get_world_size()

    context_parallel_util.init_context_parallel(context_parallel_size=1, global_rank=global_rank, world_size=num_processes)
    cp_rank = context_parallel_util.get_cp_rank()
    cp_size = context_parallel_util.get_cp_size()
    cp_split_hw = context_parallel_util.get_optimal_split(cp_size)

    device = torch.device(f"cuda:{local_rank}")

    # ===== PHASE 1: Encode text (load text encoder, use it, free it) =====
    print("[PHASE 1] Loading tokenizer + text encoder...")
    base_model_dir = os.path.join(checkpoint_dir, '..', 'LongCat-Video')
    tokenizer = AutoTokenizer.from_pretrained(base_model_dir, subfolder="tokenizer", torch_dtype=torch.bfloat16)

    text_encoder = UMT5EncoderModel.from_pretrained(base_model_dir, subfolder="text_encoder", torch_dtype=torch.bfloat16)
    text_encoder.to(device)
    text_encoder.eval()

    print("[PHASE 1] Encoding prompt...")
    with torch.no_grad():
        text_inputs = tokenizer(
            [prompt],
            padding="max_length",
            max_length=512,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(device)
        mask = text_inputs.attention_mask.to(device)
        prompt_embeds = text_encoder(text_input_ids, mask).last_hidden_state
        prompt_embeds = prompt_embeds.to(dtype=torch.bfloat16)
        prompt_embeds = prompt_embeds.view(1, 1, prompt_embeds.shape[1], -1)

    # Cache embeddings on CPU, free text encoder
    prompt_embeds_cpu = prompt_embeds.cpu()
    mask_cpu = mask.cpu()
    del text_encoder, text_inputs, text_input_ids, prompt_embeds
    torch_gc()
    print("[PHASE 1] Text encoder freed.")

    # ===== PHASE 2: Process audio =====
    print("[PHASE 2] Loading audio models...")
    audio_model_path = os.path.join(checkpoint_dir, 'whisper-large-v3')
    audio_encoder = get_audio_encoder(audio_model_path, "avatar-v1.5").to(device)
    audio_feature_extractor = get_audio_feature_extractor(audio_model_path, "avatar-v1.5")

    vocal_separator_path = os.path.join(checkpoint_dir, 'vocal_separator/Kim_Vocal_2.onnx')
    audio_output_dir_temp = Path("./audio_temp_file")
    os.makedirs(audio_output_dir_temp, exist_ok=True)
    audio_separator_model_path = os.path.dirname(vocal_separator_path)
    audio_separator_model_name = os.path.basename(vocal_separator_path)
    vocal_separator = Separator(
        output_dir=audio_output_dir_temp / "vocals",
        output_single_stem="vocals",
        model_file_dir=audio_separator_model_path,
    )
    vocal_separator.load_model(audio_separator_model_name)

    # Extract vocal
    temp_vocal_path = extract_vocal_from_speech(
        raw_speech_path,
        f"/tmp/temp_speech_{generate_random_uid()}_{global_rank}_vocal.wav",
        vocal_separator,
        audio_output_dir_temp,
    )
    assert temp_vocal_path is not None and os.path.exists(temp_vocal_path), "No vocal detected"

    # Compute audio embedding
    generate_duration = num_frames / save_fps + (num_segments - 1) * (num_frames - num_cond_frames) / save_fps
    speech_array, sr = librosa.load(temp_vocal_path, sr=16000)
    source_duration = len(speech_array) / sr
    added_sample_nums = math.ceil((generate_duration - source_duration) * sr)
    if added_sample_nums > 0:
        speech_array = np.append(speech_array, [0.0] * added_sample_nums)

    print("[PHASE 2] Computing audio embeddings...")
    # Build a temporary pipeline to use the exact get_audio_embedding_whisper method
    temp_pipe = LongCatVideoAvatarPipeline(
        tokenizer=tokenizer, text_encoder=None, vae=None, scheduler=None, dit=None,
        audio_encoder=audio_encoder, audio_feature_extractor=audio_feature_extractor,
        model_type="avatar-v1.5",
    )
    full_audio_emb = temp_pipe.get_audio_embedding(speech_array, fps=save_fps * audio_stride, device=device, sample_rate=sr, model_type="avatar-v1.5")

    if torch.isnan(full_audio_emb).any():
        raise ValueError("Broken audio embedding with NaN values")

    full_audio_emb_cpu = full_audio_emb.cpu()
    del temp_pipe, audio_encoder, audio_feature_extractor, vocal_separator
    torch_gc()
    print("[PHASE 2] Audio processed and models freed.")

    if os.path.exists(temp_vocal_path):
        os.remove(temp_vocal_path)

    # ===== PHASE 3: Load VAE + DiT and run inference =====
    print("[PHASE 3] Loading VAE...")
    vae = AutoencoderKLWan.from_pretrained(base_model_dir, subfolder="vae", torch_dtype=torch.bfloat16)
    vae.to(device)

    print("[PHASE 3] Loading INT8 DiT (streaming shards to avoid OOM)...")
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(checkpoint_dir, subfolder="scheduler", torch_dtype=torch.bfloat16)

    # Custom streaming loader: load shards one at a time directly into model
    from longcat_video.modules.quantization import QuantizedLinear, DEFAULT_SKIP_PATTERNS
    from safetensors.torch import load_file
    import torch.nn as nn

    quantized_dir = os.path.join(checkpoint_dir, "base_model_int8")
    with open(os.path.join(quantized_dir, "config.json"), "r") as f:
        config = json.load(f)
    config.pop("_class_name", None)
    config.pop("architectures", None)
    config.pop("_diffusers_version", None)
    config.pop("model_max_length", None)
    config["cp_split_hw"] = cp_split_hw

    # Build model skeleton with meta tensors to avoid RAM allocation
    with torch.device("meta"):
        dit = LongCatVideoAvatarTransformer3DModel(**config)

    # Replace Linear with QuantizedLinear (still on meta)
    modules_to_replace = {}
    for name, module in dit.named_modules():
        if isinstance(module, nn.Linear):
            if not any(p in name for p in DEFAULT_SKIP_PATTERNS):
                modules_to_replace[name] = module
    for name, module in modules_to_replace.items():
        parts = name.split(".")
        parent = dit
        for part in parts[:-1]:
            parent = getattr(parent, part)
        ql = QuantizedLinear(module.in_features, module.out_features, bias=module.bias is not None)
        setattr(parent, parts[-1], ql)

    # Materialize all parameters/buffers on CPU (empty)
    dit = dit.to_empty(device="cpu")

    # Stream each shard directly into model
    with open(os.path.join(quantized_dir, "quantized_model.safetensors.index.json"), "r") as f:
        index = json.load(f)
    shard_files = sorted(set(index["weight_map"].values()))
    for shard_file in shard_files:
        print(f"  Loading shard: {shard_file}")
        shard_path = os.path.join(quantized_dir, shard_file)
        shard_dict = load_file(shard_path, device="cpu")
        # Assign directly into model parameters
        for key, tensor in shard_dict.items():
            parts = key.split(".")
            obj = dit
            for part in parts[:-1]:
                obj = getattr(obj, part)
            attr_name = parts[-1]
            if hasattr(obj, attr_name):
                existing = getattr(obj, attr_name)
                if isinstance(existing, nn.Parameter):
                    existing.data = tensor
                else:
                    setattr(obj, attr_name, tensor)
        del shard_dict
        gc.collect()

    dit.eval()
    # Cast non-quantized params to bfloat16
    for name, module in dit.named_modules():
        if isinstance(module, QuantizedLinear):
            continue
        for param_name, param in module.named_parameters(recurse=False):
            if param.dtype == torch.float32:
                param.data = param.data.to(torch.bfloat16)

    # Build pipeline first, then load LoRA and move everything to GPU
    pipe = LongCatVideoAvatarPipeline(
        tokenizer=tokenizer,
        text_encoder=None,
        vae=vae,
        scheduler=scheduler,
        dit=dit,
        audio_encoder=None,
        audio_feature_extractor=None,
        model_type="avatar-v1.5",
    )

    # Load distillation LoRA
    distill_lora_path = os.path.join(checkpoint_dir, 'lora', 'dmd_lora.safetensors')
    if os.path.exists(distill_lora_path):
        pipe.dit.load_lora(distill_lora_path, "dmd", multiplier=1.0, lora_network_dim=128, lora_network_alpha=64)
        pipe.dit.enable_loras(["dmd"])

    # Move everything to GPU (including LoRA weights)
    pipe.to(device)
    torch_gc()

    # Optimize: replace naive INT8 dequant with torchao's optimized kernels + torch.compile
    if args.use_torchao:
        use_fp8 = getattr(args, 'use_fp8', False)
        quant_label = "float8_weight_only" if use_fp8 else "int8_weight_only"
        print(f"[OPT] Applying torchao {quant_label} quantization...")
        from torchao.quantization import int8_weight_only, quantize_
        if use_fp8:
            from torchao.quantization import float8_weight_only
        from longcat_video.modules.quantization import QuantizedLinear
        # First, convert QuantizedLinear layers back to standard Linear with dequantized weights
        # so torchao can re-quantize with optimized kernels
        for name, module in list(pipe.dit.named_modules()):
            if isinstance(module, QuantizedLinear):
                parts = name.split(".")
                parent = pipe.dit
                for part in parts[:-1]:
                    parent = getattr(parent, part)
                # Reconstruct as Linear
                linear = torch.nn.Linear(module.in_features, module.out_features,
                                         bias=module.bias is not None, device=device, dtype=torch.bfloat16)
                linear.weight.data = (module.weight_int8.to(torch.bfloat16) * module.weight_scale.to(torch.bfloat16).unsqueeze(1))
                if module.bias is not None:
                    linear.bias.data = module.bias.to(torch.bfloat16)
                setattr(parent, parts[-1], linear)
        torch_gc()
        # Now apply torchao's optimized quantization
        quant_config = float8_weight_only() if use_fp8 else int8_weight_only()
        quantize_(pipe.dit, quant_config)
        # Warm up RoPE frequency cache on GPU before torch.compile traces the forward
        print("[OPT] Warming up RoPE frequency cache on GPU...")
        N_t = num_frames  # temporal latents
        N_h = height // 16  # spatial height latents
        N_w = width // 16   # spatial width latents
        for block in pipe.dit.blocks:
            if hasattr(block, 'attn') and hasattr(block.attn, 'rope_3d'):
                rope = block.attn.rope_3d
                grid_size = (N_t, N_h, N_w)
                key_name = '.'.join([str(i) for i in grid_size]) + f"-None-None"
                if key_name not in rope.freqs_dict:
                    rope.register_grid_size(grid_size, key_name, None, None)
                rope.freqs_dict[key_name] = rope.freqs_dict[key_name].to(device)

        print("[OPT] Compiling model with torch.compile...")
        pipe.dit = torch.compile(pipe.dit, mode="max-autotune-no-cudagraphs")
        torch_gc()

    # Monkey-patch encode_prompt to use cached embeddings
    cached_prompt_embeds = prompt_embeds_cpu.to(device, dtype=dit.dtype)
    cached_mask = mask_cpu.to(device)

    def patched_encode_prompt(prompt=None, negative_prompt=None, do_classifier_free_guidance=False,
                              num_videos_per_prompt=1, max_sequence_length=512, dtype=None, device=None):
        neg_embeds = torch.zeros_like(cached_prompt_embeds) if do_classifier_free_guidance else None
        neg_mask = cached_mask if do_classifier_free_guidance else None
        return cached_prompt_embeds, cached_mask, neg_embeds, neg_mask

    pipe.encode_prompt = patched_encode_prompt

    # Mock text_encoder.config.d_model for generate_avc which needs it
    class _MockConfig:
        d_model = 4096
    class _MockTextEncoder:
        config = _MockConfig()
    pipe.text_encoder = _MockTextEncoder()

    # Seed
    generator = torch.Generator(device=device)
    generator.manual_seed(42 + global_rank)

    # Prepare audio embedding for first clip
    indices = torch.arange(2 * 2 + 1) - 2
    audio_start_idx = 0
    audio_end_idx = audio_start_idx + audio_stride * num_frames
    center_indices = torch.arange(audio_start_idx, audio_end_idx, audio_stride).unsqueeze(1) + indices.unsqueeze(0)
    center_indices = torch.clamp(center_indices, min=0, max=full_audio_emb_cpu.shape[0] - 1)
    audio_emb = full_audio_emb_cpu[center_indices][None, ...].to(device)

    print(f"[PHASE 3] Generating segment 1/{num_segments}...")

    if stage_1 == 'ai2v':
        image_path = input_data['cond_image']
        image = load_image(image_path)
        output_tuple = pipe.generate_ai2v(
            image=image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            resolution=resolution,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            text_guidance_scale=text_guidance_scale,
            audio_guidance_scale=audio_guidance_scale,
            output_type='both',
            generator=generator,
            audio_emb=audio_emb,
            use_distill=True,
        )
    elif stage_1 == 'at2v':
        output_tuple = pipe.generate_at2v(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            text_guidance_scale=text_guidance_scale,
            audio_guidance_scale=audio_guidance_scale,
            generator=generator,
            output_type='both',
            audio_emb=audio_emb,
            use_distill=True,
        )
    else:
        raise NotImplementedError(f"Unsupported stage_1: {stage_1}")

    output, latent = output_tuple
    output = output[0]
    video = [(output[i] * 255).astype(np.uint8) for i in range(output.shape[0])]
    video = [PIL.Image.fromarray(img) for img in video]

    if cp_rank == 0:
        output_tensor = torch.from_numpy(np.array(video))
        save_video_ffmpeg(output_tensor, os.path.join(output_dir, f"segment_001"), raw_speech_path, fps=save_fps, quality=5)
    del output
    torch_gc()

    # Video continuation segments — save per-segment to avoid O(n²) memory/encoding
    if num_segments > 1:
        width, height = video[0].size
        ref_latent = latent[:, :, :1].clone()
        current_video = video
        segment_files = [os.path.join(output_dir, f"segment_001.mp4")]

        for segment_idx in range(1, num_segments):
            print(f"[PHASE 3] Generating segment {segment_idx + 1}/{num_segments}...")

            audio_start_idx = audio_start_idx + audio_stride * (num_frames - num_cond_frames)
            audio_end_idx = audio_start_idx + audio_stride * num_frames
            center_indices = torch.arange(audio_start_idx, audio_end_idx, audio_stride).unsqueeze(1) + indices.unsqueeze(0)
            center_indices = torch.clamp(center_indices, min=0, max=full_audio_emb_cpu.shape[0] - 1)
            audio_emb = full_audio_emb_cpu[center_indices][None, ...].to(device)

            output_tuple = pipe.generate_avc(
                video=current_video,
                video_latent=latent,
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                num_frames=num_frames,
                num_cond_frames=num_cond_frames,
                num_inference_steps=num_inference_steps,
                text_guidance_scale=text_guidance_scale,
                audio_guidance_scale=audio_guidance_scale,
                generator=generator,
                output_type='both',
                use_kv_cache=True,
                offload_kv_cache=False,
                enhance_hf=False,
                audio_emb=audio_emb,
                ref_latent=ref_latent,
                ref_img_index=args.ref_img_index,
                mask_frame_range=args.mask_frame_range,
                use_distill=True,
            )
            output, latent = output_tuple
            output = output[0]
            new_video = [(output[i] * 255).astype(np.uint8) for i in range(output.shape[0])]
            new_video = [PIL.Image.fromarray(img) for img in new_video]
            del output

            # Save only NEW frames (skip conditioning overlap)
            if cp_rank == 0:
                new_frames_only = new_video[num_cond_frames:]
                seg_tensor = torch.from_numpy(np.array(new_frames_only))
                seg_name = f"segment_{segment_idx + 1:03d}"
                seg_path = os.path.join(output_dir, f"{seg_name}.mp4")
                save_video_ffmpeg(seg_tensor, os.path.join(output_dir, seg_name), raw_speech_path, fps=save_fps, quality=5)
                segment_files.append(seg_path)
                del seg_tensor, new_frames_only
                print(f"  Saved {seg_name}.mp4 ({len(new_video) - num_cond_frames} new frames)")

            current_video = new_video
            torch_gc()

        # Concat all segments + audio into final video
        if cp_rank == 0:
            print("[PHASE 4] Concatenating segments into final video...")
            concat_list = os.path.join(output_dir, "concat_list.txt")
            with open(concat_list, "w") as f:
                for seg_file in segment_files:
                    # Extract the raw video (no audio) for concat
                    raw_file = seg_file.replace(".mp4", "-cropvideo.mp4")
                    if os.path.exists(raw_file):
                        f.write(f"file '{os.path.abspath(raw_file)}'\n")
                    elif os.path.exists(seg_file):
                        f.write(f"file '{os.path.abspath(seg_file)}'\n")

            final_video = os.path.join(output_dir, "final_video.mp4")
            os.system(f"ffmpeg -y -f concat -safe 0 -i {concat_list} -c copy {os.path.join(output_dir, 'final_noaudio.mp4')} 2>/dev/null")
            os.system(f"ffmpeg -y -i {os.path.join(output_dir, 'final_noaudio.mp4')} -i {raw_speech_path} -c:v copy -c:a aac -shortest {final_video} 2>/dev/null")
            print(f"  Final video: {final_video}")

    print("[DONE] Generation complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_json', type=str, default='assets/avatar/single_example_1.json')
    parser.add_argument('--output_dir', type=str, default='./outputs_avatar_single')
    parser.add_argument('--resolution', type=str, default='480p', choices=['480p', '720p'])
    parser.add_argument('--num_segments', type=int, default=1)
    parser.add_argument('--ref_img_index', type=int, default=10)
    parser.add_argument('--mask_frame_range', type=int, default=3)
    parser.add_argument('--stage_1', type=str, default='ai2v', choices=['ai2v', 'at2v'])
    parser.add_argument('--checkpoint_dir', type=str, default='./weights/LongCat-Video-Avatar-1.5')
    parser.add_argument('--use_torchao', action='store_true', help='Use torchao optimized quantization + torch.compile')
    parser.add_argument('--use_fp8', action='store_true', help='Use FP8 weight-only instead of INT8 (requires compute capability 8.9+)')
    parser.add_argument('--num_steps_override', type=int, default=0, help='Override num_inference_steps for testing')
    args = parser.parse_args()
    generate(args)
