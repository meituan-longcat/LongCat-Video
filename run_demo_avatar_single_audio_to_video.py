import os
import json
import time
import math
import random
import argparse
import datetime
import gc
import weakref
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
from longcat_video.distributed_utils import broadcast_tensor_mapping, broadcast_variable_tensor

# -------- avatar related --------
import librosa
from longcat_video.audio_process import get_audio_encoder, get_audio_feature_extractor
from longcat_video.audio_process.torch_utils import save_video_ffmpeg
from audio_separator.separator import Separator


def torch_gc():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


PROMPT_CONDITIONING_KEYS = (
    "prompt_embeds",
    "prompt_attention_mask",
    "negative_prompt_embeds",
    "negative_prompt_attention_mask",
)


def log_cuda_memory(phase, rank):
    if rank != 0:
        return
    gib = 1024 ** 3
    print(
        f"[memory] {phase}: "
        f"allocated={torch.cuda.memory_allocated() / gib:.2f} GiB, "
        f"reserved={torch.cuda.memory_reserved() / gib:.2f} GiB, "
        f"peak={torch.cuda.max_memory_allocated() / gib:.2f} GiB",
        flush=True,
    )


def cuda_allocated_bytes():
    return torch.cuda.memory_allocated()


def surviving_reference_types(reference):
    """Describe unexpected owners without retaining them past this call."""
    target = reference()
    if target is None:
        return []
    descriptions = []
    for owner in gc.get_referrers(target):
        if isinstance(owner, dict):
            names = [key for key, value in owner.items() if value is target]
            descriptions.append(f"dict(keys={names[:8]})")
        else:
            descriptions.append(type(owner).__name__)
    del target
    return descriptions


def is_preprocessing_rank(cp_rank):
    return cp_rank == 0


def normalize_output_name(output_name):
    """Validate a filename-only output name and normalize its MP4 suffix."""
    if output_name is None:
        return None
    if not output_name or output_name in {".", ".."}:
        raise argparse.ArgumentTypeError(
            "--output_name must be a non-empty filename other than '.' or '..'"
        )
    if "/" in output_name or "\\" in output_name:
        raise argparse.ArgumentTypeError(
            "--output_name must be a filename only and cannot contain path separators"
        )

    stem = output_name
    while stem.lower().endswith(".mp4"):
        stem = stem[:-4]
    if not stem or stem in {".", ".."}:
        raise argparse.ArgumentTypeError(
            "--output_name must contain a filename before the .mp4 suffix"
        )
    return f"{stem}.mp4"


def build_output_paths(output_dir, output_name, stage_1, num_segments):
    """Return the primary and continuation output paths for this invocation."""
    output_dir = Path(output_dir)
    if output_name is None:
        paths = [output_dir / f"{stage_1}_demo_1.mp4"]
        paths.extend(
            output_dir / f"video_continue_{segment_idx}.mp4"
            for segment_idx in range(2, num_segments + 1)
        )
        return paths

    normalized_name = normalize_output_name(output_name)
    stem = normalized_name[:-4]
    paths = [output_dir / normalized_name]
    paths.extend(
        output_dir / f"{stem}_continue_{segment_idx}.mp4"
        for segment_idx in range(2, num_segments + 1)
    )
    return paths


def prepare_output_paths(output_dir, output_name, stage_1, num_segments):
    """Create the output directory and reject collisions for named outputs."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = build_output_paths(output_dir, output_name, stage_1, num_segments)
    if output_name is not None:
        existing = [path for path in paths if path.exists()]
        if existing:
            raise FileExistsError(
                "Refusing to overwrite existing output file(s): "
                + ", ".join(str(path) for path in existing)
            )
    return paths


def ffmpeg_output_base(output_path):
    """Adapt a canonical .mp4 path to save_video_ffmpeg's suffixless API."""
    return str(output_path.with_suffix(""))

def generate_random_uid():
    timestamp_part = str(int(time.time()))[-6:]
    random_part = str(random.randint(100000, 999999))
    uid = timestamp_part + random_part
    return uid

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

    # load parsed args
    input_json = args.input_json
    checkpoint_dir = args.checkpoint_dir
    context_parallel_size = args.context_parallel_size
    stage_1 = args.stage_1
    num_inference_steps = args.num_inference_steps
    text_guidance_scale = args.text_guidance_scale
    audio_guidance_scale = args.audio_guidance_scale
    resolution = args.resolution
    num_segments = max(1, args.num_segments)
    output_dir = args.output_dir
    model_type = args.model_type
    use_distill = args.use_distill
    use_int8 = args.use_int8

    output_paths = prepare_output_paths(
        output_dir, args.output_name, stage_1, num_segments
    )

    if use_distill and model_type == "avatar-v1.5":
        num_inference_steps = 8
        text_guidance_scale = 1.0
        audio_guidance_scale = 1.0

    # set up default inference params
    save_fps = 16
    audio_stride = 2
    if model_type == "avatar-v1.5":
        save_fps = 25
        audio_stride = 1
    num_frames = 93
    num_cond_frames = 13

    if resolution == '480p':
        height, width = 480, 832
    elif resolution == '720p':
        height, width = 768, 1280

    # case setup
    with open(input_json, 'r', encoding='utf-8') as f:
        input_data = json.load(f)
    prompt = input_data['prompt']
    negative_prompt = "Close-up, Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
    raw_speech_path = input_data['cond_audio']['person1']
    
    # prepare distributed environment
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=3600*24))
    global_rank    = dist.get_rank()
    num_processes  = dist.get_world_size()

    # initialize context parallel
    context_parallel_util.init_context_parallel(context_parallel_size=context_parallel_size, global_rank=global_rank, world_size=num_processes)
    cp_rank = context_parallel_util.get_cp_rank()
    cp_size = context_parallel_util.get_cp_size()
    cp_split_hw = context_parallel_util.get_optimal_split(cp_size)

    if args.staged_preprocessing and cp_size != num_processes:
        raise ValueError(
            "--staged_preprocessing currently requires one CP group "
            "(context_parallel_size must equal world size)"
        )
    cp_source_rank = context_parallel_util.get_cp_rank_list()[0]

    prompt_conditioning = None
    full_audio_emb = None

    if args.staged_preprocessing:
        base_model_dir = os.path.join(checkpoint_dir, '..', 'LongCat-Video')
        do_classifier_free_guidance = text_guidance_scale > 1.0 or audio_guidance_scale > 1.0

        torch.cuda.reset_peak_memory_stats()
        log_cuda_memory("before text encoder loading", global_rank)
        text_phase_baseline = cuda_allocated_bytes()
        if is_preprocessing_rank(cp_rank):
            print(f"[rank {global_rank}] staged text preprocessing", flush=True)
            tokenizer = AutoTokenizer.from_pretrained(
                base_model_dir, subfolder="tokenizer", torch_dtype=torch.bfloat16
            )
            text_encoder = UMT5EncoderModel.from_pretrained(
                base_model_dir, subfolder="text_encoder", torch_dtype=torch.bfloat16
            ).to(local_rank)
            text_pipe = LongCatVideoAvatarPipeline(
                tokenizer=tokenizer, text_encoder=text_encoder, vae=None,
                scheduler=None, dit=None, model_type=model_type,
            )
            encoded = text_pipe.encode_prompt(
                prompt=prompt,
                negative_prompt=negative_prompt,
                do_classifier_free_guidance=do_classifier_free_guidance,
                device=local_rank,
                dtype=torch.bfloat16,
            )
            # Materialize only detached tensors.  No ModelOutput, view with a
            # grad_fn, hidden-state tuple, attention, or KV cache survives.
            prompt_conditioning = {
                key: value.detach().contiguous() if value is not None else None
                for key, value in zip(PROMPT_CONDITIONING_KEYS, encoded)
            }
            encoded = None
            log_cuda_memory("after detaching prompt embeddings", global_rank)
        log_cuda_memory("after text encoding", global_rank)
        prompt_conditioning = broadcast_tensor_mapping(
            prompt_conditioning,
            PROMPT_CONDITIONING_KEYS,
            src=cp_source_rank,
            group=context_parallel_util.get_cp_group(),
        )
        if is_preprocessing_rank(cp_rank):
            text_encoder_ref = weakref.ref(text_encoder)
            # The temporary pipeline is a surviving owner, so remove its
            # module attributes as well as the local encoder name.
            text_pipe.text_encoder.to("cpu")
            log_cuda_memory("after moving text encoder to CPU", global_rank)
            text_pipe.text_encoder = None
            text_pipe.tokenizer = None
            log_cuda_memory("after setting text pipeline attributes to None", global_rank)
            del text_pipe, text_encoder, tokenizer, encoded
            log_cuda_memory("after deleting text temporary objects", global_rank)
        gc.collect()
        log_cuda_memory("after text gc.collect()", global_rank)
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        log_cuda_memory("after text torch.cuda.empty_cache()", global_rank)
        log_cuda_memory("after text encoder cleanup", global_rank)
        if is_preprocessing_rank(cp_rank):
            remaining_text_bytes = max(0, cuda_allocated_bytes() - text_phase_baseline)
            if remaining_text_bytes > 2 * 1024 ** 3:
                owners = surviving_reference_types(text_encoder_ref)
                message = (
                    "staged text cleanup retained "
                    f"{remaining_text_bytes / 1024 ** 3:.2f} GiB above its baseline; "
                    f"text encoder alive={text_encoder_ref() is not None}, owners={owners}"
                )
                print(f"[memory] ERROR: {message}", flush=True)
                if args.dry_run_after_model_load:
                    raise RuntimeError(message)
            del text_encoder_ref
        torch.cuda.reset_peak_memory_stats()

        if is_preprocessing_rank(cp_rank):
            print(f"[rank {global_rank}] staged audio preprocessing", flush=True)
            audio_model_checkpoint_path = os.path.join(
                checkpoint_dir,
                'chinese-wav2vec2-base' if model_type == "avatar-v1.0" else 'whisper-large-v3',
            )
            audio_encoder = get_audio_encoder(audio_model_checkpoint_path, model_type).to(local_rank)
            audio_feature_extractor = get_audio_feature_extractor(audio_model_checkpoint_path, model_type)
            log_cuda_memory("after audio model loading", global_rank)
            vocal_separator_path = os.path.join(checkpoint_dir, 'vocal_separator/Kim_Vocal_2.onnx')
            audio_output_dir_temp = Path("./audio_temp_file")
            os.makedirs(audio_output_dir_temp, exist_ok=True)
            vocal_separator = Separator(
                output_dir=audio_output_dir_temp / "vocals",
                output_single_stem="vocals",
                model_file_dir=os.path.dirname(vocal_separator_path),
            )
            vocal_separator.load_model(os.path.basename(vocal_separator_path))
            temp_vocal_path = extract_vocal_from_speech(
                raw_speech_path,
                f"/tmp/temp_speech_{generate_random_uid()}_{global_rank}_vocal.wav",
                vocal_separator,
                audio_output_dir_temp,
            )
            assert temp_vocal_path is not None and os.path.exists(temp_vocal_path), "No vocal detected"
            generate_duration = num_frames / save_fps + (num_segments - 1) * (num_frames - num_cond_frames) / save_fps
            speech_array, sr = librosa.load(temp_vocal_path, sr=16000)
            added_sample_nums = math.ceil((generate_duration - len(speech_array) / sr) * sr)
            if added_sample_nums > 0:
                speech_array = np.append(speech_array, [0.] * added_sample_nums)
            audio_pipe = LongCatVideoAvatarPipeline(
                tokenizer=None, text_encoder=None, vae=None, scheduler=None, dit=None,
                audio_encoder=audio_encoder,
                audio_feature_extractor=audio_feature_extractor,
                model_type=model_type,
            )
            log_cuda_memory("before audio embedding", global_rank)
            full_audio_emb = audio_pipe.get_audio_embedding(
                speech_array, fps=save_fps * audio_stride, device=local_rank,
                sample_rate=sr, model_type=model_type,
            ).detach().contiguous()
            log_cuda_memory("after audio embedding", global_rank)
            if torch.isnan(full_audio_emb).any():
                raise ValueError("broken audio embedding with nan values")
            if os.path.exists(temp_vocal_path):
                os.remove(temp_vocal_path)
        log_cuda_memory("after audio preprocessing", global_rank)
        full_audio_emb = broadcast_variable_tensor(
            full_audio_emb, src=cp_source_rank, group=context_parallel_util.get_cp_group()
        )
        if is_preprocessing_rank(cp_rank):
            audio_pipe.audio_encoder.to("cpu")
            audio_pipe.audio_encoder = None
            audio_pipe.audio_feature_extractor = None
            del audio_pipe, audio_encoder, audio_feature_extractor, vocal_separator
            del speech_array
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        log_cuda_memory("after audio-model cleanup", global_rank)
        torch.cuda.reset_peak_memory_stats()

    # initialize models
    tokenizer = None if args.staged_preprocessing else AutoTokenizer.from_pretrained(os.path.join(checkpoint_dir, '..', 'LongCat-Video'), subfolder="tokenizer", torch_dtype=torch.bfloat16)
    text_encoder = None if args.staged_preprocessing else UMT5EncoderModel.from_pretrained(os.path.join(checkpoint_dir, '..', 'LongCat-Video'), subfolder="text_encoder", torch_dtype=torch.bfloat16)
    vae = AutoencoderKLWan.from_pretrained(os.path.join(checkpoint_dir, '..', 'LongCat-Video'), subfolder="vae", torch_dtype=torch.bfloat16)
    if model_type == "avatar-v1.0":
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(os.path.join(checkpoint_dir, '..', 'LongCat-Video'), subfolder="scheduler", torch_dtype=torch.bfloat16)
    elif model_type == "avatar-v1.5":
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(checkpoint_dir, subfolder="scheduler", torch_dtype=torch.bfloat16)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}. Expected 'avatar-v1.0' or 'avatar-v1.5'.")
    
    if model_type == "avatar-v1.0":
        dit = LongCatVideoAvatarTransformer3DModel.from_pretrained(checkpoint_dir, subfolder="avatar_single", cp_split_hw=cp_split_hw, torch_dtype=torch.bfloat16)
    elif model_type == "avatar-v1.5":
        if use_int8:
            print("[INFO] Loading INT8 quantized DiT model...")
            dit = load_quantized_dit(checkpoint_dir, subfolder="base_model_int8", cp_split_hw=cp_split_hw)
        else:
            dit = LongCatVideoAvatarTransformer3DModel.from_pretrained(checkpoint_dir, subfolder="base_model", cp_split_hw=cp_split_hw, torch_dtype=torch.bfloat16)
        if use_distill:
            distill_checkpoint_path = os.path.join(checkpoint_dir, 'lora', f'dmd_lora.safetensors')
            if os.path.exists(distill_checkpoint_path):
                dit.load_lora(distill_checkpoint_path, "dmd", multiplier=1.0, lora_network_dim=128, lora_network_alpha=64)
                dit.enable_loras(["dmd"])
    else:
        raise ValueError(f"Unsupported model_type: {model_type}. Expected 'avatar-v1.0' or 'avatar-v1.5'.")
    
    # Legacy mode retains per-rank preprocessing for backward compatibility.
    if not args.staged_preprocessing:
        if model_type == "avatar-v1.0":
            audio_model_checkpoint_path = os.path.join(checkpoint_dir, 'chinese-wav2vec2-base')
        elif model_type == "avatar-v1.5":
            audio_model_checkpoint_path = os.path.join(checkpoint_dir, 'whisper-large-v3')
        audio_encoder = get_audio_encoder(audio_model_checkpoint_path, model_type).to(local_rank)
        audio_feature_extractor = get_audio_feature_extractor(audio_model_checkpoint_path, model_type)
        vocal_separator_path = os.path.join(checkpoint_dir, 'vocal_separator/Kim_Vocal_2.onnx')
        audio_output_dir_temp = Path("./audio_temp_file")
        os.makedirs(audio_output_dir_temp, exist_ok=True)
        vocal_separator = Separator(
            output_dir=audio_output_dir_temp / "vocals",
            output_single_stem="vocals",
            model_file_dir=os.path.dirname(vocal_separator_path),
        )
        vocal_separator.load_model(os.path.basename(vocal_separator_path))
    else:
        audio_encoder = audio_feature_extractor = None

    
    # initialize pipeline
    pipe = LongCatVideoAvatarPipeline(
        tokenizer = tokenizer,
        text_encoder = text_encoder,
        vae = vae,
        scheduler = scheduler,
        dit = dit,
        audio_encoder=audio_encoder,
        audio_feature_extractor=audio_feature_extractor,
        model_type=model_type
    )
    pipe.to(local_rank)
    log_cuda_memory("after DiT loading", global_rank)

    if args.dry_run_after_model_load:
        log_cuda_memory("before denoising", global_rank)
        if global_rank == 0:
            print("[dry-run] all staged inputs broadcast and generation components loaded", flush=True)
        dist.barrier(group=context_parallel_util.get_cp_group())
        return

    global_seed = 42
    seed = global_seed + global_rank

    generator = torch.Generator(device=local_rank)
    generator.manual_seed(seed)

    if not args.staged_preprocessing and cp_rank == 0:
        # extract vocal
        temp_vocal_path = extract_vocal_from_speech(raw_speech_path, f"/tmp/temp_speech_{generate_random_uid()}_{global_rank}_vocal.wav", vocal_separator, audio_output_dir_temp)
        assert temp_vocal_path is not None and os.path.exists(temp_vocal_path), f"No vocal detected"    

        # audio padding to target length
        generate_duration = num_frames / save_fps + (num_segments-1)*(num_frames-num_cond_frames) / save_fps
        speech_array, sr = librosa.load(temp_vocal_path, sr=16000)
        source_duraion = len(speech_array) / sr
        added_sample_nums = math.ceil((generate_duration - source_duraion) * sr)
        if added_sample_nums > 0:
            speech_array = np.append(speech_array, [0.]*added_sample_nums)

        # audio embedding
        full_audio_emb = pipe.get_audio_embedding(speech_array, fps=save_fps*audio_stride, device=local_rank, sample_rate=sr, model_type=model_type)
        if torch.isnan(full_audio_emb).any():
            raise ValueError(f"broken audio embedding with nan values")

        if context_parallel_util.get_cp_size() > 1:
            full_audio_emb_shape_list = list(full_audio_emb.size())
            full_audio_emb_tensor_shape_list = torch.tensor(full_audio_emb_shape_list, dtype=torch.int64, device=full_audio_emb.device)
            context_parallel_util.cp_broadcast(full_audio_emb_tensor_shape_list)
            context_parallel_util.cp_broadcast(full_audio_emb)
        
        if os.path.exists(temp_vocal_path):
            os.remove(temp_vocal_path)

    elif not args.staged_preprocessing and context_parallel_util.get_cp_size() > 1:
        full_audio_emb_tensor_shape_list = torch.zeros(3, dtype=torch.int64, device=local_rank)
        context_parallel_util.cp_broadcast(full_audio_emb_tensor_shape_list)
        full_audio_emb_shape_list = full_audio_emb_tensor_shape_list.tolist()
        full_audio_emb = torch.zeros(*full_audio_emb_shape_list, dtype=torch.float32, device=local_rank)
        context_parallel_util.cp_broadcast(full_audio_emb)

    # prepare audio embedding for the first clip
    indices = torch.arange(2 * 2 + 1) - 2
    audio_start_idx = 0
    audio_end_idx = audio_start_idx + audio_stride * num_frames

    center_indices = torch.arange(audio_start_idx, audio_end_idx, audio_stride).unsqueeze(1) + indices.unsqueeze(0)
    center_indices = torch.clamp(center_indices, min=0, max=full_audio_emb.shape[0]-1)
    audio_emb = full_audio_emb[center_indices][None,...].to(local_rank)


    if local_rank == 0:
        print(f"Generating segment 1/{num_segments}...")
    log_cuda_memory("before denoising", global_rank)

    if stage_1 == 'at2v':
        # ==============================
        #          at2v (480P)
        # ==============================
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
            use_distill=use_distill,
            prompt_conditioning=prompt_conditioning,
        )
        output, latent = output_tuple 
        output = output[0] 
        video = [(output[i] * 255).astype(np.uint8) for i in range(output.shape[0])]
        video = [PIL.Image.fromarray(img) for img in video]

        if cp_rank == 0:
            output_tensor = torch.from_numpy(np.array(video))
            save_video_ffmpeg(output_tensor, ffmpeg_output_base(output_paths[0]), raw_speech_path, fps=save_fps, quality=5)
        del output
        torch_gc()
    
    elif stage_1 == 'ai2v':
        # ==============================
        #          ai2v (480P)
        # ==============================
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
            use_distill=use_distill,
            prompt_conditioning=prompt_conditioning,
        )
        output, latent = output_tuple
        output = output[0]
        video = [(output[i] * 255).astype(np.uint8) for i in range(output.shape[0])]
        video = [PIL.Image.fromarray(img) for img in video]

        if cp_rank == 0:
            output_tensor = torch.from_numpy(np.array(video))
            save_video_ffmpeg(output_tensor, ffmpeg_output_base(output_paths[0]), raw_speech_path, fps=save_fps, quality=5)
        del output
        torch_gc()
    else:
        raise NotImplementedError(f"Not supported type of stage_1: {stage_1}")

    if context_parallel_util.get_cp_size() > 1:
        torch.distributed.barrier(group=context_parallel_util.get_cp_group())

    # =========================================
    #         long video generation (480P)
    # =========================================
    # load parsed long video args
    ref_img_index = args.ref_img_index
    mask_frame_range = args.mask_frame_range

    width, height = video[0].size
    current_video = video
    ref_latent = latent[:, :, :1].clone()
    all_generated_frames = video

    for segment_idx in range(1, num_segments):
        if local_rank == 0:
            print(f"Generating segment {segment_idx+1}/{num_segments}...")
        
        # prepare audio embedding for the next clip
        audio_start_idx = audio_start_idx + audio_stride * (num_frames - num_cond_frames)
        audio_end_idx   = audio_start_idx + audio_stride * num_frames
        center_indices = torch.arange(audio_start_idx, audio_end_idx, audio_stride).unsqueeze(1) + indices.unsqueeze(0)
        center_indices = torch.clamp(center_indices, min=0, max=full_audio_emb.shape[0]-1)
        audio_emb = full_audio_emb[center_indices][None,...].to(local_rank)
        
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
            enhance_hf=True if not use_distill else False,
            audio_emb=audio_emb,
            ref_latent=ref_latent,
            ref_img_index=ref_img_index,
            mask_frame_range=mask_frame_range,
            use_distill=use_distill,
            prompt_conditioning=prompt_conditioning,
        )
        output, latent = output_tuple

        output = output[0]
        new_video = [(output[i] * 255).astype(np.uint8) for i in range(output.shape[0])]
        new_video = [PIL.Image.fromarray(img) for img in new_video]
        del output

        all_generated_frames.extend(new_video[num_cond_frames:])

        current_video = new_video

        if cp_rank == 0:
            output_tensor = torch.from_numpy(np.array(all_generated_frames))
            save_video_ffmpeg(output_tensor, ffmpeg_output_base(output_paths[segment_idx]), raw_speech_path, fps=save_fps, quality=5)
            del output_tensor


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_json',
        type=str,
        default='assets/avatar/single_example_1.json'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./outputs_avatar_single'
    )
    parser.add_argument(
        '--output_name',
        type=normalize_output_name,
        default=None,
        help="Output filename (with or without .mp4); path separators are not allowed",
    )
    parser.add_argument(
        '--resolution',
        type=str,
        default='480p',
        choices=['480p', '720p']
    )
    parser.add_argument(
        '--num_segments',
        type=int,
        default=1
    )
    parser.add_argument(
        '--num_inference_steps',
        type=int,
        default=50
    )
    parser.add_argument(
        '--ref_img_index',
        type=int,
        default=10
    )
    parser.add_argument(
        '--mask_frame_range',
        type=int,
        default=3
    )
    parser.add_argument(
        '--text_guidance_scale',
        type=float,
        default=4.0
    )
    parser.add_argument(
        '--audio_guidance_scale',
        type=float,
        default=4.0
    )
    parser.add_argument(
        '--stage_1',
        type=str,
        default='ai2v',
        choices=['ai2v', 'at2v']
    )
    parser.add_argument(
        "--context_parallel_size",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./weights/LongCat-Video-Avatar",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="avatar-v1.0",
    )
    parser.add_argument(
        "--use_distill",
        action='store_true',
    )
    parser.add_argument(
        "--use_int8",
        action='store_true',
        help="Load INT8 quantized DiT model for reduced VRAM usage"
    )
    parser.add_argument(
        "--staged_preprocessing",
        action="store_true",
        help="Run text/audio preprocessing on CP rank 0 and free those models before loading DiT",
    )
    parser.add_argument(
        "--dry_run_after_model_load",
        action="store_true",
        help="Exit collectively after staged inputs and generation models are loaded",
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = _parse_args()
    generate(args)
