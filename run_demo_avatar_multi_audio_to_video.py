import os
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
from longcat_video.modules.quantization import enable_fp8_ffn, load_quantized_dit
from longcat_video.context_parallel import context_parallel_util

# -------- avatar related --------
import librosa
import soundfile as sf
from longcat_video.audio_process import get_audio_encoder, get_audio_feature_extractor
from longcat_video.audio_process.torch_utils import save_video_ffmpeg
from longcat_video.audio_process.vocal_separator import separate_vocals
from audio_separator.separator import Separator


def torch_gc():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

def generate_random_uid():
    timestamp_part = str(int(time.time()))[-6:]
    random_part = str(random.randint(100000, 999999))
    uid = timestamp_part + random_part
    return uid

def extract_vocal_from_speech(source_path, vocal_separator, duration):
    if source_path is None:
        return None
    speech_array, _ = separate_vocals(vocal_separator, source_path, duration=duration)
    return speech_array

def audio_prepare_multi(left_speech_array, right_speech_array, generate_duration, left_raw_speech_path, right_raw_speech_path, sample_rate=16000, audio_type='para'):
    sr = sample_rate
    if left_speech_array is not None:
        left_raw_speech_array, _ = librosa.load(left_raw_speech_path, sr=sample_rate) 
    
    if right_speech_array is not None:
        right_raw_speech_array, _ = librosa.load(right_raw_speech_path, sr=sample_rate) 
    
    if left_speech_array is None:
        left_speech_array = np.zeros_like(right_speech_array)
        left_raw_speech_array = np.zeros_like(right_raw_speech_array)
    
    if right_speech_array is None:
        right_speech_array = np.zeros_like(left_speech_array)
        right_raw_speech_array = np.zeros_like(left_raw_speech_array)
    
    if audio_type == 'add':
        left_speech_array_ext = np.concatenate([left_speech_array, np.zeros_like(right_speech_array)])
        right_speech_array_ext = np.concatenate([np.zeros_like(left_speech_array), right_speech_array])
        merge_raw_speech = np.concatenate([left_raw_speech_array, np.zeros_like(right_raw_speech_array)]) + \
                            np.concatenate([np.zeros_like(left_raw_speech_array), right_raw_speech_array])
    elif audio_type == 'para':
        aligned_length = min(
            len(left_speech_array), len(right_speech_array),
            len(left_raw_speech_array), len(right_raw_speech_array),
        )
        if aligned_length == 0:
            raise ValueError("Cannot align empty audio arrays for parallel mixing")
        left_speech_array_ext = left_speech_array[:aligned_length]
        right_speech_array_ext = right_speech_array[:aligned_length]
        merge_raw_speech = (left_raw_speech_array[:aligned_length]
                            + right_raw_speech_array[:aligned_length])
    else:
        raise NotImplementedError(f"Unsupported audio_type of {audio_type}")
    
    assert len(left_speech_array_ext) == len(right_speech_array_ext), f"The two speech lengths should be equal"
   

    source_duraion = len(left_speech_array_ext) / sr
    added_sample_nums = math.ceil((generate_duration - source_duraion) * sr)
    if added_sample_nums > 0:
        left_speech_array_ext  = np.append(left_speech_array_ext, [0.]*added_sample_nums)
        right_speech_array_ext = np.append(right_speech_array_ext, [0.]*added_sample_nums)
        merge_raw_speech = np.append(merge_raw_speech, [0.]*added_sample_nums)

    return left_speech_array_ext, right_speech_array_ext, merge_raw_speech
        

def generate(args):

    # load parsed args
    input_json = args.input_json
    checkpoint_dir = args.checkpoint_dir
    context_parallel_size = args.context_parallel_size
    num_inference_steps = args.num_inference_steps
    text_guidance_scale = args.text_guidance_scale
    audio_guidance_scale = args.audio_guidance_scale
    resolution = args.resolution
    num_segments = max(1, args.num_segments)
    output_dir = args.output_dir
    model_type = "avatar-v1.5" if args.model_type == "avatar-v1.6" else args.model_type
    use_loudness_norm = args.model_type != "avatar-v1.6"
    use_distill = args.use_distill
    use_int8 = args.use_int8

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
    negative_prompt = "Close-up, bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
    left_raw_speech_path = input_data['cond_audio'].get('person1', None)
    right_raw_speech_path = input_data['cond_audio'].get('person2', None)
    assert left_raw_speech_path is not None or right_raw_speech_path is not None, f"At least one speech is required."
    left_person_bbox, right_person_bbox = None, None
    use_background_silent_audio = False
    if 'bbox' in input_data:
        # bbox: [left_y_min, left_x_min, left_y_max, left_x_max]
        # x and y coordinates correspond to the width and height dimensions, respectively
        left_person_bbox = input_data['bbox'].get('person1', None)
        right_person_bbox = input_data['bbox'].get('person2', None)
        other_person_bbox = input_data['bbox'].get('others', None)
        use_background_silent_audio = other_person_bbox is not None and len(other_person_bbox) > 0
    audio_type = 'para'
    if 'audio_type' in input_data:
        audio_type = input_data.get('audio_type', 'para')

    # prepare distributed environment
    rank = int(os.environ['RANK'])
    num_gpus = torch.cuda.device_count()
    local_rank = rank % num_gpus
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=3600*24))
    global_rank    = dist.get_rank()
    num_processes  = dist.get_world_size()

    # initialize context parallel
    context_parallel_util.init_context_parallel(context_parallel_size=context_parallel_size, global_rank=global_rank, world_size=num_processes)
    cp_rank = context_parallel_util.get_cp_rank()
    cp_size = context_parallel_util.get_cp_size()
    cp_split_hw = context_parallel_util.get_optimal_split(cp_size)

    # initialize models
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(checkpoint_dir, '..', 'LongCat-Video'), subfolder="tokenizer", torch_dtype=torch.bfloat16)
    text_encoder = UMT5EncoderModel.from_pretrained(os.path.join(checkpoint_dir, '..', 'LongCat-Video'), subfolder="text_encoder", torch_dtype=torch.bfloat16)
    vae = AutoencoderKLWan.from_pretrained(os.path.join(checkpoint_dir, '..', 'LongCat-Video'), subfolder="vae", torch_dtype=torch.bfloat16)
    if model_type == "avatar-v1.0":
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(os.path.join(checkpoint_dir, '..', 'LongCat-Video'), subfolder="scheduler", torch_dtype=torch.bfloat16)
    elif model_type == "avatar-v1.5":
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(checkpoint_dir, subfolder="scheduler", torch_dtype=torch.bfloat16)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}. Expected 'avatar-v1.0' or 'avatar-v1.5'.")
    
    if model_type == "avatar-v1.0":
        dit = LongCatVideoAvatarTransformer3DModel.from_pretrained(checkpoint_dir, subfolder="avatar_multi", cp_split_hw=cp_split_hw, torch_dtype=torch.bfloat16)
    elif model_type == "avatar-v1.5":
        if use_int8:
            print("[INFO] Loading INT8 quantized DiT model...")
            dit = load_quantized_dit(checkpoint_dir, subfolder="base_model_int8", cp_split_hw=cp_split_hw)
        else:
            dit = LongCatVideoAvatarTransformer3DModel.from_pretrained(checkpoint_dir, subfolder="base_model", cp_split_hw=cp_split_hw, torch_dtype=torch.bfloat16)
        dit = dit.to(local_rank)
        if use_distill:
            if args.mix_steps > 0:
                dmd_lora_path = os.path.join(checkpoint_dir, "loraA", "lora.safetensors")
                opsd_lora_path = os.path.join(checkpoint_dir, "loraB", "lora.safetensors")
                missing_paths = [
                    path
                    for path in (dmd_lora_path, opsd_lora_path)
                    if not os.path.isfile(path)
                ]
                if missing_paths:
                    raise FileNotFoundError(f"Mixing LoRA not found: {missing_paths}")
                print(f"[INFO] Loading mixed LoRAs: DMD={dmd_lora_path}, OPSD={opsd_lora_path}")
                dit.load_lora(dmd_lora_path, "dmd", multiplier=1.0, lora_network_dim=128, lora_network_alpha=64)
                dit.load_lora(opsd_lora_path, "opsd", multiplier=1.0, lora_network_dim=128, lora_network_alpha=64)
                dit.enable_lora_mix("dmd", "opsd", args.mix_alpha, args.mix_steps)
            else:
                distill_checkpoint_path = (
                    args.lora_path if os.path.isabs(args.lora_path)
                    else os.path.join(checkpoint_dir, args.lora_path)
                )
                if not os.path.isfile(distill_checkpoint_path):
                    raise FileNotFoundError(f"Distillation LoRA not found: {distill_checkpoint_path}")
                print(f"[INFO] Statically merging distillation LoRA: {distill_checkpoint_path}")
                merged_count = dit.merge_lora(distill_checkpoint_path, lora_key="dmd", multiplier=args.lora_multiplier, lora_network_dim=128, lora_network_alpha=64)
                print(f"[INFO] Statically merged LoRA into {merged_count} Linear modules: multiplier={args.lora_multiplier}")

    if args.use_fp8:
        fp8_blocks = enable_fp8_ffn(dit)
        if local_rank == 0:
            print(f"[INFO] TE FP8 enabled for FFN w1/w3 in blocks {list(fp8_blocks)}")

    if args.compile_dit:
        torch._dynamo.config.cache_size_limit = 128
        torch._dynamo.config.accumulated_cache_size_limit = 1024
        if local_rank == 0:
            print("[INFO] Compiling DiT transformer blocks (CUDA graphs off)")
        for block_idx, block in enumerate(dit.blocks):
            dit.blocks[block_idx] = torch.compile(
                block,
                fullgraph=False,
                dynamic=False,
                options={"triton.cudagraphs": False, "epilogue_fusion": False, "emulate_precision_casts": True},
            )

    # initialize audio models
    if model_type == "avatar-v1.0":
        audio_model_checkpoint_path = os.path.join(checkpoint_dir, 'chinese-wav2vec2-base')
    elif model_type == "avatar-v1.5":
        audio_model_checkpoint_path = os.path.join(checkpoint_dir, 'whisper-large-v3')
    audio_encoder = get_audio_encoder(audio_model_checkpoint_path, model_type).to(local_rank)
    audio_feature_extractor = get_audio_feature_extractor(audio_model_checkpoint_path, model_type)

    vocal_separator_path = os.path.join(checkpoint_dir, 'vocal_separator/Kim_Vocal_2.onnx')
    audio_output_dir_temp = f"./audio_temp_file"
    os.makedirs(audio_output_dir_temp, exist_ok=True)
    audio_output_dir_temp = Path(audio_output_dir_temp)
    audio_separator_model_path = os.path.dirname(vocal_separator_path)
    audio_separator_model_name = os.path.basename(vocal_separator_path)
    vocal_separator = Separator(
        output_dir=audio_output_dir_temp / "vocals",
        output_single_stem="vocals",
        model_file_dir=audio_separator_model_path,
    )
    if vocal_separator.onnx_execution_provider == ["CUDAExecutionProvider"]:
        vocal_separator.onnx_execution_provider = [
            (
                "CUDAExecutionProvider",
                {"device_id": local_rank, "cudnn_conv_algo_search": "HEURISTIC"},
            )
        ]
    vocal_separator.load_model(audio_separator_model_name)

    
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

    if args.compile_vae:
        if local_rank == 0:
            print("[INFO] Compiling optimized VAE encoder and decoder (cuDNN autotune on, CUDA graphs off)")
        torch.backends.cudnn.benchmark = True
        pipe.vae.enable_encoder_fused_padding()
        pipe.vae.enable_encoder_cache_views()
        pipe.vae.enable_decoder_fused_padding()
        pipe.vae.enable_decoder_cache_views()
        pipe.vae.encoder = torch.compile(
            pipe.vae.encoder,
            fullgraph=False,
            dynamic=False,
            options={"triton.cudagraphs": False},
        )
        pipe.vae.decoder = torch.compile(
            pipe.vae.decoder,
            fullgraph=False,
            dynamic=False,
            options={"triton.cudagraphs": False},
        )

    global_seed = 42
    seed = global_seed + global_rank

    generator = torch.Generator(device=local_rank)
    generator.manual_seed(seed)

    if cp_rank == 0:
        # extract vocal
        sr = 16000
        generate_duration = num_frames / save_fps + (num_segments-1) * (num_frames-num_cond_frames) / save_fps
        audio_process_duration = generate_duration + 1.0
        left_speech_array = extract_vocal_from_speech(left_raw_speech_path, vocal_separator, audio_process_duration)
        right_speech_array = extract_vocal_from_speech(right_raw_speech_path, vocal_separator, audio_process_duration)
        
        # prepare each vocal and synthesize the sum audio
        left_speech_array_ext, right_speech_array_ext, merge_speech = audio_prepare_multi(left_speech_array, right_speech_array, audio_process_duration, \
                                                                                            left_raw_speech_path, right_raw_speech_path, sample_rate=sr, audio_type=audio_type)
        merge_speech_path = f"/tmp/temp_speech_{generate_random_uid()}_{global_rank}_merge.wav"
        sf.write(merge_speech_path, merge_speech, 16000)

        left_full_audio_emb = pipe.get_audio_embedding(
            left_speech_array_ext, fps=save_fps*audio_stride, device=local_rank, sample_rate=sr,
            model_type=model_type, use_loudness_norm=use_loudness_norm
        )
        right_full_audio_emb = pipe.get_audio_embedding(
            right_speech_array_ext, fps=save_fps*audio_stride, device=local_rank, sample_rate=sr,
            model_type=model_type, use_loudness_norm=use_loudness_norm
        )
        if torch.isnan(left_full_audio_emb).any() or torch.isnan(right_full_audio_emb).any():
            raise ValueError(f"broken audio embedding with nan values")
        if use_background_silent_audio:
            back_full_audio_emb = pipe.get_audio_embedding(
                np.zeros_like(left_speech_array_ext), fps=save_fps*audio_stride, device=local_rank,
                sample_rate=sr, model_type=model_type, use_loudness_norm=use_loudness_norm
            )
        assert left_full_audio_emb.shape == right_full_audio_emb.shape, f"Inconsistent audio embedding shape."
        if use_background_silent_audio:
            assert left_full_audio_emb.shape == back_full_audio_emb.shape, f"Inconsistent audio embedding shape between speaker and background."
  
        if context_parallel_util.get_cp_size() > 1:
            full_audio_emb_shape_list = list(left_full_audio_emb.size())
            full_audio_emb_tensor_shape_list = torch.tensor(full_audio_emb_shape_list, dtype=torch.int64, device=left_full_audio_emb.device)
            context_parallel_util.cp_broadcast(full_audio_emb_tensor_shape_list)
            context_parallel_util.cp_broadcast(left_full_audio_emb)
            context_parallel_util.cp_broadcast(right_full_audio_emb)
            if use_background_silent_audio:
                context_parallel_util.cp_broadcast(back_full_audio_emb)
        

    elif context_parallel_util.get_cp_size() > 1:
        full_audio_emb_tensor_shape_list = torch.zeros(3, dtype=torch.int64, device=local_rank)
        context_parallel_util.cp_broadcast(full_audio_emb_tensor_shape_list)
        full_audio_emb_shape_list = full_audio_emb_tensor_shape_list.tolist()
        left_full_audio_emb = torch.zeros(*full_audio_emb_shape_list, dtype=torch.float32, device=local_rank)
        context_parallel_util.cp_broadcast(left_full_audio_emb)
        right_full_audio_emb = torch.zeros(*full_audio_emb_shape_list, dtype=torch.float32, device=local_rank)
        context_parallel_util.cp_broadcast(right_full_audio_emb)
        if use_background_silent_audio:
            back_full_audio_emb = torch.zeros(*full_audio_emb_shape_list, dtype=torch.float32, device=local_rank)
            context_parallel_util.cp_broadcast(back_full_audio_emb)


    indices = torch.arange(2 * 2 + 1) - 2
    audio_start_idx = 0
    audio_end_idx = audio_start_idx + audio_stride * num_frames

    # get audio embedding for the first clip
    center_indices = torch.arange(audio_start_idx, audio_end_idx, audio_stride).unsqueeze(1) + indices.unsqueeze(0)
    center_indices = torch.clamp(center_indices, min=0, max=left_full_audio_emb.shape[0]-1)
    left_audio_emb = left_full_audio_emb[center_indices][None,...].to(local_rank)
    right_audio_emb = right_full_audio_emb[center_indices][None,...].to(local_rank)
    audio_embs = [left_audio_emb, right_audio_emb]
    if use_background_silent_audio:
        audio_embs.append(back_full_audio_emb[center_indices][None,...].to(local_rank))
    audio_embs = torch.cat(audio_embs)


    # ==============================
    #          ai2v (480P)
    # ==============================
    image_path = input_data['cond_image']
    image = load_image(image_path)
    (src_width, src_height) = image.size


    # define human / background mask
    background_mask = torch.zeros([src_height, src_width])
    human_mask1 = torch.zeros([src_height, src_width])
    human_mask2 = torch.zeros([src_height, src_width])
    if left_person_bbox is None and right_person_bbox is None:    
        face_scale = 0.1
        left_y_min, left_y_max = int(src_height * face_scale), int(src_height * (1 - face_scale))
        right_y_min, right_y_max = left_y_min, left_y_max
        src_width = src_width // 2
        left_x_min, left_x_max = int(src_width * face_scale), int(src_width * (1 - face_scale))
        right_x_min, right_x_max = int(src_width * face_scale + src_width), int(src_width * (1 - face_scale) + src_width)
    elif left_person_bbox is not None and right_person_bbox is not None:
        left_y_min, left_x_min, left_y_max, left_x_max = left_person_bbox
        right_y_min, right_x_min, right_y_max, right_x_max = right_person_bbox
    else:
        raise NotImplementedError(f"Not supported bbox type.")
    human_mask1[left_y_min:left_y_max, left_x_min:left_x_max] = 1
    human_mask2[right_y_min:right_y_max, right_x_min:right_x_max] = 1
    background_mask += human_mask1
    background_mask += human_mask2
    background_mask = torch.where(background_mask > 0, torch.tensor(0), torch.tensor(1))
    total_mask = [human_mask1, human_mask2, background_mask]
    if use_background_silent_audio:
        for i in range(len(other_person_bbox)//4):
            other_person_bbox_i = other_person_bbox[i*4:(i+1)*4]
            other_person_mask = torch.zeros([src_height, src_width])
            other_person_mask[other_person_bbox_i[0]:other_person_bbox_i[2], other_person_bbox_i[1]:other_person_bbox_i[3]] = 1
            total_mask.append(other_person_mask)
    ref_target_masks = torch.stack(total_mask, dim=0).to(local_rank)
    
    # generate video
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
        audio_emb=audio_embs,
        ref_target_masks=ref_target_masks,
        use_distill=use_distill,
    )
    output, latent = output_tuple 
    output = output[0]
    video_array = (output * 255).astype(np.uint8)
    video = [PIL.Image.fromarray(img) for img in video_array]

    if cp_rank == 0 and (args.save_every_segment or num_segments == 1):
        output_tensor = torch.from_numpy(video_array)
        save_video_ffmpeg(output_tensor, os.path.join(output_dir, "ai2v_demo_1"), merge_speech_path, fps=save_fps, quality=10)
    del output
    torch_gc()

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
    total_output_frames = len(video_array) + max(0, num_segments - 1) * (num_frames - num_cond_frames)
    all_generated_frames = np.empty((total_output_frames, height, width, 3), dtype=np.uint8)
    generated_frame_count = len(video_array)
    all_generated_frames[:generated_frame_count] = video_array

    for segment_idx in range(1, num_segments):
        if local_rank == 0:
            print(f"Generating segment {segment_idx+1}/{num_segments}...")
        
        # prepare audio embedding for the next clip
        audio_start_idx = audio_start_idx + audio_stride * (num_frames - num_cond_frames)
        audio_end_idx   = audio_start_idx + audio_stride * num_frames
        center_indices = torch.arange(audio_start_idx, audio_end_idx, audio_stride).unsqueeze(1) + indices.unsqueeze(0)
        center_indices = torch.clamp(center_indices, min=0, max=left_full_audio_emb.shape[0]-1)
        left_audio_emb = left_full_audio_emb[center_indices][None,...].to(local_rank)
        right_audio_emb = right_full_audio_emb[center_indices][None,...].to(local_rank)
        audio_embs = [left_audio_emb, right_audio_emb]
        if use_background_silent_audio:
            audio_embs.append(back_full_audio_emb[center_indices][None,...].to(local_rank))
        audio_embs = torch.cat(audio_embs)
        
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
            use_kv_cache=args.mix_steps == 0,
            offload_kv_cache=False,
            enhance_hf=True if not use_distill else False,
            audio_emb=audio_embs,
            ref_latent=ref_latent,
            ref_img_index=ref_img_index,
            mask_frame_range=mask_frame_range,
            ref_target_masks=ref_target_masks,
            use_distill=use_distill,
        )
        output, latent = output_tuple

        output = output[0]
        new_video_array = (output * 255).astype(np.uint8)
        new_video = [PIL.Image.fromarray(img) for img in new_video_array]
        del output

        appended_frames = new_video_array[num_cond_frames:]
        next_frame_count = generated_frame_count + len(appended_frames)
        all_generated_frames[generated_frame_count:next_frame_count] = appended_frames
        generated_frame_count = next_frame_count

        current_video = new_video

        if cp_rank == 0 and (args.save_every_segment or segment_idx == num_segments - 1):
            output_tensor = torch.from_numpy(all_generated_frames[:generated_frame_count])
            save_video_ffmpeg(output_tensor, os.path.join(output_dir, f"video_continue_{segment_idx+1}"), merge_speech_path, fps=save_fps, quality=10)
            del output_tensor
    
    if cp_rank == 0 and os.path.exists(merge_speech_path):
        os.remove(merge_speech_path)



def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_json',
        type=str,
        default='assets/avatar/multi_example_1.json',
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./outputs_avatar_multi'
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
        '--save_every_segment',
        action='store_true',
        help='Save cumulative video after every segment; by default only the final video is saved.',
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
        default=0
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
        "--lora_path",
        type=str,
        default="lora/dmd_lora.safetensors",
        help="Distillation LoRA path. Relative paths are resolved under checkpoint_dir.",
    )
    parser.add_argument(
        "--lora_multiplier",
        type=float,
        default=0.8,
        help="Multiplier used when statically merging the distillation LoRA.",
    )
    parser.add_argument(
        "--mix_steps",
        type=int,
        default=3,
        help="Leading denoising steps using alpha*DMD + (1-alpha)*OPSD for avatar-v1.6.",
    )
    parser.add_argument(
        "--mix_alpha",
        type=float,
        default=0.5,
        help="DMD weight used during mixed denoising for avatar-v1.6.",
    )
    parser.add_argument(
        "--use_int8",
        action='store_true',
        help="Load INT8 quantized DiT model for reduced VRAM usage"
    )
    parser.add_argument(
        "--use_fp8",
        action="store_true",
        help="Use calibrated Transformer Engine FP8 for selected DiT FFN projections.",
    )
    parser.add_argument(
        "--compile_dit",
        action="store_true",
        help="Compile DiT transformer blocks for lower denoising latency.",
    )
    parser.add_argument(
        "--compile_vae",
        action="store_true",
        help="Compile the VAE encoder and decoder with cuDNN autotuning, fused padding, and zero-copy cache views.",
    )

    args = parser.parse_args()
    if args.use_fp8 and args.use_int8:
        parser.error("--use_fp8 cannot be combined with --use_int8")
    if args.use_fp8 and args.model_type != "avatar-v1.6":
        parser.error("--use_fp8 is calibrated only for --model_type avatar-v1.6")
    if args.model_type in ("avatar-v1.0", "avatar-v1.5"):
        args.mix_steps = 0
        args.mix_alpha = 0.0
    elif not 0 <= args.mix_steps <= 8:
        parser.error("--mix_steps must be in [0, 8]")
    elif not 0.0 <= args.mix_alpha <= 1.0:
        parser.error("--mix_alpha must be in [0, 1]")
    if args.model_type == "avatar-v1.6" and args.mix_steps > 0 and not args.use_distill:
        parser.error("--mix_steps requires --use_distill")

    return args


if __name__ == "__main__":
    args = _parse_args()
    generate(args)
