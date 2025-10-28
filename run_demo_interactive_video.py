import os
import argparse
import datetime
import PIL.Image
import numpy as np

import torch
import torch.distributed as dist

from transformers import AutoTokenizer, UMT5EncoderModel
from torchvision.io import write_video

from longcat_video.pipeline_longcat_video import LongCatVideoPipeline
from longcat_video.modules.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from longcat_video.modules.autoencoder_kl_wan import AutoencoderKLWan
from longcat_video.modules.longcat_video_dit import LongCatVideoTransformer3DModel
from longcat_video.context_parallel import context_parallel_util
from longcat_video.context_parallel.context_parallel_util import init_context_parallel


def torch_gc():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

def generate(args):
    # case setup
    prompt_list = [
        "The kitchen is bright and airy, featuring white cabinets and a wooden countertop. A loaf of freshly baked bread rests on a cutting board, and a glass and a carton of milk are positioned nearby. A woman wearing a floral apron stands at the wooden countertop, skillfully slicing a golden-brown loaf of bread with a sharp knife. The bread is resting on a cutting board, and crumbs scatter around as she cuts.",
        "The woman puts down the knife in her hand, reaches for the carton of milk and then pours it into the glass on the table.",
        "The woman puts down the milk carton.",
        "The woman picks up the glass of milk and takes a sip."
    ]
    
    
    negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
    num_segments = len(prompt_list) - 1  # 1 minute video
    num_frames = 93
    num_cond_frames = 13

    # load parsed args
    checkpoint_dir = args.checkpoint_dir
    context_parallel_size = args.context_parallel_size
    enable_compile = args.enable_compile

    # prepare distributed environment
    rank = int(os.environ['RANK'])
    num_gpus = torch.cuda.device_count()
    local_rank = rank % num_gpus
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=3600*24))
    global_rank    = dist.get_rank()
    num_processes  = dist.get_world_size()

    # initialize context parallel before loading models
    init_context_parallel(context_parallel_size=context_parallel_size, global_rank=global_rank, world_size=num_processes)
    cp_size = context_parallel_util.get_cp_size()
    cp_split_hw = context_parallel_util.get_optimal_split(cp_size)

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, subfolder="tokenizer", torch_dtype=torch.bfloat16)
    text_encoder = UMT5EncoderModel.from_pretrained(checkpoint_dir, subfolder="text_encoder", torch_dtype=torch.bfloat16)
    vae = AutoencoderKLWan.from_pretrained(checkpoint_dir, subfolder="vae", torch_dtype=torch.bfloat16)
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(checkpoint_dir, subfolder="scheduler", torch_dtype=torch.bfloat16)
    dit = LongCatVideoTransformer3DModel.from_pretrained(checkpoint_dir, subfolder="dit", cp_split_hw=cp_split_hw, torch_dtype=torch.bfloat16)

    if enable_compile:
        dit = torch.compile(dit)

    pipe = LongCatVideoPipeline(
        tokenizer = tokenizer,
        text_encoder = text_encoder,
        vae = vae,
        scheduler = scheduler,
        dit = dit,
    )
    pipe.to(local_rank)

    global_seed = 42
    seed = global_seed + global_rank

    generator = torch.Generator(device=local_rank)
    generator.manual_seed(seed)

    ### t2v (480p)
    output = pipe.generate_t2v(
        prompt=prompt_list[0],
        negative_prompt=negative_prompt,
        height=480,
        width=832,
        num_frames=num_frames,
        num_inference_steps=50,
        guidance_scale=4.0,
        generator=generator,
    )[0]

    if local_rank == 0:
        output_tensor = torch.from_numpy(np.array(output))
        output_tensor = (output_tensor * 255).clamp(0, 255).to(torch.uint8)
        write_video(f"output_interactive_0.mp4", output_tensor, fps=15, video_codec="libx264", options={"crf": f"{18}"})

    video = [(output[i] * 255).astype(np.uint8) for i in range(output.shape[0])]
    video = [PIL.Image.fromarray(img) for img in video]
    del output 
    torch_gc()

    target_size = video[0].size
    current_video = video

    ### long video
    all_generated_frames = video
    for segment_idx in range(num_segments):
        if local_rank == 0:
            print(f"Generating segment {segment_idx + 1}/{num_segments}...")
        
        output = pipe.generate_vc(
            video=current_video,
            prompt=prompt_list[segment_idx + 1],
            negative_prompt=negative_prompt,
            resolution='480p', # 480p / 720p
            num_frames=num_frames,
            num_cond_frames=num_cond_frames,
            num_inference_steps=50,
            guidance_scale=4.0,
            generator=generator,
            use_kv_cache=True,
            offload_kv_cache=False,
            enhance_hf=True
        )[0]

        new_video = [(output[i] * 255).astype(np.uint8) for i in range(output.shape[0])]
        new_video = [PIL.Image.fromarray(img) for img in new_video]
        new_video = [frame.resize(target_size, PIL.Image.BICUBIC) for frame in new_video]
        del output

        all_generated_frames.extend(new_video[num_cond_frames:])

        current_video = new_video

        if local_rank == 0:
            output_tensor = torch.from_numpy(np.array(all_generated_frames))
            write_video(f"output_interactive_{segment_idx+1}.mp4", output_tensor, fps=15, video_codec="libx264", options={"crf": f"{18}"})
            del output_tensor

    ### long video refinement (720p)
    refinement_lora_path = os.path.join(checkpoint_dir, 'lora/refinement_lora.safetensors')
    pipe.dit.load_lora(refinement_lora_path, 'refinement_lora')
    pipe.dit.enable_loras(['refinement_lora'])
    pipe.dit.enable_bsa()

    if enable_compile:
        dit = torch.compile(dit)    

    torch_gc()
    cur_condition_video = None
    cur_num_cond_frames = 0
    start_id = 0
    all_refine_frames = []

    for segment_idx in range(num_segments+1):
        if local_rank == 0:
            print(f"Refine segment {segment_idx + 1}/{num_segments}...")

        output_refine = pipe.generate_refine(
            video=cur_condition_video,
            prompt='',
            stage1_video=all_generated_frames[start_id:start_id+num_frames],
            num_cond_frames=cur_num_cond_frames,
            num_inference_steps=50,
            generator=generator,
        )[0]

        new_video = [(output_refine[i] * 255).astype(np.uint8) for i in range(output_refine.shape[0])]
        new_video = [PIL.Image.fromarray(img) for img in new_video]
        del output_refine

        all_refine_frames.extend(new_video[cur_num_cond_frames:])
        cur_condition_video = new_video
        cur_num_cond_frames = num_cond_frames * 2
        start_id = start_id + num_frames - num_cond_frames
        
        if local_rank == 0:
            output_tensor = torch.from_numpy(np.array(all_refine_frames))
            write_video(f"output_interactive_refine_{segment_idx}.mp4", output_tensor, fps=30, video_codec="libx264", options={"crf": f"{10}"})

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--context_parallel_size",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
    )
    parser.add_argument(
        '--enable_compile',
        action='store_true',
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = _parse_args()
    generate(args)