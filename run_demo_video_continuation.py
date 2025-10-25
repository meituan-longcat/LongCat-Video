import os
import argparse

import cv2
import datetime
import PIL.Image
import numpy as np

import torch
import torch.distributed as dist

from transformers import AutoTokenizer, UMT5EncoderModel
from torchvision.io import write_video
from diffusers.utils import load_video

from longcat_video.pipeline_longcat_video import LongCatVideoPipeline
from longcat_video.modules.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from longcat_video.modules.autoencoder_kl_wan import AutoencoderKLWan
from longcat_video.modules.longcat_video_dit import LongCatVideoTransformer3DModel
from longcat_video.context_parallel import context_parallel_util
from longcat_video.context_parallel.context_parallel_util import init_context_parallel


def torch_gc():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


def get_fps(video_path):
    cap = cv2.VideoCapture(video_path)
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    
    return original_fps


def generate(args):
    # case setup
    video_path = "assets/motorcycle.mp4"
    video = load_video(video_path)
    prompt = "A person rides a motorcycle along a long, straight road that stretches between a body of water and a forested hillside. The rider steadily accelerates, keeping the motorcycle centered between the guardrails, while the scenery passes by on both sides. The video captures the journey from the rider’s perspective, emphasizing the sense of motion and adventure."
    negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
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

    target_fps = 15
    target_size = video[0].size  # (width, height)
    current_fps = get_fps(video_path)
    stride = max(1, round(current_fps / target_fps))

    ### vc (480p)
    output = pipe.generate_vc(
        video=video[::stride],
        prompt=prompt,
        negative_prompt=negative_prompt,
        resolution='480p', # 480p / 720p
        num_frames=93,
        num_cond_frames=num_cond_frames,
        num_inference_steps=50,
        guidance_scale=4.0,
        generator=generator,
        use_kv_cache=True,
        offload_kv_cache=False,
    )[0]

    if local_rank == 0:
        output = [(output[i] * 255).astype(np.uint8) for i in range(output.shape[0])]
        output = [PIL.Image.fromarray(img) for img in output]
        output = [frame.resize(target_size, PIL.Image.BICUBIC) for frame in output]

        output = video[::stride] + output[num_cond_frames:]
        output_tensor = torch.from_numpy(np.array(output))
        write_video("output_vc.mp4", output_tensor, fps=15, video_codec="libx264", options={"crf": f"{18}"})
    del output
    torch_gc()

    ### vc distill (480p)
    cfg_step_lora_path = os.path.join(checkpoint_dir, 'lora/cfg_step_lora.safetensors')
    pipe.dit.load_lora(cfg_step_lora_path, 'cfg_step_lora')
    pipe.dit.enable_loras(['cfg_step_lora'])

    if enable_compile:
        dit = torch.compile(dit)

    output_distill = pipe.generate_vc(
        video=video[::stride],
        prompt=prompt,
        resolution='480p', # 480p / 720p
        num_frames=93,
        num_cond_frames=num_cond_frames,
        num_inference_steps=16,
        use_distill=True,
        guidance_scale=1.0,
        generator=generator,
        use_kv_cache=True,
        offload_kv_cache=False,
        enhance_hf=False,
    )[0]
    pipe.dit.disable_all_loras()

    if local_rank == 0:
        output_processed = [(output_distill[i] * 255).astype(np.uint8) for i in range(output_distill.shape[0])]
        output_processed = [PIL.Image.fromarray(img) for img in output_processed]
        output_processed = [frame.resize(target_size, PIL.Image.BICUBIC) for frame in output_processed]

        output = video[::stride] + output_processed[num_cond_frames:]
        output_tensor = torch.from_numpy(np.array(output))
        write_video("output_vc_distill.mp4", output_tensor, fps=15, video_codec="libx264", options={"crf": f"{18}"})

    ### vc refinement (720p)
    refinement_lora_path = os.path.join(checkpoint_dir, 'lora/refinement_lora.safetensors')
    pipe.dit.load_lora(refinement_lora_path, 'refinement_lora')
    pipe.dit.enable_loras(['refinement_lora'])
    pipe.dit.enable_bsa()

    if enable_compile:
        dit = torch.compile(dit)
    
    stage1_video = [(output_distill[i] * 255).astype(np.uint8) for i in range(output_distill.shape[0])]
    stage1_video = [PIL.Image.fromarray(img) for img in stage1_video]
    del output_distill 
    torch_gc()

    target_fps = 30
    stride = max(1, round(current_fps / target_fps))

    output_refine = pipe.generate_refine(
        video=video[::stride],
        prompt=prompt,
        stage1_video=stage1_video,
        num_cond_frames=num_cond_frames*2,
        num_inference_steps=50,
        generator=generator,
    )[0]

    pipe.dit.disable_all_loras()
    pipe.dit.disable_bsa()

    if local_rank == 0:
        output_refine = [(output_refine[i] * 255).astype(np.uint8) for i in range(output_refine.shape[0])]
        output_refine = [PIL.Image.fromarray(img) for img in output_refine]
        output_refine = [frame.resize(target_size, PIL.Image.BICUBIC) for frame in output_refine]
        output_refine = video[::stride] + output_refine[num_cond_frames*2:]

        output_tensor = torch.from_numpy(np.array(output_refine))
        write_video("output_vc_refine.mp4", output_tensor, fps=30, video_codec="libx264", options={"crf": f"{10}"})


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