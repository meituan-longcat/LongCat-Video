import os
import json
import librosa
import binascii
import imageio
import subprocess
import numpy as np
import os.path as osp
import pyloudnorm as pyln
from einops import rearrange
import scipy.signal as ss

import torch
import torch.nn.functional as F
import torchvision

from ..context_parallel import context_parallel_util


def torch_gc():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


def linear_interpolation(features, seq_len):
    features = features.transpose(1, 2)
    output_features = F.interpolate(features, size=seq_len, align_corners=True, mode='linear')
    return output_features.transpose(1, 2)


@torch.compile
def calculate_x_ref_attn_map(noise_q, ref_k, ref_target_masks, attn_bias=None):

    # compute cross-reference attention maps between query features and reference key features.
    ref_k = ref_k.to(noise_q.dtype).to(noise_q.device)
    scale = 1.0 / noise_q.shape[-1] ** 0.5
    noise_q = noise_q * scale
    noise_q = noise_q.transpose(1, 2)
    ref_k = ref_k.transpose(1, 2)
    attn = noise_q @ ref_k.transpose(-2, -1)

    if attn_bias is not None:
        attn = attn + attn_bias

    x_ref_attn_map_source = attn.softmax(-1)

    x_ref_attn_maps = []
    ref_target_masks = ref_target_masks.to(noise_q.dtype)
    x_ref_attn_map_source = x_ref_attn_map_source.to(noise_q.dtype)

    for _, ref_target_mask in enumerate(ref_target_masks):
        torch_gc()
        ref_target_mask = ref_target_mask[None, None, None, ...]
        x_ref_attn_map = x_ref_attn_map_source.clone()
        x_ref_attn_map = x_ref_attn_map * ref_target_mask
        x_ref_attn_map = x_ref_attn_map.sum(-1) / ref_target_mask.sum() 
        x_ref_attn_map = x_ref_attn_map.permute(0, 2, 1) 
        x_ref_attn_map = x_ref_attn_map.mean(-1) 
        
        x_ref_attn_maps.append(x_ref_attn_map)
    
    del attn
    del x_ref_attn_map_source
    torch_gc()

    return torch.concat(x_ref_attn_maps, dim=0)


def get_attn_map_with_target(noise_q, key, shape, ref_target_masks=None, split_num=2, cp_split_hw=None):
    
    N_t, N_h, N_w = shape
    x_seqlens = N_h * N_w
    if cp_split_hw[0] * cp_split_hw[1] > 1:
        cp_size = context_parallel_util.get_cp_size()
        (split_h, split_w) = (cp_size, 1) if cp_split_hw is None else cp_split_hw

        assert N_h % split_h == 0, N_w % split_w == 0

        N_h_ = N_h // split_h
        N_w_ = N_w // split_w
        x_seqlens = N_h_ * N_w_
    
    ref_k = key[:, :x_seqlens] # reference image size after cp split
    noise_q = noise_q.contiguous()

    # gather reference image key features
    if cp_split_hw[0] * cp_split_hw[1] > 1:
        _, _, H, _ = ref_k.shape
        ref_k = ref_k.permute(0, 2, 1, 3)
        ref_k = rearrange(ref_k, "b h m k -> b (h m) k")
        ref_k = context_parallel_util.gather_cp_2d(ref_k, shape=(H, N_h, N_w), split_hw=cp_split_hw) 
        ref_k = rearrange(ref_k, "b (h m) k -> b h m k", h=H) 
        ref_k = ref_k.permute(0, 2, 1, 3)

    _, seq_lens, heads, _ = noise_q.shape
    class_num, _ = ref_target_masks.shape
    x_ref_attn_maps = torch.zeros(class_num, seq_lens).to(noise_q.device).to(noise_q.dtype)

    split_chunk = heads // split_num
    
    # calculate attn map within each group and take the mean
    for i in range(split_num):
        x_ref_attn_maps_perhead = calculate_x_ref_attn_map(noise_q[:, :, i*split_chunk:(i+1)*split_chunk, :], ref_k[:, :, i*split_chunk:(i+1)*split_chunk, :], ref_target_masks)
        x_ref_attn_maps += x_ref_attn_maps_perhead
    
    return x_ref_attn_maps / split_num


def rand_name(length=8, suffix=''):
    name = binascii.b2a_hex(os.urandom(length)).decode('utf-8')
    if suffix:
        if not suffix.startswith('.'):
            suffix = '.' + suffix
        name += suffix
    return name

def cache_video(tensor,
                save_file=None,
                fps=30,
                suffix='.mp4',
                nrow=8,
                normalize=True,
                value_range=(-1, 1),
                retry=5):
    
    # cache file
    cache_file = osp.join('/tmp', rand_name(
        suffix=suffix)) if save_file is None else save_file

    # save to cache
    error = None
    for _ in range(retry):
       
        # preprocess
        tensor = tensor.clamp(min(value_range), max(value_range))
        tensor = torch.stack([
                torchvision.utils.make_grid(
                    u, nrow=nrow, normalize=normalize, value_range=value_range)
                for u in tensor.unbind(2)
            ],
                                 dim=1).permute(1, 2, 3, 0)
        tensor = (tensor * 255).type(torch.uint8).cpu()

        # write video
        writer = imageio.get_writer(cache_file, fps=fps, codec='libx264', quality=10, ffmpeg_params=["-crf", "10"])
        for frame in tensor.numpy():
            writer.append_data(frame)
        writer.close()
        return cache_file

def get_audio_duration(audio_path):
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_entries", "format=duration",
        audio_path,
    ]
    out = subprocess.check_output(cmd)
    info = json.loads(out)
    return float(info["format"]["duration"])


def save_video_ffmpeg(gen_video_samples, save_path, audio_path, fps=25, quality=10, high_quality_save=False):
    video = gen_video_samples.cpu().numpy()
    if video.dtype != np.uint8:
        video = np.clip(video, 0, 255).astype(np.uint8)
    video = np.ascontiguousarray(video)
    frame_count, height, width, channels = video.shape
    if channels != 3 or frame_count == 0:
        raise ValueError(f"Expected non-empty RGB video, got shape {video.shape}")

    output_path = save_path + ".mp4"
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    if high_quality_save:
        video_options = ["-c:v", "libx264", "-crf", "0", "-preset", "veryslow"]
    else:
        video_options = ["-c:v", "libx264", "-crf", "16", "-preset", "medium"]

    command = [
        "ffmpeg", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-video_size", f"{width}x{height}", "-framerate", str(fps),
        "-i", "pipe:0", "-i", audio_path,
        "-t", f"{frame_count / fps:.9f}",
        "-map", "0:v:0", "-map", "1:a:0",
        *video_options, "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-shortest", output_path,
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    try:
        for start in range(0, frame_count, 16):
            process.stdin.write(memoryview(video[start:start + 16]).cast("B"))
        process.stdin.close()
        return_code = process.wait()
    except BaseException:
        process.kill()
        process.wait()
        raise
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)
