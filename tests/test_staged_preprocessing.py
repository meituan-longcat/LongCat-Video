import os
import tempfile
from types import SimpleNamespace

import unittest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from longcat_video.distributed_utils import broadcast_tensor_mapping
from longcat_video.pipeline_longcat_video_avatar import LongCatVideoAvatarPipeline
from run_demo_avatar_single_audio_to_video import is_preprocessing_rank


class _Movable(torch.nn.Module):
    pass


class _Tokenizer:
    def __call__(self, prompt, **kwargs):
        return SimpleNamespace(
            input_ids=torch.tensor([[1, 2, 3]]),
            attention_mask=torch.tensor([[1, 1, 1]]),
        )


class _TextEncoder(torch.nn.Module):
    dtype = torch.float32

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.grad_enabled_during_call = None

    def forward(self, input_ids, attention_mask):
        self.grad_enabled_during_call = torch.is_grad_enabled()
        hidden = input_ids.float().unsqueeze(-1) * self.weight
        return SimpleNamespace(last_hidden_state=hidden)


def test_prompt_encoding_returns_graph_free_tensors():
    pipe = LongCatVideoAvatarPipeline.__new__(LongCatVideoAvatarPipeline)
    pipe.tokenizer = _Tokenizer()
    pipe.text_encoder = _TextEncoder()

    embeds, mask = pipe._get_t5_prompt_embeds("hello", device="cpu")

    assert pipe.text_encoder.grad_enabled_during_call is False
    assert embeds.grad_fn is None
    assert not embeds.requires_grad
    assert mask.grad_fn is None


def test_pipeline_to_allows_absent_preprocessing_components():
    pipe = LongCatVideoAvatarPipeline.__new__(LongCatVideoAvatarPipeline)
    pipe.dit = _Movable()
    pipe.vae = _Movable()
    pipe.text_encoder = None
    pipe.audio_encoder = None
    pipe.audio_feature_extractor = None

    assert pipe.to("cpu") is pipe
    assert pipe.text_encoder is None
    assert pipe.audio_encoder is None
    assert pipe.device == "cpu"


def test_precomputed_prompt_conditioning_works_without_text_components():
    pipe = LongCatVideoAvatarPipeline.__new__(LongCatVideoAvatarPipeline)
    pipe.text_encoder = None
    pipe.tokenizer = None
    conditioning = {
        "prompt_embeds": torch.ones(1, 1, 3, 4),
        "prompt_attention_mask": torch.ones(1, 3, dtype=torch.int64),
        "negative_prompt_embeds": None,
        "negative_prompt_attention_mask": None,
    }

    result = pipe.prepare_prompt_conditioning(
        "prompt", "", False, 1, 3, torch.float32, "cpu", conditioning
    )
    assert torch.equal(result[0], conditioning["prompt_embeds"])
    assert result[2] is None


def test_only_cp_rank_zero_runs_preprocessing():
    assert is_preprocessing_rank(0)
    assert not is_preprocessing_rank(1)
    assert not is_preprocessing_rank(3)


def _broadcast_worker(rank, world_size, init_file):
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    source = {
        "float": torch.arange(12, dtype=torch.bfloat16).reshape(3, 4),
        "mask": torch.tensor([[1, 0, 1]], dtype=torch.int64),
        "optional": None,
    } if rank == 0 else None
    received = broadcast_tensor_mapping(
        source, ("float", "mask", "optional"), src=0, group=dist.group.WORLD
    )
    assert received["float"].shape == (3, 4)
    assert received["float"].dtype == torch.bfloat16
    assert torch.equal(received["float"], torch.arange(12, dtype=torch.bfloat16).reshape(3, 4))
    assert received["mask"].dtype == torch.int64
    assert torch.equal(received["mask"], torch.tensor([[1, 0, 1]], dtype=torch.int64))
    assert received["optional"] is None
    dist.destroy_process_group()


@unittest.skipUnless(dist.is_gloo_available(), "Gloo is unavailable")
def test_variable_tensor_mapping_broadcast_two_processes():
    with tempfile.TemporaryDirectory() as directory:
        init_file = os.path.join(directory, "dist-init")
        mp.spawn(_broadcast_worker, args=(2, init_file), nprocs=2, join=True)
