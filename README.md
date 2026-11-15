# LongCat-Video

<div align="center">
  <img src="assets/longcat-video_logo.svg" width="45%" alt="LongCat-Video" />
</div>
<hr>


<div align="center" style="line-height: 1;">
  <img src='assets/longcat_video_title.svg' alt="LongCat-Video">
  <a href='https://meituan-longcat.github.io/LongCat-Video/'><img src='https://img.shields.io/badge/Project-Page-green'></a>
  <a href='https://arxiv.org/abs/2510.22200'><img src='https://img.shields.io/badge/Technique-Report-red'></a>
  <a href='https://huggingface.co/meituan-longcat/LongCat-Video'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue'></a>
</div>
<div align="center" style="line-height: 1;">
  <img src='assets/longcat_video_avatar_1.6_title.svg' alt="LongCat-Video-Avatar 1.6">
  <a href='https://meigen-ai.github.io/LongCat-Video-Avatar-1.6-Page/'><img src='https://img.shields.io/badge/Project-Page-green'></a>
  <a href='https://github.com/meituan-longcat/LongCat-Video/blob/main/assets/LongCat-Video-Avatar-1.6-Tech-Report.pdf'><img src='https://img.shields.io/badge/Technique-Report-red'></a>
  <a href='https://huggingface.co/meituan-longcat/LongCat-Video-Avatar-1.6'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue'></a>
  <a href='https://www.modelscope.cn/models/meituan-longcat/LongCat-Video-Avatar-1.6'><img src='https://img.shields.io/badge/ModelScope-Model-purple'></a>
</div>
<div align="center" style="line-height: 1;">
  <img src='assets/longcat_video_avatar_1.5_title.svg' alt="LongCat-Video-Avatar 1.5">
  <a href='https://meigen-ai.github.io/LongCat-Video-Avatar-1.5-Page/'><img src='https://img.shields.io/badge/Project-Page-green'></a>
  <a href='https://github.com/meituan-longcat/LongCat-Video/blob/main/assets/LongCat-Video-Avatar-1.5-Tech-Report.pdf'><img src='https://img.shields.io/badge/Technique-Report-red'></a>
  <a href='https://huggingface.co/meituan-longcat/LongCat-Video-Avatar-1.5'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue'></a>
  <a href='https://www.modelscope.cn/models/meituan-longcat/LongCat-Video-Avatar-1.5'><img src='https://img.shields.io/badge/ModelScope-Model-purple'></a>
</div>
<div align="center" style="line-height: 1;">
  <img src='assets/title_placeholder.svg' alt="placeholder">
  </a>
  <a href='https://github.com/meituan-longcat/LongCat-Flash-Chat/blob/main/figures/wechat_official_accounts.png'><img src='https://img.shields.io/badge/WeChat-LongCat-brightgreen?logo=wechat&logoColor=white'></a>  
  <a href='https://x.com/Meituan_LongCat'><img src='https://img.shields.io/badge/Twitter-LongCat-white?logo=x&logoColor=white'></a>
<a href="https://discord.gg/EXsG52D8SW"><img src="https://img.shields.io/badge/Discord-Join%20Chat-5865F2?logo=discord&logoColor=white"></a>
  <a href='LICENSE'><img src='https://img.shields.io/badge/License-MIT-f5de53?&color=f5de53'></a>
</div>

## Model Introduction
We introduce LongCat-Video, a foundational video generation model with 13.6B parameters, delivering strong performance across *Text-to-Video*, *Image-to-Video*, and *Video-Continuation* generation tasks. It particularly excels in efficient and high-quality long video generation, representing our first step toward world models.

### Key Features
- 🌟 **Unified architecture for multiple tasks**: LongCat-Video unifies *Text-to-Video*, *Image-to-Video*, and *Video-Continuation* tasks within a single video generation framework. It natively supports all these tasks with a single model and consistently delivers strong performance across each individual task.
- 🌟 **Long video generation**: LongCat-Video is natively pretrained on *Video-Continuation* tasks, enabling it to produce minutes-long videos without color drifting or quality degradation.
- 🌟 **Efficient inference**: LongCat-Video generates $720p$, $30fps$ videos within minutes by employing a coarse-to-fine generation strategy along both the temporal and spatial axes. Block Sparse Attention further enhances efficiency, particularly at high resolutions
- 🌟 **Strong performance with multi-reward RLHF**: Powered by multi-reward Group Relative Policy Optimization (GRPO), comprehensive evaluations on both internal and public benchmarks demonstrate that LongCat-Video achieves performance comparable to leading open-source video generation models as well as the latest commercial solutions.

For more detail, please refer to the comprehensive [***LongCat-Video Technical Report***](https://arxiv.org/abs/2510.22200).

## 🎥 Teaser Video

<div align="center">
  <video src="https://github.com/user-attachments/assets/00fa63f0-9c4e-461a-a79e-c662ad596d7d" width="2264" height="384"> </video>
</div>

## 🔥 Latest News!!
- Nov 15, 2026: 🚀 We release [***LongCat-Video-Avatar 1.6***](https://meigen-ai.github.io/LongCat-Video-Avatar-1.6-Page/), an upgraded open-source framework for audio-driven human video generation. Building on LongCat-Video-Avatar 1.5, v1.6 delivers more natural lip movements and expressive facial, head, and body motion, while reducing error accumulation during long-form generation. A refined post-training pipeline combining reinforcement learning, improved classifier-free guidance for DMD-based few-step distillation, long-video rollouts, and on-policy distillation enables videos exceeding 10 minutes with improved visual quality and temporal consistency. Comprehensive inference optimizations further accelerate end-to-end generation and deployment.[ [***code***](https://github.com/meituan-longcat/LongCat-Video) | 🤗 [***weights***](https://huggingface.co/meituan-longcat/LongCat-Video-Avatar-1.6) | [***project page***](https://meigen-ai.github.io/LongCat-Video-Avatar-1.6-Page/) ]
- May 21, 2026: 🚀 We release [***LongCat-Video-Avatar-1.5***](https://meigen-ai.github.io/LongCat-Video-Avatar-1.5-Page/), an upgraded open-source framework for audio-driven human video generation. v1.5 replaces Wav2Vec2 with Whisper-Large for more accurate lip synchronization, achieves production-ready physical rationality and temporal stability with robust long-video generation, generalizes to stylized domains (anime, animals, complex real-world conditions), supports both single-stream and multi-stream audio inputs, and accelerates inference to 8 steps via step distillation. [ [***code***](https://github.com/meituan-longcat/LongCat-Video) | 🤗 [***weights***](https://huggingface.co/meituan-longcat/LongCat-Video-Avatar-1.5) | [***project page***](https://meigen-ai.github.io/LongCat-Video-Avatar-1.5-Page/) ]
- Dec 16, 2025: 🚀 We are excited to announce the release of [***LongCat-Video-Avatar***](https://meigen-ai.github.io/LongCat-Video-Avatar/), a unified model that delivers expressive and highly dynamic audio-driven character animation, supporting native tasks including *Audio-Text-to-Video*, *Audio-Text-Image-to-Video*, and *Video Continuation* with seamless compatibility for both *single-stream* and *multi-stream* audio inputs. The release includes our [***Technical Report***](https://github.com/meituan-longcat/LongCat-Video), [***inference code***](https://github.com/meituan-longcat/LongCat-Video), 🤗 [***model weights***](https://huggingface.co/meituan-longcat/LongCat-Video-Avatar), and [***project page***](https://meigen-ai.github.io/LongCat-Video-Avatar/).
- Oct 25, 2025: 🚀 We've released LongCat-Video, a foundational video generation model.  Tech report and models are available at [***LongCat-Video Technical Report***](https://arxiv.org/abs/2510.22200) and 🤗 [***Huggingface***](https://huggingface.co/meituan-longcat/LongCat-Video) !



## Quick Start

### Installation

Clone the repo:

```shell
git clone --single-branch --branch main https://github.com/meituan-longcat/LongCat-Video
cd LongCat-Video
```

Install dependencies:

```shell
# create conda environment
conda create -n longcat-video python=3.10
conda activate longcat-video

# install torch (configure according to your CUDA version)
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# install flash-attn-2
pip install ninja 
pip install psutil 
pip install packaging 
pip install flash_attn==2.7.4.post1

# install other requirements
pip install -r requirements.txt

# install longcat-video-avatar requirements
conda install -c conda-forge librosa
conda install -c conda-forge ffmpeg
pip install -r requirements_avatar.txt

```

FlashAttention-2 is enabled in the model config by default; you can also change the model config ("./weights/LongCat-Video/dit/config.json") to use FlashAttention-3 or xformers once installed.

### Model Download

| Models | Description | Download Link |
| --- | --- | --- |
| LongCat-Video | foundational video generation | 🤗 [Huggingface](https://huggingface.co/meituan-longcat/LongCat-Video) |
| LongCat-Video-Avatar | single- and multi-character audio-driven video generation (wav2vec2) | 🤗 [Huggingface](https://huggingface.co/meituan-longcat/LongCat-Video-Avatar) |
| LongCat-Video-Avatar-1.5 | upgraded avatar model with Whisper-large-v3 audio encoder, distillation-based fast inference | 🤗 [Huggingface](https://huggingface.co/meituan-longcat/LongCat-Video-Avatar-1.5) |
| LongCat-Video-Avatar-1.6 | Improves the naturalness of speech-related mouth movements and reduces error accumulation during long-video generation | 🤗 [Huggingface](https://huggingface.co/meituan-longcat/LongCat-Video-Avatar-1.6) |

Download models using huggingface-cli:
```shell
pip install "huggingface_hub[cli]"
huggingface-cli download meituan-longcat/LongCat-Video --local-dir ./weights/LongCat-Video
huggingface-cli download meituan-longcat/LongCat-Video-Avatar --local-dir ./weights/LongCat-Video-Avatar
huggingface-cli download meituan-longcat/LongCat-Video-Avatar-1.5 --local-dir ./weights/LongCat-Video-Avatar-1.5
huggingface-cli download meituan-longcat/LongCat-Video-Avatar-1.6 --local-dir ./weights/LongCat-Video-Avatar-1.6
```

### Run Text-to-Video

```shell
# Single-GPU inference
torchrun run_demo_text_to_video.py --checkpoint_dir=./weights/LongCat-Video --enable_compile

# Multi-GPU inference
torchrun --nproc_per_node=2 run_demo_text_to_video.py --context_parallel_size=2 --checkpoint_dir=./weights/LongCat-Video --enable_compile
```

### Run Image-to-Video

```shell
# Single-GPU inference
torchrun run_demo_image_to_video.py --checkpoint_dir=./weights/LongCat-Video --enable_compile

# Multi-GPU inference
torchrun --nproc_per_node=2 run_demo_image_to_video.py --context_parallel_size=2 --checkpoint_dir=./weights/LongCat-Video --enable_compile
```

### Run Video-Continuation

```shell
# Single-GPU inference
torchrun run_demo_video_continuation.py --checkpoint_dir=./weights/LongCat-Video --enable_compile

# Multi-GPU inference
torchrun --nproc_per_node=2 run_demo_video_continuation.py --context_parallel_size=2 --checkpoint_dir=./weights/LongCat-Video --enable_compile
```

### Run Long-Video Generation

```shell
# Single-GPU inference
torchrun run_demo_long_video.py --checkpoint_dir=./weights/LongCat-Video --enable_compile

# Multi-GPU inference
torchrun --nproc_per_node=2 run_demo_long_video.py --context_parallel_size=2 --checkpoint_dir=./weights/LongCat-Video --enable_compile
```

### Run Interactive Video Generation

```shell
# Single-GPU inference
torchrun run_demo_interactive_video.py --checkpoint_dir=./weights/LongCat-Video --enable_compile

# Multi-GPU inference
torchrun --nproc_per_node=2 run_demo_interactive_video.py --context_parallel_size=2 --checkpoint_dir=./weights/LongCat-Video --enable_compile
```

### Run LongCat-Video-Avatar

<details>
<summary>💡 User tips for 1.6</summary>

1. The Avatar 1.6 checkpoint uses the same Avatar 1.5 architecture and adds two LoRAs:

- `loraA/lora.safetensors`: DMD LoRA.
- `loraB/lora.safetensors`: OPSD LoRA.
- `base_model_int8`: optional weight-only INT8 DiT, enabled by `--use_int8`.

With `--mix_steps=N`, the first `N` denoising steps use
`mix_alpha * DMD + (1 - mix_alpha) * OPSD`; the remaining `8 - N` steps use
OPSD only. The default settings are `--mix_steps=4 --mix_alpha=1.0`, which
disable mixing and load the single LoRA given by `--lora_path`. Increasing
`--mix_steps` and `--mix_alpha` gives the DMD LoRA more influence during the
early denoising steps, which can improve structural stability and reduce
visual distortions.

2. **FP8 dependency and INT8 exclusivity:** Install Transformer Engine with
   PyTorch/CUDA support before using `--use_fp8`. Do not use `--use_fp8` and
   `--use_int8` together; enable only one quantization mode at a time. If output
   quality is more important, reduce the number of layers that use FP8.

3. **Compilation warm-up:** When `--compile_dit` and `--compile_vae` are enabled, the first few inference runs include graph tracing and compilation overhead and may be slower due to cold start. Subsequent runs with the same input shapes reuse the compiled graphs and run significantly faster. New input shapes may trigger recompilation.

4. **FlashAttention 3:** For Hopper GPUs such as H20, H100, H200, and H800, make sure FlashAttention 3 is installed and enabled. FA3 can provide a significant inference speedup compared with FA2. Confirm that the model configuration contains:

   ```json
   {
     "enable_flashattn3": true,
     "enable_flashattn2": false
   }
   ```

5. **Multi-GPU communication:** When running with more than 2 GPUs, enable the optimized All-to-All workspace to reduce context-parallel communication overhead:

   ```shell
   export LONGCAT_A2A_WORKSPACE=1
   ```

   Set `--nproc_per_node` and `--context_parallel_size` to the corresponding GPU count.

</details>

<details>
<summary>💡 User tips for 1.5</summary>

> - **Lip synchronization accuracy:** Audio CFG works optimally between 3–5. Increase the audio CFG value for better synchronization.
> - **Prompt Enhancement:** Longer, more descriptive prompts yield better consistency and naturalness than short ones. We recommend including rich details such as character appearance, actions, and scene context (e.g., *"A young woman with long black hair is speaking and smiling, wearing a white blouse, sitting in a bright café"*) for best results.
> - **Mitigate repeated actions:** Setting the reference image index（--ref_img_index, default to 10） between 0 and 24 ensures better consistency; setting it to 30 helps reduce repeated actions. Additionally, increasing the mask frame range (--mask_frame_range, default to 3) can further help mitigate repeated actions, but excessively large values may introduce artifacts.
> - **Super resolution:** Our model is compatible with both 480P and 720P, which can be controlled via --resolution.
> - **Dual-Audio Modes:** Merge mode (set audio_type to para) requires two audio clips of equal length, and the resulting audio is obtained by summing the two clips; Concatenation mode (set audio_type to add) does not require equal-length inputs, and the resulting audio is formed by sequentially concatenating the two clips with silence padding for any gaps, where by default person1 speaks first and person2 speaks afterward.
> - **Model versions:** `--model_type avatar-v1.0` uses wav2vec2 (default); `--model_type avatar-v1.5` uses Whisper-large-v3; 
> - **Distillation mode:** Add `--use_distill` to enable 8-step inference. It is required for Avatar 1.5 examples and for Avatar 1.6 LoRA mixing.
> - **INT8 quantization:** Add `--use_int8` to load `base_model_int8` instead of `base_model`. It is supported with `avatar-v1.5`.

</details>

<details>
<summary>💡 User tips for 1.0</summary>

> - Lip synchronization accuracy:​​ Audio CFG works optimally between 3–5. Increase the audio CFG value for better synchronization.
> - Prompt Enhancement: Include clear verbal-action cues (e.g., talking, speaking) in the prompt to achieve more natural lip movements.
> - Mitigate repeated actions: Setting the reference image index（--ref_img_index, default to 10） between 0 and 24 ensures better consistency, while selecting other ranges (e.g., -10 or 30) helps reduce repeated actions. Additionally, increasing the mask frame range (--mask_frame_range, default to 3) can further help mitigate repeated actions, but excessively large values may introduce artifacts.
> - Super resolution: Our model is compatible with both 480P and 720P, which can be controlled via --resolution.
> - Dual-Audio Modes: Merge mode (set audio_type to para) requires two audio clips of equal length, and the resulting audio is obtained by summing the two clips; Concatenation mode (set audio_type to add) does not require equal-length inputs, and the resulting audio is formed by sequentially concatenating the two clips with silence padding for any gaps, where by default person1 speaks first and person2 speaks afterward.

</details>

#### LongCat-Video-Avatar-1.6

- Single-Audio-to-Video Generation
```shell
# Audio-Text-to-Video
torchrun --nproc_per_node=2 run_demo_avatar_single_audio_to_video.py --context_parallel_size=2 --checkpoint_dir=./weights/LongCat-Video-Avatar-1.6 --stage_1=at2v --input_json=assets/avatar/single_example_1.json --use_distill --lora_path loraB/lora.safetensors --model_type avatar-v1.6 --mix_steps=4 --mix_alpha=1.0

# Audio-Image-to-Video
torchrun --nproc_per_node=2 run_demo_avatar_single_audio_to_video.py --context_parallel_size=2 --checkpoint_dir=./weights/LongCat-Video-Avatar-1.6 --stage_1=ai2v --input_json=assets/avatar/single_example_1.json --use_distill --lora_path loraB/lora.safetensors --model_type avatar-v1.6 --mix_steps=4 --mix_alpha=1.0

# Audio-Text-to-Video and Video-Continuation
torchrun --nproc_per_node=2 run_demo_avatar_single_audio_to_video.py --context_parallel_size=2 --checkpoint_dir=./weights/LongCat-Video-Avatar-1.6 --stage_1=at2v --input_json=assets/avatar/single_example_1.json --num_segments=5 --ref_img_index=10 --mask_frame_range=0 --use_distill --lora_path loraB/lora.safetensors --model_type avatar-v1.6 --mix_steps=4 --mix_alpha=1.0

# Audio-Image-to-Video and Video-Continuation
torchrun --nproc_per_node=2 run_demo_avatar_single_audio_to_video.py --context_parallel_size=2 --checkpoint_dir=./weights/LongCat-Video-Avatar-1.6 --stage_1=ai2v --input_json=assets/avatar/single_example_1.json --num_segments=5 --ref_img_index=10 --mask_frame_range=0 --use_distill --lora_path loraB/lora.safetensors --model_type avatar-v1.6 --mix_steps=4 --mix_alpha=1.0
```

- Multi-Audio-to-Video Generation
```shell
# Audio-Image-to-Video
torchrun --nproc_per_node=2 run_demo_avatar_multi_audio_to_video.py --context_parallel_size=2 --checkpoint_dir=./weights/LongCat-Video-Avatar-1.6 --input_json=assets/avatar/multi_example_1.json --use_distill --lora_path loraB/lora.safetensors --model_type avatar-v1.6 --mix_steps=4 --mix_alpha=1.0

# Audio-Image-to-Video and Video-Continuation
torchrun --nproc_per_node=2 run_demo_avatar_multi_audio_to_video.py --context_parallel_size=2 --checkpoint_dir=./weights/LongCat-Video-Avatar-1.6 --input_json=assets/avatar/multi_example_1.json --num_segments=5 --ref_img_index=10 --mask_frame_range=0 --use_distill --lora_path loraB/lora.safetensors --model_type avatar-v1.6 --mix_steps=4 --mix_alpha=1.0
```

### Run Streamlit

```shell
# Single-GPU inference
streamlit run ./run_streamlit.py --server.fileWatcherType none --server.headless=false
```



## Evaluation Results

### Text-to-Video
The *Text-to-Video* MOS evaluation results on our internal benchmark.

| **MOS score** | **Veo3** | **PixVerse-V5** | **Wan 2.2-T2V-A14B** | **LongCat-Video** |
|---------------|-------------------|--------------------|-------------|-------------|
| **Accessibility** | Proprietary | Proprietary | Open Source | Open Source |
| **Architecture** | - | - | MoE | Dense |
| **# Total Params** | - | - | 28B | 13.6B |
| **# Activated Params** | - | - | 14B | 13.6B |
| Text-Alignment↑ | 3.99 | 3.81 | 3.70 | 3.76 |
| Visual Quality↑ | 3.23 | 3.13 | 3.26 | 3.25 |
| Motion Quality↑ | 3.86 | 3.81 | 3.78 | 3.74 |
| Overall Quality↑ | 3.48 | 3.36 | 3.35 | 3.38 |

### Image-to-Video
The *Image-to-Video* MOS evaluation results on our internal benchmark.

| **MOS score** | **Seedance 1.0** | **Hailuo-02** | **Wan 2.2-I2V-A14B** | **LongCat-Video** |
|---------------|-------------------|--------------------|-------------|-------------|
| **Accessibility** | Proprietary | Proprietary | Open Source | Open Source |
| **Architecture** | - | - | MoE | Dense |
| **# Total Params** | - | - | 28B | 13.6B |
| **# Activated Params** | - | - | 14B | 13.6B |
| Image-Alignment↑ | 4.12 | 4.18 | 4.18 | 4.04 |
| Text-Alignment↑ | 3.70 | 3.85 | 3.33 | 3.49 |
| Visual Quality↑ | 3.22 | 3.18 | 3.23 | 3.27 |
| Motion Quality↑ | 3.77 | 3.80 | 3.79 | 3.59 |
| Overall Quality↑ | 3.35 | 3.27 | 3.26 | 3.17 |

## Community Works

Community works are welcome! Please PR or inform us in Issue to add your work.

- [CacheDiT](https://github.com/vipshop/cache-dit) offers Fully Cache Acceleration support for LongCat-Video with DBCache and TaylorSeer, achieved nearly 1.7x speedup without obvious loss of precision. Visit their [example](https://github.com/vipshop/cache-dit/blob/main/examples/pipeline/run_longcat_video.py) for more details.


## License Agreement

The **model weights** are released under the **MIT License**. 

Any contributions to this repository are licensed under the MIT License, unless otherwise stated. This license does not grant any rights to use Meituan trademarks or patents. 

See the [LICENSE](LICENSE) file for the full license text.


## Usage Considerations 
This model has not been specifically designed or comprehensively evaluated for every possible downstream application. 

Developers should take into account the known limitations of large language models, including performance variations across different languages, and carefully assess accuracy, safety, and fairness before deploying the model in sensitive or high-risk scenarios. 
It is the responsibility of developers and downstream users to understand and comply with all applicable laws and regulations relevant to their use case, including but not limited to data protection, privacy, and content safety requirements. 

Nothing in this Model Card should be interpreted as altering or restricting the terms of the MIT License under which the model is released. 

## Citation
We kindly encourage citation of our work if you find it useful.

```
@misc{meituanlongcatteam2025longcatvideotechnicalreport,
      title={LongCat-Video Technical Report}, 
      author={Meituan LongCat Team and Xunliang Cai and Qilong Huang and Zhuoliang Kang and Hongyu Li and Shijun Liang and Liya Ma and Siyu Ren and Xiaoming Wei and Rixu Xie and Tong Zhang},
      year={2025},
      eprint={2510.22200},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2510.22200}, 
}

@misc{meituanlongcatteam2026longcatvideoavatar15technicalreport,
      title={LongCat-Video-Avatar 1.5 Technical Report}, 
      author={Meituan LongCat Team and Xunliang Cai and Meng Cheng and Feng Gao and Zhe Kong and Jiamu Li and Le Li and Weiheng Li and Hongyu Liu and Shuai Tan and Xiaoming Wei and Tianyu Yang and Yong Zhang},
      year={2026},
      eprint={2605.26486},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2605.26486}, 
}

@misc{meituanlongcatteam2025longcatvideoavatartechnicalreport,
      title={LongCat-Video-Avatar Technical Report}, 
      author={Meituan LongCat Team},
      year={2025},
      eprint={},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={}, 
}
```

## Acknowledgements

We would like to thank the contributors to the [Wan](https://huggingface.co/Wan-AI), [UMT5-XXL](https://huggingface.co/google/umt5-xxl), [Diffusers](https://github.com/huggingface/diffusers) and [HuggingFace](https://huggingface.co) repositories, for their open research.


## Contact
Please contact us at <a href="mailto:longcat-team@meituan.com">longcat-team@meituan.com</a> or scan the QR code to join our WeChat Group if you have any questions.  
<img src="https://raw.githubusercontent.com/meituan-longcat/LongCat-Flash-Chat/main/wechat-assets/Wechat.png" width="200px">
