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
  <img src='assets/longcat_video_avatar_title.svg' alt="LongCat-Video-Avatar">
  <a href='https://meigen-ai.github.io/LongCat-Video-Avatar/'><img src='https://img.shields.io/badge/Project-Page-green'></a>
  <a href='https://github.com/meituan-longcat/LongCat-Video/blob/main/assets/LongCat-Video-Avatar-Tech-Report.pdf'><img src='https://img.shields.io/badge/Technique-Report-red'></a>
  <a href='https://huggingface.co/meituan-longcat/LongCat-Video-Avatar'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue'></a>
</div>
<div align="center" style="line-height: 1;">
  <img src='assets/title_placeholder.svg' alt="placeholder">
  </a>
  <a href='https://github.com/meituan-longcat/LongCat-Flash-Chat/blob/main/figures/wechat_official_accounts.png'><img src='https://img.shields.io/badge/WeChat-LongCat-brightgreen?logo=wechat&logoColor=white'></a>  
  <a href='https://x.com/Meituan_LongCat'><img src='https://img.shields.io/badge/Twitter-LongCat-white?logo=x&logoColor=white'></a>
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
| LongCat-Video-Avatar | single- and multi-character audio-driven video generation | 🤗 [Huggingface](https://huggingface.co/meituan-longcat/LongCat-Video-Avatar) |

Download models using huggingface-cli:
```shell
pip install "huggingface_hub[cli]"
huggingface-cli download meituan-longcat/LongCat-Video --local-dir ./weights/LongCat-Video
huggingface-cli download meituan-longcat/LongCat-Video-Avatar --local-dir ./weights/LongCat-Video-Avatar
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
💡 User tips
> - Lip synchronization accuracy:​​ Audio CFG works optimally between 3–5. Increase the audio CFG value for better synchronization.
> - Prompt Enhancement: Include clear verbal-action cues (e.g., talking, speaking) in the prompt to achieve more natural lip movements.
> - Mitigate repeated actions: Setting the reference image index（--ref_img_index, default to 10） between 0 and 24 ensures better consistency, while selecting other ranges (e.g., -10 or 30) helps reduce repeated actions. Additionally, increasing the mask frame range (--mask_frame_range, default to 3) can further help mitigate repeated actions, but excessively large values may introduce artifacts.
> - Super resolution: Our model is compatible with both 480P and 720P, which can be controlled via --resolution.
> - Dual-Audio Modes: Merge mode (set audio_type to para) requires two audio clips of equal length, and the resulting audio is obtained by summing the two clips; Concatenation mode (set audio_type to add) does not require equal-length inputs, and the resulting audio is formed by sequentially concatenating the two clips with silence padding for any gaps, where by default person1 speaks first and person2 speaks afterward.

- Single-Audio-to-Video Generation
```shell
# Audio-Text-to-Video
torchrun --nproc_per_node=2 run_demo_avatar_single_audio_to_video.py --context_parallel_size=2 --checkpoint_dir=./weights/LongCat-Video-Avatar --stage_1=at2v --input_json=assets/avatar/single_example_1.json

# Audio-Image-to-Video
torchrun --nproc_per_node=2 run_demo_avatar_single_audio_to_video.py --context_parallel_size=2 --checkpoint_dir=./weights/LongCat-Video-Avatar  --stage_1=ai2v --input_json=assets/avatar/single_example_1.json

# Audio-Text-to-Video and Video-Continuation
torchrun --nproc_per_node=2 run_demo_avatar_single_audio_to_video.py --context_parallel_size=2 --checkpoint_dir=./weights/LongCat-Video-Avatar --stage_1=at2v --input_json=assets/avatar/single_example_1.json --num_segments=5 --ref_img_index=10 --mask_frame_range=3

# Audio-Image-to-Video and Video-Continuation
torchrun --nproc_per_node=2 run_demo_avatar_single_audio_to_video.py --context_parallel_size=2 --checkpoint_dir=./weights/LongCat-Video-Avatar --stage_1=ai2v --input_json=assets/avatar/single_example_1.json --num_segments=5 --ref_img_index=10 --mask_frame_range=3
```

- Multi-Audio-to-Video Generation
```shell
# Audio-Image-to-Video
torchrun --nproc_per_node=2 run_demo_avatar_multi_audio_to_video.py --context_parallel_size=2 --checkpoint_dir=./weights/LongCat-Video-Avatar --input_json=assets/avatar/multi_example_1.json

# Audio-Image-to-Video and Video-Continuation
torchrun --nproc_per_node=2 run_demo_avatar_multi_audio_to_video.py --context_parallel_size=2 --checkpoint_dir=./weights/LongCat-Video-Avatar --input_json=assets/avatar/multi_example_1.json --num_segments=5 --ref_img_index=10 --mask_frame_range=3
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
<img src="assets/Wechat.png" width="200px">
