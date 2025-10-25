# LongCat-Video

<div align="center">
  <img src="assets/longcat-video_logo.svg" width="45%" alt="LongCat-Video" />
</div>
<hr>

<div align="center" style="line-height: 1;">
  <a href='https://meituan-longcat.github.io/longcatvideo/'><img src='https://img.shields.io/badge/Project-Page-green'></a>
  <a href='https://arxiv.org/abs/2508.14033'><img src='https://img.shields.io/badge/Technique-Report-red'></a>
  <a href='https://huggingface.co/meituan-longcat/LongCat-Video'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue'></a>
</div>

<div align="center" style="line-height: 1;">
  <a href='https://github.com/meituan-longcat/LongCat-Flash-Chat/blob/main/figures/wechat_official_accounts.png'><img src='https://img.shields.io/badge/WeChat-LongCat-brightgreen?logo=wechat&logoColor=white'></a>  
  <a href='https://x.com/Meituan_LongCat'><img src='https://img.shields.io/badge/Twitter-LongCat-white?logo=x&logoColor=white'></a>
</div>

<div align="center" style="line-height: 1;">
  <a href='LICENSE'><img src='https://img.shields.io/badge/License-MIT-f5de53?&color=f5de53'></a>
</div>

## Model Introduction
We introduce LongCat-Video, a foundational video generation model with 13.6B parameters, delivering strong performance across *Text-to-Video*, *Image-to-Video*, and *Video-Continuation* generation tasks. It particularly excels in efficient and high-quality long video generation, representing our first step toward world models.

### Key Features
- 🌟 **Unified architecture for multiple tasks**: LongCat-Video unifies *Text-to-Video*, *Image-to-Video*, and *Video-Continuation* tasks within a single video generation framework. It natively supports all these tasks with a single model and consistently delivers strong performance across each individual task.
- 🌟 **Long video generation**: LongCat-Video is natively pretrained on *Video-Continuation* tasks, enabling it to produce minutes-long videos without color drifting or quality degradation.
- 🌟 **Efficient inference**: LongCat-Video generates $720p$, $30fps$ videos within minutes by employing a coarse-to-fine generation strategy along both the temporal and spatial axes. Block Sparse Attention further enhances efficiency, particularly at high resolutions
- 🌟 **Strong performance with multi-reward RLHF**: Powered by multi-reward Group Relative Policy Optimization (GRPO), comprehensive evaluations on both internal and public benchmarks demonstrate that LongCat-Video achieves performance comparable to leading open-source video generation models as well as the latest commercial solutions.

For more detail, please refer to the comprehensive [***LongCat-Video Technical Report***](https://github.com/meituan-longcat/LongCat-Video/blob/main/longcatvideo_tech_report.pdf).

## Quick Start

### Installation

Clone the repo:

```shell
git clone https://github.com/meituan-longcat/LongCat-Video
cd LongCat-Video
```

Install dependencies:

```shell
pip install -r requirements.txt
```

FlashAttention-2 is enabled in the model config by default; you can also change the model config to use FlashAttention-3 or xformers.

### Model Download

| Models | Download Link |
| --- | --- |
| LongCat-Video | 🤗 [Huggingface](https://huggingface.co/meituan-longcat/LongCat-Video) |

Download models using huggingface-cli:
```shell
pip install "huggingface_hub[cli]"
huggingface-cli download meituan-longcat/LongCat-Video --local-dir ./weights/LongCat-Video
```

### Run Text-to-Video

```shell
# Single-GPU inference
torchrun run_demo_image_to_video.py --checkpoint_dir=./weights/LongCat-Video --enable_compile

# Multi-GPU inference
torchrun --nproc_per_node=2 run_demo_image_to_video.py --context_parallel_size=2 --checkpoint_dir=./weights/LongCat-Video --enable_compile
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
| Text-Alignment | 3.99 | 3.81 | 3.70 | 3.76 |
| Visual Quality | 3.23 | 3.13 | 3.26 | 3.25 |
| Motion Quality | 3.86 | 3.81 | 3.78 | 3.74 |
| Overall Quality | 3.48 | 3.36 | 3.35 | 3.38 |

### Image-to-Video
The *Image-to-Video* MOS evaluation results on our internal benchmark.

| **MOS score** | **Seedance 1.0** | **Hailuo-02** | **Wan 2.2-I2V-A14B** | **LongCat-Video** |
|---------------|-------------------|--------------------|-------------|-------------|
| **Accessibility** | Proprietary | Proprietary | Open Source | Open Source |
| **Architecture** | - | - | MoE | Dense |
| **# Total Params** | - | - | 28B | 13.6B |
| **# Activated Params** | - | - | 14B | 13.6B |
| Image-Alignment | 4.12 | 4.18 | 4.18 | 4.04 |
| Text-Alignment | 3.70 | 3.85 | 3.33 | 3.49 |
| Visual Quality | 3.22 | 3.18 | 3.23 | 3.27 |
| Motion Quality | 3.77 | 3.80 | 3.79 | 3.59 |
| Overall Quality | 3.35 | 3.27 | 3.26 | 3.17 |

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
@misc{meituan2025longcatvideotechnicalreport, 
    title={LongCat-Video Technical Report}, 
    author={Meituan LongCat Team}, 
    year={2025}, 
    eprint={xxx}, 
    archivePrefix={arXiv}, 
    primaryClass={cs.CL}, 
    url={https://arxiv.org/abs/xxx}, 
}
```

## Acknowledgements

We would like to thank the contributors to the [Wan](https://huggingface.co/Wan-AI), [UMT5-XXL](https://huggingface.co/google/umt5-xxl), [Diffusers](https://github.com/huggingface/diffusers) and [HuggingFace](https://huggingface.co) repositories, for their open research.


## Contact
Please contact us at <a href="mailto:longcat-team@meituan.com">longcat-team@meituan.com</a> or join our WeChat Group if you have any questions.