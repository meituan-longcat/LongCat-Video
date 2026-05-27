FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV TORCH_CUDA_ARCH_LIST="8.9;9.0"

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-dev python3.10-venv python3-pip \
    git wget curl ffmpeg libsndfile1 libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/python3.10 /usr/bin/python3

WORKDIR /app

RUN pip install --no-cache-dir \
    torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu124

RUN pip install --no-cache-dir ninja psutil packaging && \
    pip install --no-cache-dir flash_attn==2.7.4.post1

COPY requirements.txt requirements_avatar.txt ./
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir librosa && \
    pip install --no-cache-dir -r requirements_avatar.txt

COPY longcat_video/ ./longcat_video/
COPY run_demo_*.py run_streamlit.py ./
COPY assets/ ./assets/

EXPOSE 8501

ENTRYPOINT ["python"]
CMD ["--help"]
