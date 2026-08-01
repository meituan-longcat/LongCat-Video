#!/usr/bin/env bash
set -Eeuo pipefail

# Fresh-clone setup for meituan-longcat/LongCat-Video using uv.
#
# Usage:
#   chmod +x setup-with-uv.sh
#   ./setup-with-uv.sh
#
# Optional:
#   SKIP_APT=1 ./setup-with-uv.sh
#   PYTHON_VERSION=3.10 ./setup-with-uv.sh
#
# Run this from the LongCat-Video repository root.

PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
SKIP_APT="${SKIP_APT:-0}"

log() {
  printf '\n\033[1;34m==> %s\033[0m\n' "$*"
}

die() {
  printf '\n\033[1;31mERROR: %s\033[0m\n' "$*" >&2
  exit 1
}

command -v uv >/dev/null 2>&1 || die "uv is not installed or not in PATH."
[[ -f requirements.txt ]] || die "requirements.txt not found. Run this script from the LongCat-Video repository root."
[[ -f requirements_avatar.txt ]] || die "requirements_avatar.txt not found."

if [[ "$SKIP_APT" != "1" ]]; then
  log "Installing Ubuntu system packages"
  sudo apt update
  sudo apt install -y \
    build-essential \
    ffmpeg \
    libsndfile1 \
    patchelf
else
  log "Skipping apt packages because SKIP_APT=1"
fi

log "Initializing uv project metadata"
uv python pin "$PYTHON_VERSION"

if [[ ! -f pyproject.toml ]]; then
  uv init --bare
fi

uv run python - <<'PY'
from pathlib import Path
import re

path = Path("pyproject.toml")
text = path.read_text()

if "[project]" not in text:
    text = (
        '[project]\n'
        'name = "longcat-video"\n'
        'version = "0.1.0"\n'
        'requires-python = ">=3.10,<3.11"\n'
        'dependencies = []\n\n'
        + text
    )
elif re.search(r'^requires-python\s*=', text, flags=re.M):
    text = re.sub(
        r'^requires-python\s*=.*$',
        'requires-python = ">=3.10,<3.11"',
        text,
        flags=re.M,
    )
else:
    text = text.replace(
        "[project]\n",
        '[project]\nrequires-python = ">=3.10,<3.11"\n',
        1,
    )

uv_config = '''
[tool.uv.sources]
torch = { index = "pytorch-cu124" }
torchvision = { index = "pytorch-cu124" }
torchaudio = { index = "pytorch-cu124" }

[[tool.uv.index]]
name = "pytorch-cu124"
url = "https://download.pytorch.org/whl/cu124"
explicit = true

[tool.pytest.ini_options]
pythonpath = ["."]
'''

if "[[tool.uv.index]]" not in text:
    text = text.rstrip() + "\n\n" + uv_config.lstrip()
elif "[tool.pytest.ini_options]" not in text:
    text = text.rstrip() + '\n\n[tool.pytest.ini_options]\npythonpath = ["."]\n'

path.write_text(text)
PY

log "Installing CUDA 12.4 PyTorch stack"
uv add \
  "torch==2.6.0+cu124" \
  "torchvision==0.21.0+cu124" \
  "torchaudio==2.6.0"

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

BASE_REQ="$TMP_DIR/requirements-base-uv.txt"
AVATAR_REQ="$TMP_DIR/requirements-avatar-uv.txt"

grep -vE '^[[:space:]]*(torch|torchvision|torchaudio|flash[-_]attn)([[:space:]]|[=<>!~].*)?$' \
  requirements.txt > "$BASE_REQ"

grep -vE '^[[:space:]]*(libsndfile1|tritonserverclient)([[:space:]]|[=<>!~].*)?$' \
  requirements_avatar.txt > "$AVATAR_REQ"

log "Installing base Python dependencies"
uv add -r "$BASE_REQ"

log "Installing FlashAttention build prerequisites"
uv add ninja psutil packaging setuptools wheel

log "Building FlashAttention against installed PyTorch"
uv pip install "flash-attn==2.7.4.post1" --no-build-isolation

log "Installing Avatar 1.5 dependencies"
uv add -r "$AVATAR_REQ"

log "Constraining Hugging Face dependencies"
uv add \
  accelerate \
  "huggingface-hub>=0.23,<1.0"

log "Installing test dependency"
uv add --dev pytest

log "Clearing executable-stack flags from ONNX Runtime shared libraries"
SITE_PACKAGES="$(
  uv run python - <<'PY'
import site
paths = site.getsitepackages()
if not paths:
    raise SystemExit("Could not determine site-packages directory")
print(paths[0])
PY
)"

ONNX_DIR="$SITE_PACKAGES/onnxruntime"
[[ -d "$ONNX_DIR" ]] || die "ONNX Runtime directory not found at $ONNX_DIR"

find "$ONNX_DIR" \
  -type f -name '*.so' \
  -exec patchelf --clear-execstack {} +

log "Verifying core imports and CUDA"
uv run python - <<'PY'
import accelerate
import flash_attn
import huggingface_hub
import librosa
import onnx
import onnxruntime as ort
import soundfile
import torch
import transformers
from audio_separator.separator import Separator

print("torch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
print("flash-attn:", flash_attn.__version__)
print("transformers:", transformers.__version__)
print("huggingface-hub:", huggingface_hub.__version__)
print("accelerate:", accelerate.__version__)
print("librosa:", librosa.__version__)
print("soundfile:", soundfile.__version__)
print("onnx:", onnx.__version__)
print("onnxruntime:", ort.__version__)
print("ONNX providers:", ort.get_available_providers())
print("audio-separator import: OK")

if not torch.cuda.is_available():
    raise SystemExit("PyTorch cannot see CUDA.")
PY

log "Compiling repository Python files"
uv run python -m compileall \
  run_demo_avatar_single_audio_to_video.py \
  longcat_video

log "Running tests"
uv run pytest -q

log "Setup completed successfully"
printf '\nUse the environment through uv, for example:\n'
printf '  uv run python run_demo_avatar_single_audio_to_video.py --help\n\n'

