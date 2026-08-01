#!/usr/bin/env bash
set -Eeuo pipefail

# Create the local ./weights layout expected by LongCat-Video Avatar 1.5
# while keeping all model data in the Hugging Face cache.
#
# Default assumptions:
#   HF cache root: /data/huggingface
#   repo root:     current working directory
#   weights dir:   ./weights
#
# Usage:
#   chmod +x setup-local-weights-from-hf-cache.sh
#   ./setup-local-weights-from-hf-cache.sh
#
# Optional overrides:
#   HF_HOME=/data/huggingface ./setup-local-weights-from-hf-cache.sh
#   WEIGHTS_DIR=/some/path/weights ./setup-local-weights-from-hf-cache.sh

HF_HOME="${HF_HOME:-/data/huggingface}"
HF_HUB="${HF_HOME}/hub"
WEIGHTS_DIR="${WEIGHTS_DIR:-./weights}"

BASE_REPO_ID="meituan-longcat/LongCat-Video"
AVATAR_REPO_ID="meituan-longcat/LongCat-Video-Avatar-1.5"

BASE_CACHE_DIR="${HF_HUB}/models--meituan-longcat--LongCat-Video"
AVATAR_CACHE_DIR="${HF_HUB}/models--meituan-longcat--LongCat-Video-Avatar-1.5"

log() {
  printf '\n\033[1;34m==> %s\033[0m\n' "$*"
}

die() {
  printf '\n\033[1;31mERROR: %s\033[0m\n' "$*" >&2
  exit 1
}

resolve_snapshot() {
  local repo_cache_dir="$1"
  local repo_id="$2"
  local ref_file="${repo_cache_dir}/refs/main"

  [[ -d "$repo_cache_dir" ]] || \
    die "Cache directory missing for ${repo_id}: ${repo_cache_dir}"

  [[ -f "$ref_file" ]] || \
    die "refs/main missing for ${repo_id}: ${ref_file}"

  local commit
  commit="$(tr -d '[:space:]' < "$ref_file")"

  [[ -n "$commit" ]] || \
    die "Empty refs/main for ${repo_id}: ${ref_file}"

  local snapshot="${repo_cache_dir}/snapshots/${commit}"

  [[ -d "$snapshot" ]] || \
    die "Snapshot directory missing for ${repo_id}: ${snapshot}"

  printf '%s\n' "$snapshot"
}

log "Resolving Hugging Face cache snapshots"
BASE_SNAPSHOT="$(resolve_snapshot "$BASE_CACHE_DIR" "$BASE_REPO_ID")"
AVATAR_SNAPSHOT="$(resolve_snapshot "$AVATAR_CACHE_DIR" "$AVATAR_REPO_ID")"

printf 'Base snapshot:   %s\n' "$BASE_SNAPSHOT"
printf 'Avatar snapshot: %s\n' "$AVATAR_SNAPSHOT"

log "Creating weights directory"
mkdir -p "$WEIGHTS_DIR"

# The demo computes:
#   checkpoint_dir/../LongCat-Video
#
# Therefore LongCat-Video-Avatar-1.5 must be a real directory, not a directory
# symlink. Otherwise ".." resolves relative to the symlink target inside the
# Hugging Face cache instead of relative to ./weights.
BASE_LINK="${WEIGHTS_DIR}/LongCat-Video"
AVATAR_VIEW="${WEIGHTS_DIR}/LongCat-Video-Avatar-1.5"

log "Linking base model snapshot"
rm -rf "$BASE_LINK"
ln -s "$BASE_SNAPSHOT" "$BASE_LINK"

log "Creating real Avatar wrapper directory"
rm -rf "$AVATAR_VIEW"
mkdir -p "$AVATAR_VIEW"

# Do not copy the snapshot's internal symlinks. Those are relative to the
# snapshot's own blobs directory and would break if copied elsewhere.
# Instead, create absolute symlinks for each top-level entry back into the
# original cached snapshot.
shopt -s nullglob dotglob
avatar_items=("$AVATAR_SNAPSHOT"/*)

if (( ${#avatar_items[@]} == 0 )); then
  die "Avatar snapshot is empty: $AVATAR_SNAPSHOT"
fi

for item in "${avatar_items[@]}"; do
  name="$(basename "$item")"
  ln -s "$item" "${AVATAR_VIEW}/${name}"
done

shopt -u nullglob dotglob

log "Validating required files"

required_paths=(
  "${BASE_LINK}/tokenizer/tokenizer_config.json"
  "${AVATAR_VIEW}/scheduler/scheduler_config.json"
  "${AVATAR_VIEW}/vocal_separator/Kim_Vocal_2.onnx"
)

for path in "${required_paths[@]}"; do
  [[ -e "$path" ]] || die "Required path missing: $path"
  printf 'OK: %s\n' "$path"
done

if [[ -d "${AVATAR_VIEW}/base_model_int8" ]]; then
  printf 'OK: %s\n' "${AVATAR_VIEW}/base_model_int8"
elif [[ -d "${AVATAR_VIEW}/base_model" ]]; then
  printf 'WARNING: base_model_int8 not found; base_model exists instead.\n'
else
  die "Neither base_model_int8 nor base_model exists in the Avatar snapshot."
fi

if [[ -d "${AVATAR_VIEW}/whisper-large-v3" ]]; then
  printf 'OK: %s\n' "${AVATAR_VIEW}/whisper-large-v3"
else
  die "Whisper model directory missing: ${AVATAR_VIEW}/whisper-large-v3"
fi

log "Resolved local layout"
printf '%s -> %s\n' "$BASE_LINK" "$(readlink -f "$BASE_LINK")"
printf '%s is a real directory\n' "$AVATAR_VIEW"

log "Exact demo lookup check"
DEMO_BASE_LOOKUP="${AVATAR_VIEW}/../LongCat-Video/tokenizer/tokenizer_config.json"

[[ -f "$DEMO_BASE_LOOKUP" ]] || \
  die "The demo's sibling lookup still fails: $DEMO_BASE_LOOKUP"

printf 'OK: %s\n' "$DEMO_BASE_LOOKUP"

log "Local weight setup completed"
printf '\nUse:\n'
printf '  --checkpoint_dir=%s/LongCat-Video-Avatar-1.5\n\n' "$WEIGHTS_DIR"
