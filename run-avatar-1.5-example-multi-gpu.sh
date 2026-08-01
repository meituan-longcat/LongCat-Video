CUDA_VISIBLE_DEVICES=0,1,2,3 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
uv run torchrun \
   --nproc_per_node=4 \
   run_demo_avatar_single_audio_to_video.py \
   --context_parallel_size=4 \
   --checkpoint_dir=./weights/LongCat-Video-Avatar-1.5 \
   --stage_1=ai2v \
   --input_json=assets/avatar/single_example_1.json \
   --use_distill \
   --model_type=avatar-v1.5 \
   --use_int8 \
   --staged_preprocessing
