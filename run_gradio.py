import gradio as gr
import os
import torch
import tempfile
import uuid
import PIL.Image
import numpy as np

# --- Set TF32 for performance, as recommended by the warning log
if torch.cuda.is_available():
    torch.set_float32_matmul_precision('high')

from transformers import AutoTokenizer, UMT5EncoderModel
from torchvision.io import write_video
from diffusers.utils import load_image

from longcat_video.pipeline_longcat_video import LongCatVideoPipeline
from longcat_video.modules.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from longcat_video.modules.autoencoder_kl_wan import AutoencoderKLWan
from longcat_video.modules.longcat_video_dit import LongCatVideoTransformer3DModel

# --- Global State for Models ---
MODELS = {
    "pipe": None,
    "checkpoint_dir": None,
    "compiled": False
}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Utility Functions ---
def torch_gc():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def save_video(video_frames, fps=15, crf=18, codec="libx264"):
    """Saves a list of PIL Images or a NumPy array to a temporary MP4 file."""
    temp_dir = tempfile.gettempdir()
    video_path = os.path.join(temp_dir, f"{uuid.uuid4()}.mp4")

    if isinstance(video_frames, list) and isinstance(video_frames[0], PIL.Image.Image):
        video_tensor = torch.from_numpy(np.stack([np.array(img) for img in video_frames]))
    elif isinstance(video_frames, np.ndarray):
        if video_frames.dtype == np.float32 or video_frames.dtype == np.float64:
            video_frames = (video_frames * 255).astype(np.uint8)
        video_tensor = torch.from_numpy(video_frames)
    else:
        raise TypeError("Unsupported video_frames type. Must be a list of PIL Images or a NumPy array.")

    if video_tensor.dtype != torch.uint8:
        video_tensor = video_tensor.to(torch.uint8)
        
    write_video(video_path, video_tensor, fps=fps, video_codec=codec, options={"crf": str(crf)})
    return video_path


# --- Core Logic ---

def load_models_if_needed(checkpoint_dir, enable_compile):
    """
    Loads the pipeline and its components if they haven't been loaded yet
    or if the checkpoint directory/compile option has changed.
    """
    if MODELS["pipe"] is not None and MODELS["checkpoint_dir"] == checkpoint_dir and MODELS["compiled"] == enable_compile:
        return "Models already loaded."

    if not checkpoint_dir or not os.path.isdir(checkpoint_dir):
        raise gr.Error(f"Checkpoint directory '{checkpoint_dir}' not found. Please provide a valid path.")

    print(f"Loading models from: {checkpoint_dir}")
    MODELS["checkpoint_dir"] = checkpoint_dir
    MODELS["compiled"] = enable_compile
    
    # --- FIX: Provide [1, 1] for cp_split_hw ---
    # In a non-distributed setup, the "split" is 1x1. This prevents the TypeError.
    cp_split_hw = [1, 1] 

    try:
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, subfolder="tokenizer", torch_dtype=torch.bfloat16)
        text_encoder = UMT5EncoderModel.from_pretrained(checkpoint_dir, subfolder="text_encoder", torch_dtype=torch.bfloat16)
        vae = AutoencoderKLWan.from_pretrained(checkpoint_dir, subfolder="vae", torch_dtype=torch.bfloat16)
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(checkpoint_dir, subfolder="scheduler", torch_dtype=torch.bfloat16)
        dit = LongCatVideoTransformer3DModel.from_pretrained(checkpoint_dir, subfolder="dit", cp_split_hw=cp_split_hw, torch_dtype=torch.bfloat16)

        if enable_compile and hasattr(torch, 'compile'):
            print("Compiling DiT model...")
            dit = torch.compile(dit)

        pipe = LongCatVideoPipeline(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            vae=vae,
            scheduler=scheduler,
            dit=dit,
        )
        pipe.to(DEVICE)
        MODELS["pipe"] = pipe
        print("Models loaded successfully.")
        return "Models loaded successfully."
    except Exception as e:
        MODELS["pipe"] = None
        MODELS["checkpoint_dir"] = None
        raise gr.Error(f"Failed to load models: {e}")

# --- Tab 1: Image-to-Video Generation ---

def generate_i2v_ui(
    checkpoint_dir, enable_compile, image, prompt, negative_prompt,
    num_frames, steps, guidance_scale, seed
):
    status = load_models_if_needed(checkpoint_dir, enable_compile)
    yield None, None, None, status

    pipe = MODELS["pipe"]
    generator = torch.Generator(device=DEVICE).manual_seed(int(seed))
    
    if isinstance(image, np.ndarray):
        image = PIL.Image.fromarray(image)
        
    target_size = image.size

    # Stage 1: i2v (480p)
    yield None, None, None, "Generating base video..."
    output_base = pipe.generate_i2v(
        image=image, prompt=prompt, negative_prompt=negative_prompt,
        resolution='480p', num_frames=num_frames, num_inference_steps=steps,
        guidance_scale=guidance_scale, generator=generator
    )[0]
    
    output_base_pil = [(frame * 255).astype(np.uint8) for frame in output_base]
    output_base_pil = [PIL.Image.fromarray(img).resize(target_size, PIL.Image.BICUBIC) for img in output_base_pil]
    base_video_path = save_video(output_base_pil, fps=15, crf=18)
    yield base_video_path, None, None, "Base video generated. Starting distill..."

    # Stage 2: i2v distill (480p)
    cfg_step_lora_path = os.path.join(checkpoint_dir, 'lora/cfg_step_lora.safetensors')
    pipe.dit.load_lora(cfg_step_lora_path, 'cfg_step_lora')
    pipe.dit.enable_loras(['cfg_step_lora'])
    
    output_distill = pipe.generate_i2v(
        image=image, prompt=prompt, resolution='480p', num_frames=num_frames,
        num_inference_steps=steps, use_distill=True, guidance_scale=1.0, generator=generator
    )[0]
    pipe.dit.disable_all_loras()

    distill_video_pil = [(frame * 255).astype(np.uint8) for frame in output_distill]
    distill_video_pil = [PIL.Image.fromarray(img).resize(target_size, PIL.Image.BICUBIC) for img in distill_video_pil]
    distill_video_path = save_video(distill_video_pil, fps=15, crf=18)
    yield base_video_path, distill_video_path, None, "Distilled video generated. Starting refinement..."

    # Stage 3: i2v refinement (720p)
    refinement_lora_path = os.path.join(checkpoint_dir, 'lora/refinement_lora.safetensors')
    pipe.dit.load_lora(refinement_lora_path, 'refinement_lora')
    pipe.dit.enable_loras(['refinement_lora'])
    pipe.dit.enable_bsa()
    
    output_refine = pipe.generate_refine(
        image=image, prompt=prompt, stage1_video=distill_video_pil,
        num_cond_frames=1, num_inference_steps=50, generator=generator
    )[0]
    pipe.dit.disable_all_loras()
    pipe.dit.disable_bsa()

    output_refine_pil = [(frame * 255).astype(np.uint8) for frame in output_refine]
    output_refine_pil = [PIL.Image.fromarray(img).resize(target_size, PIL.Image.BICUBIC) for img in output_refine_pil]
    refine_video_path = save_video(output_refine_pil, fps=30, crf=10)
    
    yield base_video_path, distill_video_path, refine_video_path, "All stages complete!"
    torch_gc()

# --- Tab 2: Text-to-Video Generation ---

def generate_t2v_ui(
    checkpoint_dir, enable_compile, prompt, negative_prompt, height, width,
    num_frames, steps, guidance_scale, seed
):
    status = load_models_if_needed(checkpoint_dir, enable_compile)
    yield None, None, None, status

    pipe = MODELS["pipe"]
    generator = torch.Generator(device=DEVICE).manual_seed(int(seed))

    # Stage 1: t2v (480p)
    yield None, None, None, "Generating base video..."
    output_base = pipe.generate_t2v(
        prompt=prompt, negative_prompt=negative_prompt, height=height, width=width,
        num_frames=num_frames, num_inference_steps=steps, guidance_scale=guidance_scale,
        generator=generator
    )[0]
    base_video_path = save_video(output_base, fps=15, crf=18)
    yield base_video_path, None, None, "Base video generated. Starting distill..."

    # Stage 2: t2v distill (480p)
    cfg_step_lora_path = os.path.join(checkpoint_dir, 'lora/cfg_step_lora.safetensors')
    pipe.dit.load_lora(cfg_step_lora_path, 'cfg_step_lora')
    pipe.dit.enable_loras(['cfg_step_lora'])

    output_distill = pipe.generate_t2v(
        prompt=prompt, height=height, width=width, num_frames=num_frames,
        num_inference_steps=16, use_distill=True, guidance_scale=1.0, generator=generator
    )[0]
    pipe.dit.disable_all_loras()
    distill_video_path = save_video(output_distill, fps=15, crf=18)
    
    stage1_video_pil = [(frame * 255).astype(np.uint8) for frame in output_distill]
    stage1_video_pil = [PIL.Image.fromarray(img) for img in stage1_video_pil]
    yield base_video_path, distill_video_path, None, "Distilled video generated. Starting refinement..."
    
    # Stage 3: t2v refinement (720p)
    refinement_lora_path = os.path.join(checkpoint_dir, 'lora/refinement_lora.safetensors')
    pipe.dit.load_lora(refinement_lora_path, 'refinement_lora')
    pipe.dit.enable_loras(['refinement_lora'])
    pipe.dit.enable_bsa()

    output_refine = pipe.generate_refine(
        prompt=prompt, stage1_video=stage1_video_pil, num_inference_steps=50, generator=generator
    )[0]
    pipe.dit.disable_all_loras()
    pipe.dit.disable_bsa()
    refine_video_path = save_video(output_refine, fps=30, crf=10)

    yield base_video_path, distill_video_path, refine_video_path, "All stages complete!"
    torch_gc()

# --- Tab 3: Long Video Generation ---

def generate_long_video_ui(
    checkpoint_dir, enable_compile, prompt, negative_prompt, num_segments,
    num_frames, num_cond_frames, steps, guidance_scale, seed, progress=gr.Progress()
):
    status = load_models_if_needed(checkpoint_dir, enable_compile)
    yield None, None, status

    pipe = MODELS["pipe"]
    generator = torch.Generator(device=DEVICE).manual_seed(int(seed))

    # Stage 1: Initial T2V Generation
    progress(0, desc="Generating initial T2V segment...")
    output_t2v = pipe.generate_t2v(
        prompt=prompt, negative_prompt=negative_prompt, height=480, width=832,
        num_frames=num_frames, num_inference_steps=steps, guidance_scale=guidance_scale,
        generator=generator
    )[0]

    video_pil = [(frame * 255).astype(np.uint8) for frame in output_t2v]
    video_pil = [PIL.Image.fromarray(img) for img in video_pil]
    del output_t2v
    torch_gc()
    
    target_size = video_pil[0].size
    current_video = video_pil
    all_generated_frames = video_pil
    
    # Stage 2: Long Video Continuation
    for i in range(num_segments):
        progress((i + 1) / (num_segments * 2), desc=f"Generating continuation segment {i+1}/{num_segments}...")
        
        output_vc = pipe.generate_vc(
            video=current_video, prompt=prompt, negative_prompt=negative_prompt,
            resolution='480p', num_frames=num_frames, num_cond_frames=num_cond_frames,
            num_inference_steps=steps, guidance_scale=guidance_scale, generator=generator,
            use_kv_cache=True, offload_kv_cache=False, enhance_hf=True
        )[0]

        new_video = [(frame * 255).astype(np.uint8) for frame in output_vc]
        new_video = [PIL.Image.fromarray(img).resize(target_size, PIL.Image.BICUBIC) for img in new_video]
        del output_vc

        all_generated_frames.extend(new_video[num_cond_frames:])
        current_video = new_video

    base_long_video_path = save_video(all_generated_frames, fps=15, crf=18)
    yield base_long_video_path, None, "Base long video generated. Starting refinement..."

    # Stage 3: Long Video Refinement
    refinement_lora_path = os.path.join(checkpoint_dir, 'lora/refinement_lora.safetensors')
    pipe.dit.load_lora(refinement_lora_path, 'refinement_lora')
    pipe.dit.enable_loras(['refinement_lora'])
    pipe.dit.enable_bsa()

    cur_condition_video = None
    cur_num_cond_frames = 0
    start_id = 0
    all_refine_frames = []

    num_refine_segments = (len(all_generated_frames) - num_frames) // (num_frames - num_cond_frames) + 1
    for i in range(num_refine_segments):
        progress(0.5 + (i + 1) / (num_refine_segments * 2), desc=f"Refining segment {i+1}/{num_refine_segments}...")
        
        chunk = all_generated_frames[start_id : start_id + num_frames]
        if not chunk: break

        output_refine = pipe.generate_refine(
            video=cur_condition_video, prompt='', stage1_video=chunk,
            num_cond_frames=cur_num_cond_frames, num_inference_steps=50, generator=generator
        )[0]

        new_video_refined = [(frame * 255).astype(np.uint8) for frame in output_refine]
        new_video_refined = [PIL.Image.fromarray(img) for img in new_video_refined]
        del output_refine

        all_refine_frames.extend(new_video_refined[cur_num_cond_frames:])
        cur_condition_video = new_video_refined
        cur_num_cond_frames = num_cond_frames * 2
        start_id = start_id + num_frames - num_cond_frames

    refine_long_video_path = save_video(all_refine_frames, fps=30, crf=10)
    yield base_long_video_path, refine_long_video_path, "All stages complete!"
    torch_gc()

# --- Gradio UI ---

DEFAULT_NEGATIVE_PROMPT = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"

def create_ui():
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("# LongCat Video Generation UI")
        
        with gr.Row():
            checkpoint_dir = gr.Textbox(
                label="Checkpoint Directory", 
                # --- CHANGE: Set default value ---
                value="./weights/LongCat-Video",
                info="The pipeline will be loaded from this directory."
            )
            enable_compile = gr.Checkbox(label="Enable torch.compile()", value=True, info="May speed up generation but increases initial load time.")

        with gr.Tabs():
            # Tab 1: Image-to-Video
            with gr.TabItem("Image-to-Video"):
                with gr.Row():
                    with gr.Column(scale=1):
                        i2v_image = gr.Image(type="pil", label="Input Image", value="assets/girl.png")
                        i2v_prompt = gr.Textbox(label="Prompt", value="A woman sits at a wooden table by the window in a cozy café. She reaches out with her right hand, picks up the white coffee cup from the saucer, and gently brings it to her lips to take a sip. After drinking, she places the cup back on the table and looks out the window, enjoying the peaceful atmosphere.")
                        i2v_neg_prompt = gr.Textbox(label="Negative Prompt", value=DEFAULT_NEGATIVE_PROMPT)
                        with gr.Row():
                            i2v_frames = gr.Slider(label="Number of Frames", minimum=16, maximum=240, value=93, step=1)
                            i2v_steps = gr.Slider(label="Inference Steps (Base)", minimum=1, maximum=100, value=16, step=1)
                        with gr.Row():
                            i2v_cfg = gr.Slider(label="Guidance Scale (Base)", minimum=1.0, maximum=15.0, value=4.0, step=0.5)
                            i2v_seed = gr.Number(label="Seed", value=42, precision=0)
                        i2v_button = gr.Button("Generate Image-to-Video", variant="primary")
                    with gr.Column(scale=1):
                        i2v_status = gr.Textbox(label="Status", interactive=False)
                        gr.Markdown("### Outputs")
                        i2v_output_base = gr.Video(label="Base Output (480p)")
                        i2v_output_distill = gr.Video(label="Distilled Output (480p)")
                        i2v_output_refine = gr.Video(label="Refined Output (720p)")

            # Tab 2: Text-to-Video
            with gr.TabItem("Text-to-Video"):
                with gr.Row():
                    with gr.Column(scale=1):
                        t2v_prompt = gr.Textbox(label="Prompt", value="In a realistic photography style, a white boy around seven or eight years old sits on a park bench, wearing a light blue T-shirt, denim shorts, and white sneakers. He holds an ice cream cone with vanilla and chocolate flavors, and beside him is a medium-sized golden Labrador. Smiling, the boy offers the ice cream to the dog, who eagerly licks it with its tongue. The sun is shining brightly, and the background features a green lawn and several tall trees, creating a warm and loving scene.")
                        t2v_neg_prompt = gr.Textbox(label="Negative Prompt", value=DEFAULT_NEGATIVE_PROMPT)
                        with gr.Row():
                            t2v_height = gr.Slider(label="Height", minimum=256, maximum=1024, value=480, step=32)
                            t2v_width = gr.Slider(label="Width", minimum=256, maximum=1024, value=832, step=32)
                        with gr.Row():
                            t2v_frames = gr.Slider(label="Number of Frames", minimum=16, maximum=240, value=93, step=1)
                            t2v_steps = gr.Slider(label="Inference Steps (Base)", minimum=1, maximum=100, value=50, step=1)
                        with gr.Row():
                            t2v_cfg = gr.Slider(label="Guidance Scale (Base)", minimum=1.0, maximum=15.0, value=4.0, step=0.5)
                            t2v_seed = gr.Number(label="Seed", value=42, precision=0)
                        t2v_button = gr.Button("Generate Text-to-Video", variant="primary")
                    with gr.Column(scale=1):
                        t2v_status = gr.Textbox(label="Status", interactive=False)
                        gr.Markdown("### Outputs")
                        t2v_output_base = gr.Video(label="Base Output (480p)")
                        t2v_output_distill = gr.Video(label="Distilled Output (480p)")
                        t2v_output_refine = gr.Video(label="Refined Output (720p)")
            
            # Tab 3: Long Video
            with gr.TabItem("Long Video"):
                with gr.Row():
                    with gr.Column(scale=1):
                        lv_prompt = gr.Textbox(label="Prompt", lines=5, value="realistic filming style, a person wearing a dark helmet, a deep-colored jacket, blue jeans, and bright yellow shoes rides a skateboard along a winding mountain road...")
                        lv_neg_prompt = gr.Textbox(label="Negative Prompt", value=DEFAULT_NEGATIVE_PROMPT)
                        with gr.Row():
                            lv_segments = gr.Slider(label="Continuation Segments", minimum=0, maximum=20, value=5, step=1, info="Number of times to extend the video. 0 means only the initial T2V clip is generated.")
                            lv_frames = gr.Slider(label="Frames per Segment", minimum=16, maximum=128, value=93, step=1)
                        with gr.Row():
                            lv_cond_frames = gr.Slider(label="Conditional Frames", minimum=1, maximum=32, value=13, step=1, info="Number of frames from the previous segment to use for conditioning.")
                            lv_steps = gr.Slider(label="Inference Steps", minimum=1, maximum=100, value=50, step=1)
                        with gr.Row():
                            lv_cfg = gr.Slider(label="Guidance Scale", minimum=1.0, maximum=15.0, value=4.0, step=0.5)
                            lv_seed = gr.Number(label="Seed", value=42, precision=0)
                        lv_button = gr.Button("Generate Long Video", variant="primary")
                    with gr.Column(scale=1):
                        lv_status = gr.Textbox(label="Status", interactive=False)
                        gr.Markdown("### Outputs")
                        lv_output_base = gr.Video(label="Base Long Video (480p)")
                        lv_output_refine = gr.Video(label="Refined Long Video (720p)")

        # Event Handlers
        i2v_button.click(
            fn=generate_i2v_ui,
            inputs=[checkpoint_dir, enable_compile, i2v_image, i2v_prompt, i2v_neg_prompt, i2v_frames, i2v_steps, i2v_cfg, i2v_seed],
            outputs=[i2v_output_base, i2v_output_distill, i2v_output_refine, i2v_status]
        )
        t2v_button.click(
            fn=generate_t2v_ui,
            inputs=[checkpoint_dir, enable_compile, t2v_prompt, t2v_neg_prompt, t2v_height, t2v_width, t2v_frames, t2v_steps, t2v_cfg, t2v_seed],
            outputs=[t2v_output_base, t2v_output_distill, t2v_output_refine, t2v_status]
        )
        lv_button.click(
            fn=generate_long_video_ui,
            inputs=[checkpoint_dir, enable_compile, lv_prompt, lv_neg_prompt, lv_segments, lv_frames, lv_cond_frames, lv_steps, lv_cfg, lv_seed],
            outputs=[lv_output_base, lv_output_refine, lv_status]
        )
        
    return demo

if __name__ == "__main__":
    app = create_ui()
    app.launch(server_name="0.0.0.0", share=True)
