import os
import tempfile

import cv2
import torch
import streamlit as st
import numpy as np
from PIL import Image

from transformers import AutoTokenizer, UMT5EncoderModel
from diffusers.utils import export_to_video, load_image, load_video

from longcat_video.context_parallel import context_parallel_util
from longcat_video.pipeline_longcat_video import LongCatVideoPipeline
from longcat_video.modules.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from longcat_video.modules.autoencoder_kl_wan import AutoencoderKLWan
from longcat_video.modules.longcat_video_dit import LongCatVideoTransformer3DModel


def torch_gc():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

# Page configuration
st.set_page_config(
    page_title="LongCatVideo Generator",
    page_icon="🎬",
    layout="wide"
)

def get_fps(video_path):
    cap = cv2.VideoCapture(video_path)
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    
    return original_fps

@st.cache_resource
def load_model(checkpoint_dir):
    """Load model, use cache to avoid reloading"""    
    # Check GPU availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.bfloat16 if device == "cuda" else torch.float32
    
    with st.spinner('Loading model...'):
        cp_split_hw = context_parallel_util.get_optimal_split(1)
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, subfolder="tokenizer", torch_dtype=torch_dtype)
        text_encoder = UMT5EncoderModel.from_pretrained(checkpoint_dir, subfolder="text_encoder", torch_dtype=torch_dtype)
        vae = AutoencoderKLWan.from_pretrained(checkpoint_dir, subfolder="vae", torch_dtype=torch_dtype)
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(checkpoint_dir, subfolder="scheduler", torch_dtype=torch_dtype)
        dit = LongCatVideoTransformer3DModel.from_pretrained(checkpoint_dir, subfolder="dit", cp_split_hw=cp_split_hw, torch_dtype=torch_dtype)

        pipe = LongCatVideoPipeline(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            vae=vae,
            scheduler=scheduler,
            dit=dit,
        )
        pipe.to(device)
        
        cfg_step_lora_path = os.path.join(checkpoint_dir, 'lora/cfg_step_lora.safetensors')
        pipe.dit.load_lora(cfg_step_lora_path, 'cfg_step_lora')

        refinement_lora_path = os.path.join(checkpoint_dir, 'lora/refinement_lora.safetensors')
        pipe.dit.load_lora(refinement_lora_path, 'refinement_lora')
    
    return pipe, device

def main():
    st.title("🎬 LongCatVideo Generator")
    st.markdown("Supports Text-to-Video (T2V), Image-to-Video (I2V), and Video Continuation (VC) generation")
    
    checkpoint_dir = st.text_input("Model Dir", "./weights/LongCat-Video")

    # Load model
    try:
        pipe, device = load_model(checkpoint_dir)
        st.success(f"Model loaded successfully! Device: {device}")
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        return

    with st.expander("💡 Example Prompts"):
        st.markdown("""
        **Text-to-Video (T2V) Example:**
        - In a realistic photography style, a white boy around seven or eight years old sits on a park bench, wearing a light blue T-shirt, denim shorts, and white sneakers. He holds an ice cream cone with vanilla and chocolate flavors, and beside him is a medium-sized golden Labrador. Smiling, the boy offers the ice cream to the dog, who eagerly licks it with its tongue. The sun is shining brightly, and the background features a green lawn and several tall trees, creating a warm and loving scene.
        
        **Image-to-Video (I2V) Example:**
        - A woman sits at a wooden table by the window in a cozy café. She reaches out with her right hand, picks up the white coffee cup from the saucer, and gently brings it to her lips to take a sip. After drinking, she places the cup back on the table and looks out the window, enjoying the peaceful atmosphere.
        
        **Video Continuation (VC) Example:**
        - A person rides a motorcycle along a long, straight road that stretches between a body of water and a forested hillside. The rider steadily accelerates, keeping the motorcycle centered between the guardrails, while the scenery passes by on both sides. The video captures the journey from the rider’s perspective, emphasizing the sense of motion and adventure.
        """)

    mode_options = {
        "t2v": "T2V (Text-to-Video)",
        "i2v": "I2V (Image-to-Video)", 
        "vc": "VC (Video Continuation)"
    }
    
    # Sidebar - select generation mode
    st.sidebar.title("⚙️ Settings")
    mode = st.sidebar.selectbox(
        "Select generation mode",
        options=list(mode_options.keys()),
        format_func=lambda x: mode_options[x]
    )

    use_distill = st.sidebar.checkbox("Enable Distill Mode (Faster Generation)", value=False)
    use_refine = st.sidebar.checkbox("Enable Super-Resolution Mode (Low-res first, then upsample)", value=False)

    st.sidebar.subheader("Generation Parameters")
    
    if mode != "t2v":
        resolution = st.sidebar.selectbox("Resolution", ["480p", "720p"], index=0)
    else:
        col1, col2 = st.sidebar.columns(2)
        with col1:
            height = st.number_input("Height", min_value=256, max_value=1024, value=480, step=64)
        with col2:
            width = st.number_input("Width", min_value=256, max_value=1024, value=832, step=64)
    
    num_frames = 93
    
    if use_distill:
        num_inference_steps = 16  # Distill mode: fixed 16 steps
        guidance_scale = 1.0
    else:
        num_inference_steps = 50  # Normal mode: fixed 50 steps
        guidance_scale = 4.0

    seed = st.sidebar.number_input("Random Seed", min_value=0, max_value=2**32-1, value=42)
    
    # Main interface
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📝 Input")
        
        # Prompt input
        prompt = st.text_area(
            "Positive Prompt",
            height=100,
            placeholder="Please enter text describing the video content..."
        )
        
        negative_prompt = st.text_area(
            "Negative Prompt",
            value="Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality",
            height=80,
            disabled=use_distill
        )
        
        # Show different input controls according to mode
        uploaded_file = None
        if mode == "i2v":
            uploaded_file = st.file_uploader(
                "Upload Image",
                type=['png', 'jpg', 'jpeg'],
                help="Supports PNG, JPG, JPEG formats"
            )
            if uploaded_file:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_container_width=True)
        
        elif mode == "vc":
            uploaded_file = st.file_uploader(
                "Upload Video",
                type=['mp4', 'avi', 'mov'],
                help="Supports MP4, AVI, MOV formats"
            )
            if uploaded_file:
                st.video(uploaded_file)
            
            num_cond_frames = 13
        
        # Generate button
        generate_btn = st.button("🎬 Generate", type="primary", width='stretch')
    
    with col2:
        st.subheader("🎥 Output")
        result_placeholder = st.empty()
        
        if generate_btn:
            if not prompt.strip():
                st.error("Please enter a prompt!")
                return
            
            if mode != "t2v" and uploaded_file is None:
                st.error(f"Please upload an {'image' if mode == 'i2v' else 'video'} file!")
                return
            
            # Set random seed
            generator = torch.Generator(device=device)
            generator.manual_seed(seed)
            
            # Generate video according to mode
            with st.spinner('Generating video, please wait...'):
                if mode == "t2v":
                    if use_distill:
                        pipe.dit.enable_loras(['cfg_step_lora'])
                    output = pipe.generate_t2v(
                        prompt=prompt,
                        negative_prompt=None if use_distill else negative_prompt,
                        height=height,
                        width=width,
                        num_frames=num_frames,
                        num_inference_steps=num_inference_steps,
                        use_distill=use_distill,
                        guidance_scale=guidance_scale,
                        generator=generator,
                    )[0]
                    pipe.dit.disable_all_loras()
                    torch_gc()

                    if use_refine:
                        pipe.dit.enable_loras(['refinement_lora'])
                        stage1_video = [(output[i] * 255).astype(np.uint8) for i in range(output.shape[0])]
                        stage1_video = [Image.fromarray(img) for img in stage1_video]
                        del output
                        pipe.dit.enable_bsa()
                        output = pipe.generate_refine(
                            prompt="",
                            stage1_video=stage1_video,
                            num_inference_steps=50,
                            generator=generator
                        )[0]
                        pipe.dit.disable_all_loras()
                        pipe.dit.disable_bsa()
                        torch_gc()
                
                elif mode == "i2v":
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
                        image.save(tmp_file.name)
                        input_image = load_image(tmp_file.name)
                    
                    if use_distill:
                        pipe.dit.enable_loras(['cfg_step_lora'])
                    output = pipe.generate_i2v(
                        image=input_image,
                        prompt=prompt,
                        negative_prompt=None if use_distill else negative_prompt,
                        resolution=resolution,
                        num_frames=num_frames,
                        num_inference_steps=num_inference_steps,
                        use_distill=use_distill,
                        guidance_scale=guidance_scale,
                        generator=generator
                    )[0]
                    pipe.dit.disable_all_loras()
                    torch_gc()

                    if use_refine:
                        pipe.dit.enable_loras(['refinement_lora'])
                        stage1_video = [(output[i] * 255).astype(np.uint8) for i in range(output.shape[0])]
                        stage1_video = [Image.fromarray(img) for img in stage1_video]
                        del output
                        pipe.dit.enable_bsa()
                        output = pipe.generate_refine(
                            image=input_image,
                            prompt="",
                            stage1_video=stage1_video,
                            num_cond_frames=1,
                            num_inference_steps=50,
                            generator=generator
                        )[0]
                        pipe.dit.disable_all_loras()
                        pipe.dit.disable_bsa()
                        torch_gc()
                
                elif mode == "vc":
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                        tmp_file.write(uploaded_file.read())
                        input_video = load_video(tmp_file.name)
                        current_fps = get_fps(tmp_file.name)
                    
                    target_fps = 15
                    stride = max(1, round(current_fps / target_fps))
                    if use_distill:
                        pipe.dit.enable_loras(['cfg_step_lora'])
                    output = pipe.generate_vc(
                        video=input_video[::stride],
                        prompt=prompt,
                        negative_prompt=None if use_distill else negative_prompt,
                        resolution=resolution,
                        num_frames=num_frames,
                        num_cond_frames=num_cond_frames,
                        num_inference_steps=num_inference_steps,
                        use_distill=use_distill,
                        guidance_scale=guidance_scale,
                        generator=generator,
                        use_kv_cache=True,
                        offload_kv_cache=False,
                        enhance_hf=False if use_distill else True
                    )[0]
                    pipe.dit.disable_all_loras()
                    torch_gc()

                    if use_refine:
                        pipe.dit.enable_loras(['refinement_lora'])
                        stage1_video = [(output[i] * 255).astype(np.uint8) for i in range(output.shape[0])]
                        stage1_video = [Image.fromarray(img) for img in stage1_video]
                        del output
                        target_fps = 30
                        stride = max(1, round(current_fps / target_fps))
                        pipe.dit.enable_bsa()
                        output = pipe.generate_refine(
                            video=input_video[::stride],
                            prompt="",
                            stage1_video=stage1_video,
                            num_cond_frames=num_cond_frames*2,
                            num_inference_steps=50,
                            generator=generator
                        )[0]
                        pipe.dit.disable_all_loras()
                        pipe.dit.disable_bsa()
                        torch_gc()
            
            # Save and display result
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as output_file:
                fps = 30 if use_refine else 15
                export_to_video(output, output_file.name, fps=fps)
                
                with result_placeholder.container():
                    st.success("Generation complete!")
                    st.video(output_file.name)
                    
                    # Provide download button
                    with open(output_file.name, 'rb') as f:
                        st.download_button(
                            label="📥 Download Video",
                            data=f.read(),
                            file_name=f"generated_video_{mode}_{seed}.mp4",
                            mime="video/mp4"
                        )


if __name__ == "__main__":
    main()