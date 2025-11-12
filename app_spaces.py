"""
Personalize Your Room AI - Hugging Face Spaces Deployment

Optimized for Hugging Face Spaces with GPU access and sequential model loading.
"""

import gradio as gr
import torch
from diffusers import (
    StableDiffusionXLPipeline,
    StableDiffusionXLInpaintPipeline,
    AutoencoderKL,
)
from PIL import Image
import numpy as np
import gc
from pathlib import Path

# --- Configuration ---
BASE_GEN_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
EDIT_MODEL_ID = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
VAE_ID = "madebyollin/sdxl-vae-fp16-fix"
QLORA_ADAPTER_PATH = "./my-room-editor-qlora"

# Device configuration for Spaces (GPU if available)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
print(f"Using device: {DEVICE}, dtype: {DTYPE}")

# Global variables - sequential loading (only one model at a time)
base_generator_pipe = None
editor_pipe = None
vae = None


def load_base_generator():
    """Load base generator pipeline. Unloads editor if loaded."""
    global base_generator_pipe, editor_pipe, vae
    
    # Unload editor to free memory
    if editor_pipe is not None:
        print("Unloading editor pipeline to free memory...")
        del editor_pipe
        editor_pipe = None
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
    
    if base_generator_pipe is not None:
        return True
    
    try:
        print("Loading VAE...")
        if vae is None:
            vae = AutoencoderKL.from_pretrained(
                VAE_ID,
                torch_dtype=DTYPE,
            )
        
        print("Loading Base Generator Pipeline...")
        base_generator_pipe = StableDiffusionXLPipeline.from_pretrained(
            BASE_GEN_MODEL_ID,
            vae=vae,
            torch_dtype=DTYPE,
            variant="fp16" if DTYPE == torch.float16 else None,
            use_safetensors=True,
            low_cpu_mem_usage=True,
        )
        
        # Memory optimizations
        base_generator_pipe.enable_attention_slicing(slice_size="max")
        if DEVICE == "cuda":
            try:
                base_generator_pipe.enable_xformers_memory_efficient_attention()
            except:
                pass
        base_generator_pipe.enable_vae_slicing()
        base_generator_pipe.enable_vae_tiling()
        
        if DEVICE == "cuda":
            base_generator_pipe = base_generator_pipe.to(DEVICE)
        else:
            base_generator_pipe.enable_model_cpu_offload()
        
        print("✓ Base generator loaded!")
        return True
    except Exception as e:
        print(f"Error loading base generator: {e}")
        import traceback
        traceback.print_exc()
        return False


def load_editor():
    """Load editor pipeline. Unloads base generator if loaded."""
    global base_generator_pipe, editor_pipe, vae
    
    # Unload base generator to free memory
    if base_generator_pipe is not None:
        print("Unloading base generator to free memory...")
        del base_generator_pipe
        base_generator_pipe = None
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
    
    if editor_pipe is not None:
        return True
    
    try:
        print("Loading VAE...")
        if vae is None:
            vae = AutoencoderKL.from_pretrained(
                VAE_ID,
                torch_dtype=DTYPE,
            )
        
        print("Loading Editor Pipeline...")
        editor_pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
            EDIT_MODEL_ID,
            vae=vae,
            torch_dtype=DTYPE,
            variant="fp16" if DTYPE == torch.float16 else None,
            use_safetensors=True,
            low_cpu_mem_usage=True,
        )
        
        # Memory optimizations
        editor_pipe.enable_attention_slicing(slice_size="max")
        if DEVICE == "cuda":
            try:
                editor_pipe.enable_xformers_memory_efficient_attention()
            except:
                pass
        editor_pipe.enable_vae_slicing()
        editor_pipe.enable_vae_tiling()
        
        # Load QLoRA adapter if available
        adapter_path = Path(QLORA_ADAPTER_PATH)
        if adapter_path.exists():
            try:
                print(f"🎯 Loading fine-tuned QLoRA adapter from: {QLORA_ADAPTER_PATH}")
                editor_pipe.load_lora_weights(QLORA_ADAPTER_PATH)
                print("✅ Fine-tuned adapter loaded!")
            except Exception as e:
                print(f"Could not load adapter: {e}")
        else:
            print(f"⚠️  Adapter not found at: {QLORA_ADAPTER_PATH}")
            print("   Running with base model (not fine-tuned)")
        
        if DEVICE == "cuda":
            editor_pipe = editor_pipe.to(DEVICE)
        else:
            editor_pipe.enable_model_cpu_offload()
        
        print("✓ Editor loaded!")
        return True
    except Exception as e:
        print(f"Error loading editor: {e}")
        import traceback
        traceback.print_exc()
        return False


def generate_base_room(prompt: str, num_inference_steps: int = 30, guidance_scale: float = 7.5):
    """Generate base room image."""
    if not prompt or not prompt.strip():
        raise gr.Error("Please provide a room description prompt.")
    
    if not load_base_generator():
        raise gr.Error("Failed to load base generator model.")
    
    try:
        print(f"Generating base room: {prompt}")
        
        negative_prompt = (
            "low quality, worst quality, blurry, people, person, text, watermark, "
            "deformed, distorted, disfigured, bad anatomy, bad proportions"
        )
        
        with torch.inference_mode():
            image = base_generator_pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=int(num_inference_steps),
                guidance_scale=float(guidance_scale),
                height=1024,
                width=1024,
            ).images[0]
        
        print("✓ Base room generated!")
        return image
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        raise gr.Error(f"Failed to generate room: {str(e)}")


def edit_room(image_editor_output: dict, instruction: str, num_inference_steps: int = 20, guidance_scale: float = 7.0):
    """Edit room with mask-based inpainting."""
    if not instruction or not instruction.strip():
        raise gr.Error("Please provide an edit instruction.")
    
    if image_editor_output is None:
        raise gr.Error("Please generate or upload a room image first.")
    
    if not load_editor():
        raise gr.Error("Failed to load editor model.")
    
    try:
        # Extract image and mask from ImageEditor output
        if isinstance(image_editor_output, dict):
            input_image = image_editor_output.get("background") or image_editor_output.get("composite")
            layers = image_editor_output.get("layers", [])
            
            if input_image is None:
                raise gr.Error("Invalid image editor output.")
            
            if not layers:
                raise gr.Error("Please draw a mask on the image to indicate which areas to edit.")
            
            # Create mask from layers
            mask_array = np.zeros((input_image.height, input_image.width), dtype=np.uint8)
            
            for layer in layers:
                if isinstance(layer, dict) and "image" in layer:
                    layer_img = layer["image"]
                    if isinstance(layer_img, Image.Image):
                        layer_array = np.array(layer_img.convert("RGB"))
                    else:
                        layer_array = np.array(layer_img)
                    
                    if len(layer_array.shape) == 3:
                        white_threshold = 200
                        white_mask = (layer_array[:, :, 0] > white_threshold) & \
                                    (layer_array[:, :, 1] > white_threshold) & \
                                    (layer_array[:, :, 2] > white_threshold)
                        mask_array[white_mask] = 255
            
            # Fallback: extract from composite if no layers
            if mask_array.sum() == 0:
                composite = image_editor_output.get("composite")
                if composite:
                    composite_array = np.array(composite.convert("RGB"))
                    bg_array = np.array(input_image.convert("RGB"))
                    diff = np.abs(composite_array.astype(int) - bg_array.astype(int)).sum(axis=2)
                    mask_array[diff > 50] = 255
            
            if mask_array.sum() == 0:
                raise gr.Error("Please draw a mask on the image.")
            
            mask_pil = Image.fromarray(mask_array, mode="L")
        else:
            raise gr.Error("Please draw a mask on the image.")
        
        # Resize to 1024x1024
        original_size = input_image.size
        image_1024 = input_image.resize((1024, 1024), Image.Resampling.LANCZOS)
        mask_1024 = mask_pil.resize((1024, 1024), Image.Resampling.LANCZOS)
        
        # Binarize mask
        mask_array = np.array(mask_1024)
        mask_array = (mask_array > 128).astype(np.uint8) * 255
        mask_1024 = Image.fromarray(mask_array, mode="L")
        
        print(f"Editing room: {instruction}")
        
        # Run inpainting
        with torch.inference_mode():
            edited_image = editor_pipe(
                prompt=instruction,
                image=image_1024,
                mask_image=mask_1024,
                num_inference_steps=int(num_inference_steps),
                guidance_scale=float(guidance_scale),
                strength=0.95,
            ).images[0]
        
        edited_image = edited_image.resize(original_size, Image.Resampling.LANCZOS)
        print("✓ Room edited successfully!")
        return edited_image
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        raise gr.Error(f"Failed to edit room: {str(e)}")


def create_interface():
    with gr.Blocks(title="Personalize Your Room AI", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 🎨 Personalize Your Room AI
            
            **Memory-optimized deployment** - Only one model loaded at a time!
            
            **How to use:**
            1. Enter a room description and generate a base image
            2. Draw a white mask on areas you want to edit
            3. Enter an instruction and click "Apply Edit"
            """
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Generate Base Room")
                base_prompt = gr.Textbox(
                    label="Room Description",
                    placeholder="A modern living room with large windows, minimalist furniture, and plants",
                    lines=3,
                )
                gen_steps = gr.Slider(label="Steps", minimum=10, maximum=50, value=30, step=1)
                gen_guidance = gr.Slider(label="Guidance Scale", minimum=1.0, maximum=20.0, value=7.5, step=0.5)
                base_button = gr.Button("Generate Room", variant="primary")
                clear_base = gr.Button("Clear", variant="secondary")
                
                gr.Markdown("### Edit Room")
                edit_instruction = gr.Textbox(
                    label="Edit Instruction",
                    placeholder="Add a red sofa",
                    lines=2,
                )
                edit_steps = gr.Slider(label="Steps", minimum=10, maximum=50, value=20, step=1)
                edit_guidance = gr.Slider(label="Guidance Scale", minimum=1.0, maximum=20.0, value=7.0, step=0.5)
                edit_button = gr.Button("Apply Edit", variant="primary")
                clear_edit = gr.Button("Clear", variant="secondary")
            
            with gr.Column(scale=2):
                gr.Markdown("### Your Design Canvas")
                image_canvas = gr.ImageEditor(
                    label="Room Canvas - Draw white mask to indicate edit areas",
                    type="pil",
                    height=600,
                    brush=gr.Brush(colors=["#FFFFFF"], color_mode="fixed"),
                )
                clear_canvas = gr.Button("Clear Canvas", variant="secondary")
        
        # Wire up UI
        base_button.click(
            fn=generate_base_room,
            inputs=[base_prompt, gen_steps, gen_guidance],
            outputs=[image_canvas],
        )
        
        edit_button.click(
            fn=edit_room,
            inputs=[image_canvas, edit_instruction, edit_steps, edit_guidance],
            outputs=[image_canvas],
        )
        
        clear_base.click(lambda: "", outputs=[base_prompt])
        clear_edit.click(lambda: "", outputs=[edit_instruction])
        clear_canvas.click(lambda: None, outputs=[image_canvas])
    
    return demo


# For Hugging Face Spaces, use app.py as the entry point
if __name__ == "__main__":
    demo = create_interface()
    demo.launch()

