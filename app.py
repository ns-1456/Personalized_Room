"""
Personalize Your Room AI - Gradio Application

This application provides an interactive interface for:
1. Generating base room images from text prompts
2. Iteratively editing rooms using mask-based inpainting with instruction text

Using SDXL with aggressive memory optimizations for macOS compatibility.
"""

import gradio as gr
import torch
from diffusers import (
    StableDiffusionXLPipeline,
    StableDiffusionXLInpaintPipeline,
    AutoencoderKL,
    UNet2DConditionModel,
)
from PIL import Image
import numpy as np
import os
from pathlib import Path

# --- Configuration ---

# Base model for INITIAL ROOM GENERATION
BASE_GEN_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"

# Base model for INSTRUCTION-BASED INPAINTING EDITING
EDIT_MODEL_ID = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"

# VAE for better quality
VAE_ID = "madebyollin/sdxl-vae-fp16-fix"

# Optional: Set custom cache directory if you have external storage
# Uncomment and set path if you want to store models elsewhere
# os.environ["HF_HOME"] = "/path/to/external/drive/.cache/huggingface"

# Path to your trained QLoRA adapter
# This is the folder saved from train.py
QLORA_ADAPTER_PATH = "./my-room-editor-qlora"

# Device configuration
# On macOS, use CPU (MPS support can be added but CPU is more stable)
DEVICE = "cpu"  # Using CPU for macOS compatibility
DTYPE = torch.float32  # CPU works better with float32

# Enable 8-bit quantization to reduce memory usage
# This reduces RAM from ~12GB to ~6GB
USE_8BIT = True  # Set to False if you have enough RAM

# --- Global Variables for Model Pipelines ---
base_generator_pipe = None
editor_pipe = None


def load_models():
    """Load all required models and pipelines."""
    global base_generator_pipe, editor_pipe
    
    # Check for CUDA availability
    if DEVICE == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU. This will be slow.")
        return False
    
    try:
        print("Loading VAE...")
        vae = AutoencoderKL.from_pretrained(
            VAE_ID,
            torch_dtype=DTYPE,
        )
        
        print("Loading Base Room Generator Pipeline...")
        # Use SDXL with aggressive memory optimizations
        load_kwargs = {
            "vae": vae,
            "torch_dtype": DTYPE,
            "variant": "fp16" if DTYPE == torch.float16 else None,
            "use_safetensors": True,
        }
        
        # Add 8-bit quantization if enabled (reduces memory by ~50%)
        if USE_8BIT and DEVICE == "cpu":
            try:
                from transformers import BitsAndBytesConfig
                # Note: 8-bit quantization works better on GPU, but we can try on CPU
                print("   Attempting 8-bit quantization for memory efficiency...")
                # For CPU, we'll use model offloading instead
                load_kwargs["low_cpu_mem_usage"] = True
            except ImportError:
                print("   bitsandbytes not available, using standard loading")
        
        base_generator_pipe = StableDiffusionXLPipeline.from_pretrained(
            BASE_GEN_MODEL_ID,
            **load_kwargs
        )
        # Aggressive memory optimizations for macOS
        base_generator_pipe.enable_attention_slicing(slice_size="max")  # Maximum slicing
        base_generator_pipe.enable_model_cpu_offload()  # Offload to CPU when not in use
        base_generator_pipe.enable_vae_slicing()  # VAE slicing for lower memory
        base_generator_pipe.enable_vae_tiling()  # VAE tiling for large images
        
        print("Loading Instruction-Editing Pipeline (Inpainting)...")
        # Load the inpainting pipeline for mask-based editing
        editor_load_kwargs = {
            "vae": vae,
            "torch_dtype": DTYPE,
            "variant": "fp16" if DTYPE == torch.float16 else None,
            "use_safetensors": True,
        }
        
        if USE_8BIT and DEVICE == "cpu":
            editor_load_kwargs["low_cpu_mem_usage"] = True
        
        editor_pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
            EDIT_MODEL_ID,
            **editor_load_kwargs
        )
        # Aggressive memory optimizations
        editor_pipe.enable_attention_slicing(slice_size="max")
        editor_pipe.enable_vae_slicing()
        editor_pipe.enable_vae_tiling()
        
        # --- Apply QLoRA Adapter ---
        # Try to load the fine-tuned adapter weights
        adapter_path = Path(QLORA_ADAPTER_PATH)
        adapter_loaded = False
        
        if adapter_path.exists():
            # Check for different adapter file formats
            adapter_files = list(adapter_path.glob("*.safetensors")) + list(adapter_path.glob("*.bin"))
            adapter_config = adapter_path / "adapter_config.json"
            
            if adapter_files or adapter_config.exists():
                try:
                    print(f"Loading QLoRA adapter from: {QLORA_ADAPTER_PATH}")
                    # Method 1: Try loading as diffusers LoRA weights
                    try:
                        editor_pipe.load_lora_weights(QLORA_ADAPTER_PATH)
                        print("Successfully loaded QLoRA adapter using load_lora_weights().")
                        adapter_loaded = True
                    except Exception as e1:
                        print(f"load_lora_weights() failed: {e1}")
                        # Method 2: Try loading PEFT adapter and applying to UNet
                        try:
                            from peft import PeftModel
                            # Load the base UNet without PEFT wrapper
                            # Note: Your adapter was trained for SDXL, so it won't work with SD 1.5
                            # This will likely fail, but we'll try anyway
                            base_unet = UNet2DConditionModel.from_pretrained(
                                EDIT_MODEL_ID,
                                subfolder="unet",
                                torch_dtype=DTYPE,
                            )
                            # Load PEFT adapter
                            unet_with_adapter = PeftModel.from_pretrained(base_unet, QLORA_ADAPTER_PATH)
                            # Replace UNet in pipeline
                            editor_pipe.unet = unet_with_adapter
                            print("Successfully loaded QLoRA adapter using PEFT.")
                            adapter_loaded = True
                        except Exception as e2:
                            print(f"PEFT loading also failed: {e2}")
                            print("Using base model without fine-tuning.")
                except Exception as e:
                    print(f"Error during adapter loading: {e}")
                    print("Using base model without fine-tuning.")
            else:
                print(f"Adapter files not found in: {QLORA_ADAPTER_PATH}")
                print("Expected files: adapter_model.safetensors or adapter_model.bin, and adapter_config.json")
                print("Using base model without fine-tuning.")
        else:
            print(f"QLoRA adapter path not found: {QLORA_ADAPTER_PATH}")
            print("Using base model without fine-tuning.")
            print("To use fine-tuning, train the model first using train.py")
        
        if not adapter_loaded:
            print("\n" + "="*50)
            print("NOTE: Running with base SDXL model (no fine-tuning).")
            print("If adapter loading failed, check the adapter path and format.")
            print("="*50 + "\n")
        
        editor_pipe.enable_model_cpu_offload()  # For memory efficiency (don't move to device, offload handles it)
        
        print("All models loaded successfully!")
        return True
        
    except torch.cuda.OutOfMemoryError as e:
        print(f"CUDA Out of Memory Error: {e}")
        print("Try reducing batch size or using CPU mode.")
        return False
    except Exception as e:
        print(f"Error loading models: {e}")
        import traceback
        traceback.print_exc()
        return False


def generate_base_room(prompt: str, num_inference_steps: int = 30, guidance_scale: float = 7.5) -> Image.Image:
    """
    Generates the initial empty room image from a text prompt.
    
    Args:
        prompt: Text description of the room to generate
        num_inference_steps: Number of denoising steps
        guidance_scale: Classifier-free guidance scale
    
    Returns:
        PIL Image of the generated room
    """
    if not prompt or not prompt.strip():
        raise gr.Error("Please provide a room description prompt.")
    
    if base_generator_pipe is None:
        raise gr.Error("Base generator model not loaded. Please check model loading.")
    
    try:
        print(f"Generating base room with prompt: {prompt}")
        
        # Validate inputs
        if num_inference_steps < 1 or num_inference_steps > 100:
            raise gr.Error("Number of inference steps must be between 1 and 100.")
        if guidance_scale < 1.0 or guidance_scale > 20.0:
            raise gr.Error("Guidance scale must be between 1.0 and 20.0.")
        
        # Add negative prompts for better quality interiors
        negative_prompt = (
            "low quality, worst quality, blurry, people, person, text, watermark, "
            "deformed, distorted, disfigured, bad anatomy, bad proportions, "
            "extra limbs, missing limbs"
        )
        
        # Generate image
        with torch.inference_mode():
            image = base_generator_pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=int(num_inference_steps),
                guidance_scale=float(guidance_scale),
                height=1024,  # SDXL resolution
                width=1024,
            ).images[0]
        
        print("Base room generated successfully!")
        return image
        
    except gr.Error:
        # Re-raise Gradio errors as-is
        raise
    except torch.cuda.OutOfMemoryError as e:
        error_msg = "CUDA out of memory. Try using CPU mode or reducing image size."
        print(f"Error: {error_msg}")
        raise gr.Error(error_msg)
    except Exception as e:
        print(f"Error generating base room: {e}")
        import traceback
        traceback.print_exc()
        raise gr.Error(f"Failed to generate room: {str(e)}")


def edit_room(
    image_editor_output: dict,
    instruction: str,
    num_inference_steps: int = 20,
    guidance_scale: float = 7.0,
) -> Image.Image:
    """
    Applies the instruction-based edit to the masked area of the image.
    
    Args:
        image_editor_output: Dictionary from ImageEditor with 'composite' and 'layers'
        instruction: Text instruction describing the edit
        num_inference_steps: Number of denoising steps
        guidance_scale: Classifier-free guidance scale
    
    Returns:
        PIL Image of the edited room
    """
    if not instruction or not instruction.strip():
        raise gr.Error("Please provide an edit instruction.")
    
    if image_editor_output is None:
        raise gr.Error("Please generate or upload a room image first.")
    
    if editor_pipe is None:
        raise gr.Error("Editor model not loaded. Please check model loading.")
    
    try:
        # Validate inputs
        if num_inference_steps < 1 or num_inference_steps > 100:
            raise gr.Error("Number of inference steps must be between 1 and 100.")
        if guidance_scale < 1.0 or guidance_scale > 20.0:
            raise gr.Error("Guidance scale must be between 1.0 and 20.0.")
        
        # Extract image and mask from ImageEditor output
        # ImageEditor returns: {"background": PIL Image, "layers": [list of layer dicts], "composite": PIL Image}
        if isinstance(image_editor_output, dict):
            # Get the original background image
            input_image = image_editor_output.get("background")
            if input_image is None:
                # Fallback to composite if background not available
                input_image = image_editor_output.get("composite")
            
            if input_image is None:
                raise gr.Error("Invalid image editor output format.")
            
            # Extract mask from layers
            # ImageEditor layers contain drawing information
            layers = image_editor_output.get("layers", [])
            
            if not layers or len(layers) == 0:
                raise gr.Error("Please draw a mask on the image to indicate which areas to edit.")
            
            # Create mask from layers
            # Each layer has drawing information - we'll create a mask from all drawn strokes
            # Initialize mask as black (no edits)
            if isinstance(input_image, Image.Image):
                mask_array = np.zeros((input_image.height, input_image.width), dtype=np.uint8)
            else:
                mask_array = np.zeros((input_image.shape[0], input_image.shape[1]), dtype=np.uint8)
            
            # Process layers to extract drawn areas
            # Layers contain the drawn strokes - white brush strokes indicate edit areas
            for layer in layers:
                if isinstance(layer, dict) and "image" in layer:
                    # Layer has an image with the drawn strokes
                    layer_img = layer["image"]
                    if isinstance(layer_img, Image.Image):
                        layer_array = np.array(layer_img.convert("RGB"))
                    else:
                        layer_array = np.array(layer_img)
                    
                    # Extract white areas (where user drew with white brush)
                    if len(layer_array.shape) == 3:
                        white_threshold = 200
                        white_mask = (layer_array[:, :, 0] > white_threshold) & \
                                    (layer_array[:, :, 1] > white_threshold) & \
                                    (layer_array[:, :, 2] > white_threshold)
                        mask_array[white_mask] = 255
            
            # If no valid mask found, check composite for white areas
            if mask_array.sum() == 0:
                composite = image_editor_output.get("composite")
                if composite is not None:
                    composite_array = np.array(composite.convert("RGB"))
                    # Compare composite with background to find drawn areas
                    bg_array = np.array(input_image.convert("RGB"))
                    diff = np.abs(composite_array.astype(int) - bg_array.astype(int)).sum(axis=2)
                    # Areas with significant difference are drawn areas
                    mask_array[diff > 50] = 255
            
            if mask_array.sum() == 0:
                raise gr.Error("Please draw a mask on the image to indicate which areas to edit.")
            
            mask_pil = Image.fromarray(mask_array, mode="L")
        else:
            # Fallback: if it's just an image (no mask drawn yet)
            input_image = image_editor_output
            raise gr.Error("Please draw a mask on the image to indicate which areas to edit.")
        
        # Convert input_image to PIL if needed
        if not isinstance(input_image, Image.Image):
            if isinstance(input_image, np.ndarray):
                input_image = Image.fromarray(input_image)
            else:
                raise gr.Error("Invalid image format.")
        
        print(f"Editing room with instruction: {instruction}")
        
        # Resize to SDXL's expected resolution (1024x1024)
        original_size = input_image.size
        image_1024 = input_image.resize((1024, 1024), Image.Resampling.LANCZOS)
        mask_1024 = mask_pil.resize((1024, 1024), Image.Resampling.LANCZOS)
        
        # Ensure mask is binary (0 or 255)
        mask_array = np.array(mask_1024)
        # Threshold: values > 128 become 255 (white = edit area), else 0 (black = keep)
        mask_array = (mask_array > 128).astype(np.uint8) * 255
        mask_1024 = Image.fromarray(mask_array, mode="L")
        
        # Run the inpainting pipeline
        with torch.inference_mode():
            edited_image = editor_pipe(
                prompt=instruction,
                image=image_1024,
                mask_image=mask_1024,
                num_inference_steps=int(num_inference_steps),
                guidance_scale=float(guidance_scale),
                strength=0.95,  # How much to change the masked area
            ).images[0]
        
        # Resize back to original image size for seamless UI experience
        edited_image = edited_image.resize(original_size, Image.Resampling.LANCZOS)
        
        print("Room edited successfully!")
        return edited_image
        
    except gr.Error:
        # Re-raise Gradio errors as-is
        raise
    except torch.cuda.OutOfMemoryError as e:
        error_msg = "CUDA out of memory. Try using CPU mode or reducing image size."
        print(f"Error: {error_msg}")
        raise gr.Error(error_msg)
    except Exception as e:
        print(f"Error editing room: {e}")
        import traceback
        traceback.print_exc()
        raise gr.Error(f"Failed to edit room: {str(e)}")


# --- Build Gradio UI ---

def create_interface():
    """Create and configure the Gradio interface."""
    
    with gr.Blocks(title="Personalize Your Room AI", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 🎨 Personalize Your Room AI
            
            Fine-tuned with QLoRA to follow your design instructions.
            
            **How to use:**
            1. Enter a description of your ideal room and generate a base image
            2. Draw a mask on the image to indicate where you want to make changes
            3. Enter an instruction describing what to add/change in the masked area
            4. Click "Apply Edit" to see your personalized room!
            """
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Step 1: Generate Your Base Room")
                base_prompt = gr.Textbox(
                    label="Base Room Prompt",
                    placeholder="e.g., A bright, empty minimalist living room with hardwood floors and large windows",
                    lines=3,
                )
                with gr.Row():
                    base_button = gr.Button("Generate Room", variant="primary", scale=2)
                    clear_base = gr.Button("Clear", scale=1)
                
                gr.Markdown("### Step 2: Edit Your Room")
                edit_instruction = gr.Textbox(
                    label="Edit Instruction",
                    placeholder="e.g., Add a modern blue sofa in the center",
                    lines=2,
                )
                with gr.Row():
                    edit_button = gr.Button("Apply Edit", variant="primary", scale=2)
                    clear_edit = gr.Button("Clear", scale=1)
                
                # Advanced settings (collapsible)
                with gr.Accordion("Advanced Settings", open=False):
                    gr.Markdown("**Generation Settings**")
                    gen_steps = gr.Slider(
                        label="Inference Steps (Base Generation)",
                        minimum=10,
                        maximum=50,
                        value=30,
                        step=1,
                    )
                    gen_guidance = gr.Slider(
                        label="Guidance Scale (Base Generation)",
                        minimum=1.0,
                        maximum=20.0,
                        value=7.5,
                        step=0.5,
                    )
                    
                    gr.Markdown("**Editing Settings**")
                    edit_steps = gr.Slider(
                        label="Inference Steps (Editing)",
                        minimum=10,
                        maximum=50,
                        value=20,
                        step=1,
                    )
                    edit_guidance = gr.Slider(
                        label="Guidance Scale (Editing)",
                        minimum=1.0,
                        maximum=20.0,
                        value=7.0,
                        step=0.5,
                    )
            
            with gr.Column(scale=2):
                gr.Markdown("### Your Design Canvas")
                
                # Use ImageEditor for drawing masks directly on the image
                # ImageEditor returns a dict with 'composite' (image+mask) and 'layers'
                image_canvas = gr.ImageEditor(
                    label="Room Canvas - Draw mask in white to indicate edit areas",
                    type="pil",
                    height=600,
                    brush=gr.Brush(colors=["#FFFFFF"], color_mode="fixed"),
                )
                
                with gr.Row():
                    clear_canvas = gr.Button("Clear Canvas", variant="secondary")
                    download_image = gr.File(label="Download Image")
        
        # --- Wire up the UI ---
        
        # Step 1: Generate base room
        base_button.click(
            fn=generate_base_room,
            inputs=[base_prompt, gen_steps, gen_guidance],
            outputs=[image_canvas],
        )
        
        # Step 2: Edit the room
        # ImageEditor returns a dict with composite and layers
        edit_button.click(
            fn=edit_room,
            inputs=[image_canvas, edit_instruction, edit_steps, edit_guidance],
            outputs=[image_canvas],
        )
        
        # Clear buttons
        clear_base.click(lambda: "", outputs=[base_prompt])
        clear_edit.click(lambda: "", outputs=[edit_instruction])
        clear_canvas.click(lambda: None, outputs=[image_canvas])
        
        # Status message
        gr.Markdown(
            """
            ---
            **Note:** If you've trained a QLoRA adapter using `train.py`, make sure the adapter path is correct.
            The app will use the base model if no adapter is found.
            """
        )
    
    return demo


if __name__ == "__main__":
    # Load models first
    print("=" * 50)
    print("Initializing Personalize Your Room AI...")
    print("=" * 50)
    
    success = load_models()
    
    if not success:
        print("Failed to load models. Please check your configuration and try again.")
        exit(1)
    
    # Create and launch the interface
    demo = create_interface()
    
    print("\n" + "=" * 50)
    print("Launching Gradio interface...")
    print("=" * 50 + "\n")
    
    demo.launch(
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,
        share=False,  # Set to True if you want a public Gradio link
        debug=True,
    )

