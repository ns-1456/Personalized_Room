# 🎨 Personalize Your Room AI

An AI-powered room design application using Stable Diffusion XL (SDXL) with QLoRA fine-tuning for personalized room generation and editing.

## Features

- **Base Room Generation**: Generate room images from text descriptions
- **Mask-Based Editing**: Draw masks and edit specific areas with text instructions
- **Fine-Tuned Model**: Uses QLoRA adapter for personalized instruction following
- **Memory Optimized**: Sequential model loading for lower RAM usage
- **Google Colab Ready**: Optimized notebook for cloud deployment

## 🚀 Live Demo

Try the app right now! No installation required:

**[👉 Try Live Demo](https://9b534637a8e4d78c17.gradio.live)**

The demo is running on Google Colab with GPU acceleration. Note that Gradio Spaces have a 72-hour timeout, so if the link is down, you can run it yourself using the instructions below.

## Quick Start

### Option 1: Google Colab (Recommended)

1. Open `app_colab.ipynb` in Google Colab
2. Enable GPU: Runtime → Change runtime type → GPU
3. Run all cells in order
4. Access the Gradio interface via the generated URL

**Expected generation time:**
- First generation: 30-60 seconds
- Subsequent generations: 15-30 seconds
- First edit: 20-40 seconds
- Subsequent edits: 10-20 seconds

### Option 2: Local Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
```

**Note:** Local installation requires ~12GB RAM and ~10GB disk space for SDXL models.

## Project Structure

```
Personalized_Room/
├── app.py                 # Local Gradio application
├── app_colab.ipynb        # Google Colab optimized version
├── train_colab.ipynb      # Training notebook for QLoRA fine-tuning
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Training Your Own Adapter

1. Open `train_colab.ipynb` in Google Colab
2. Prepare your dataset (images + edit instructions)
3. Run all cells to train the QLoRA adapter
4. Download the adapter folder (`my-room-editor-qlora`)
5. Place it in the same directory as `app.py` or update the path in `app_colab.ipynb`

## Configuration

### Model Paths

- **Base Generator**: `stabilityai/stable-diffusion-xl-base-1.0`
- **Editor (Inpainting)**: `diffusers/stable-diffusion-xl-1.0-inpainting-0.1`
- **VAE**: `madebyollin/sdxl-vae-fp16-fix`
- **QLoRA Adapter**: `./my-room-editor-qlora` (default)

### Memory Optimizations

The Colab version includes:
- Sequential model loading (only one model in memory at a time)
- XFormers memory-efficient attention
- Maximum attention slicing
- VAE slicing and tiling
- GPU memory clearing between operations

## Requirements

- Python 3.8+
- CUDA-capable GPU (recommended) or CPU
- 6-12GB RAM (depending on configuration)
- ~10GB disk space for models

## Dependencies

See `requirements.txt` for full list. Key packages:
- `torch` - PyTorch
- `diffusers` - Hugging Face Diffusers
- `transformers` - Hugging Face Transformers
- `peft` - Parameter-Efficient Fine-Tuning
- `gradio` - Web interface
- `xformers` - Memory-efficient attention (optional)

## Usage

### Generate Base Room

1. Enter a room description (e.g., "A modern living room with plants")
2. Adjust steps and guidance scale if needed
3. Click "Generate Room"
4. Wait 15-30 seconds for generation

### Edit Room

1. Draw a white mask on areas you want to edit
2. Enter an edit instruction (e.g., "Add a red sofa")
3. Click "Apply Edit"
4. Wait 10-20 seconds for editing

## Troubleshooting

### Out of Memory

- Use the Colab version (sequential loading)
- Reduce image resolution
- Enable more aggressive memory optimizations

### Slow Generation

- Use GPU instead of CPU
- Reduce number of inference steps
- Use Colab Pro for faster GPU (A100)

### Adapter Not Loading

- Verify adapter path is correct
- Check that adapter folder contains `adapter_config.json` and `adapter_model.safetensors`
- Ensure adapter was trained for SDXL inpainting model

## License

This project uses models with their respective licenses:
- SDXL: [CreativeML Open RAIL-M License](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)

## Acknowledgments

- Stable Diffusion XL by Stability AI
- Hugging Face Diffusers
- QLoRA by Tim Dettmers

