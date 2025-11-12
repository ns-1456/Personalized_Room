# 🚀 Deployment Guide

This guide covers deploying the Personalize Your Room AI app to various cloud platforms with GPU access.

## Option 1: Hugging Face Spaces (Recommended - Free GPU)

**Best for:** Free, persistent deployment with GPU access

### Steps:

1. **Create a Hugging Face account** (if you don't have one)
   - Go to https://huggingface.co/join
   - Verify your email

2. **Create a new Space**
   - Go to https://huggingface.co/new-space
   - Name: `personalized-room-ai` (or your choice)
   - SDK: **Gradio**
   - Hardware: **GPU T4 small** (free tier) or **GPU A10G** (paid)
   - Visibility: Public or Private
   - Click "Create Space"

3. **Upload your files**
   - Clone the Space repository:
     ```bash
     git clone https://huggingface.co/spaces/YOUR_USERNAME/personalized-room-ai
     cd personalized-room-ai
     ```
   - Copy these files:
     - `app.py` (use the Spaces-compatible version below)
     - `requirements.txt`
     - `README.md`
   - Upload your adapter folder (if you have one):
     - `my-room-editor-qlora/` (upload via web interface or git)

4. **Push to Space**
   ```bash
     git add .
     git commit -m "Initial deployment"
     git push
   ```

5. **Wait for build** (~5-10 minutes)
   - Hugging Face will automatically build and deploy
   - Check the "Logs" tab for progress
   - Your app will be live at: `https://YOUR_USERNAME-personalized-room-ai.hf.space`

### Advantages:
- ✅ Free GPU (T4 small)
- ✅ Persistent (doesn't disconnect)
- ✅ Automatic HTTPS
- ✅ Easy updates via git push
- ✅ Public or private deployment

### Limitations:
- Free tier: 16GB RAM, T4 GPU (slower than A100)
- Paid tier ($9/month): A10G GPU, more RAM

---

## Option 2: Replicate (Easy ML Deployment)

**Best for:** Simple deployment with good GPU access

### Steps:

1. **Create Replicate account**
   - Go to https://replicate.com/signin

2. **Deploy via API or Web**
   - Replicate is more API-focused
   - You can create a Cog model (Docker-based)
   - Or use their web interface for Gradio apps

### Advantages:
- ✅ Good GPU access
- ✅ Pay-per-use pricing
- ✅ Easy API integration

### Limitations:
- ⚠️ More complex setup
- ⚠️ Better for API than web UI

---

## Option 3: RunPod (Dedicated GPU Instances)

**Best for:** Full control, dedicated GPU

### Steps:

1. **Create RunPod account**
   - Go to https://www.runpod.io/
   - Add credits ($10 minimum)

2. **Create a Pod**
   - Choose GPU: RTX 3090, A100, etc.
   - Select template: "RunPod PyTorch"
   - Configure storage and networking

3. **Deploy your app**
   - SSH into the pod
   - Install dependencies
   - Run `python app.py`
   - Use ngrok or RunPod's networking for public access

### Advantages:
- ✅ Full control
- ✅ Powerful GPUs available
- ✅ Pay only when running

### Limitations:
- ⚠️ More technical setup
- ⚠️ Need to manage the instance
- ⚠️ Costs money (but cheap: ~$0.20-0.50/hour)

---

## Option 4: Vast.ai (Cheapest GPU Cloud)

**Best for:** Budget-friendly GPU access

### Steps:

1. **Create Vast.ai account**
   - Go to https://vast.ai/
   - Add credits

2. **Rent a GPU instance**
   - Search for RTX 3090 or A100
   - Select instance with good price/performance
   - SSH into instance

3. **Deploy app**
   - Similar to RunPod setup
   - Install dependencies and run

### Advantages:
- ✅ Very cheap ($0.10-0.30/hour)
- ✅ Good GPU selection

### Limitations:
- ⚠️ Less reliable than RunPod
- ⚠️ More technical setup
- ⚠️ Need to manage instance

---

## Option 5: AWS/GCP/Azure (Enterprise)

**Best for:** Production, scaling, enterprise needs

### Steps:

1. **Create cloud account**
2. **Launch GPU instance** (g4dn.xlarge on AWS, etc.)
3. **Deploy using Docker or directly**
4. **Set up load balancer and domain**

### Advantages:
- ✅ Highly reliable
- ✅ Scalable
- ✅ Enterprise features

### Limitations:
- ⚠️ More expensive
- ⚠️ Complex setup
- ⚠️ Overkill for personal projects

---

## Recommended: Hugging Face Spaces

For most users, **Hugging Face Spaces** is the best option:
- Free GPU tier
- Easy deployment
- Persistent hosting
- Built for ML apps

See the `app_spaces.py` file for a Spaces-compatible version of the app.

---

## Quick Comparison

| Platform | Cost | GPU | Setup Difficulty | Persistence |
|----------|------|-----|-----------------|-------------|
| **Hugging Face Spaces** | Free/Paid | T4/A10G | ⭐ Easy | ✅ Yes |
| **Replicate** | Pay-per-use | Various | ⭐⭐ Medium | ✅ Yes |
| **RunPod** | ~$0.20-0.50/hr | RTX 3090/A100 | ⭐⭐⭐ Hard | ⚠️ Manual |
| **Vast.ai** | ~$0.10-0.30/hr | Various | ⭐⭐⭐ Hard | ⚠️ Manual |
| **AWS/GCP** | ~$0.50-2/hr | Various | ⭐⭐⭐⭐ Very Hard | ✅ Yes |

