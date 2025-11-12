# Deploying to Hugging Face Spaces

Quick guide to deploy this app to Hugging Face Spaces with free GPU access.

## Step-by-Step Instructions

### 1. Create Hugging Face Account
- Go to https://huggingface.co/join
- Sign up and verify your email

### 2. Create a New Space
- Go to https://huggingface.co/new-space
- Fill in:
  - **Space name**: `personalized-room-ai` (or your choice)
  - **SDK**: Select **Gradio**
  - **Hardware**: Select **GPU T4 small** (free) or **GPU A10G** (paid, faster)
  - **Visibility**: Public or Private
- Click **"Create Space"**

### 3. Clone Your Space Repository
```bash
git clone https://huggingface.co/spaces/YOUR_USERNAME/personalized-room-ai
cd personalized-room-ai
```

### 4. Copy Files to Space
Copy these files to your Space repository:
- `app_spaces.py` → rename to `app.py` (or just copy `app_spaces.py` as `app.py`)
- `requirements.txt`
- `README.md` (optional, but recommended)

If you have a trained adapter:
- Upload `my-room-editor-qlora/` folder via the web interface (Files tab) or git

### 5. Push to Space
```bash
git add .
git commit -m "Initial deployment"
git push
```

### 6. Wait for Build
- Go to your Space page
- Click the **"Logs"** tab
- Wait 5-10 minutes for the build to complete
- Models will download automatically (~10GB)

### 7. Your App is Live!
Your app will be available at:
```
https://YOUR_USERNAME-personalized-room-ai.hf.space
```

## File Structure for Spaces

```
personalized-room-ai/
├── app.py              # Main Gradio app (use app_spaces.py)
├── requirements.txt    # Dependencies
├── README.md          # Space description (optional)
└── my-room-editor-qlora/  # Your adapter (if you have one)
    ├── adapter_config.json
    └── adapter_model.safetensors
```

## Important Notes

1. **First Launch**: Takes 5-10 minutes to download models
2. **GPU Selection**: 
   - Free tier: T4 small (16GB RAM, slower)
   - Paid tier ($9/month): A10G (faster, more RAM)
3. **Adapter Path**: Make sure `my-room-editor-qlora` is in the root directory
4. **Updates**: Just `git push` to update your Space

## Troubleshooting

### Build Fails
- Check the Logs tab for errors
- Verify `requirements.txt` has all dependencies
- Make sure `app.py` exists in the root

### Out of Memory
- Use sequential loading (already implemented in `app_spaces.py`)
- Consider upgrading to A10G GPU

### Adapter Not Loading
- Verify adapter folder is uploaded
- Check that path in code matches folder name
- Look at Logs tab for error messages

## Updating Your Space

To update your Space:
```bash
cd personalized-room-ai
# Make your changes
git add .
git commit -m "Update description"
git push
```

Spaces automatically rebuild on push!

