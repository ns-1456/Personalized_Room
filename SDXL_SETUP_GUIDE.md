# SDXL Setup Guide for Limited Disk Space

## Current Situation
- **Need:** ~10.3GB for SDXL base model
- **Have:** ~6.7GB free disk space
- **Gap:** Need ~3.6GB more

## Solutions (Choose One or Combine)

### Solution 1: Free Up Disk Space (Quickest)

Run the helper script:
```bash
./free_disk_space.sh
```

Or manually:
```bash
# Clear Hugging Face cache
rm -rf ~/.cache/huggingface

# Clear pip cache
rm -rf ~/Library/Caches/pip

# Clear browser caches (Chrome, Safari, etc.)
# Check Downloads folder for large files
# Empty Trash
```

**Expected:** Can free 2-5GB easily

### Solution 2: Use External Storage

If you have an external drive or more space elsewhere:

1. **Option A: Move Hugging Face cache**
   ```bash
   # Create cache on external drive
   mkdir -p /Volumes/YourDrive/.cache/huggingface
   
   # Set environment variable (add to ~/.zshrc)
   export HF_HOME=/Volumes/YourDrive/.cache/huggingface
   ```

2. **Option B: Use symbolic link**
   ```bash
   # Move cache to external drive
   mv ~/.cache/huggingface /Volumes/YourDrive/
   
   # Create symbolic link
   ln -s /Volumes/YourDrive/huggingface ~/.cache/huggingface
   ```

### Solution 3: Download Models Selectively

SDXL models are large, but you can:
- Download only what you need
- Use `low_cpu_mem_usage=True` in from_pretrained()
- Models will download incrementally as needed

### Solution 4: Use Google Drive Cache (If Available)

Since your project is in Google Drive, you could:
1. Download models to Google Drive folder
2. Set `HF_HOME` to point there
3. Models sync to cloud automatically

**Warning:** This might sync 10GB+ to cloud, check your quota!

## Memory Optimizations (Already Added)

The app now includes:
- ✅ Maximum attention slicing
- ✅ CPU offloading (models moved to CPU when not in use)
- ✅ VAE slicing and tiling
- ✅ Float32 for CPU stability

## Recommended Steps

1. **First, free up space:**
   ```bash
   ./free_disk_space.sh
   ```

2. **Check if you have enough space now:**
   ```bash
   df -h ~
   ```

3. **If still not enough, use external storage:**
   - Connect external drive
   - Update `HF_HOME` in app.py or set environment variable

4. **Run the app:**
   ```bash
   python3 app.py
   ```

## Expected Behavior

- **First run:** Will download ~10GB models (takes 10-30 minutes)
- **Subsequent runs:** Models load from cache (much faster)
- **Memory usage:** ~8-12GB RAM during generation
- **Generation time:** 2-5 minutes per image on CPU

## Troubleshooting

**If you get "No space left on device":**
- Free up more space
- Use external storage
- Consider using SD 1.5 instead (4GB models)

**If system crashes due to RAM:**
- Close other applications
- Reduce image resolution to 768x768
- Use fewer inference steps (20 instead of 30)

**If download fails:**
- Check internet connection
- Retry - downloads can resume
- Clear partial downloads: `rm -rf ~/.cache/huggingface/hub/models--stabilityai--stable-diffusion-xl-base-1.0`

