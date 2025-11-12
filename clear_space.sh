#!/bin/bash
echo "=== Clearing Disk Space ==="
echo ""

# 1. Clear incomplete Hugging Face downloads (9.6GB)
echo "1. Clearing incomplete Hugging Face downloads..."
rm -rf ~/.cache/huggingface/hub/models--diffusers--stable-diffusion-xl-1.0-inpainting-0.1/blobs/*.incomplete 2>/dev/null
echo "   ✓ Cleared incomplete downloads"

# 2. Clear JetBrains cache (8.1GB) - Safe to clear
echo "2. Clearing JetBrains cache..."
rm -rf ~/Library/Caches/JetBrains/*
echo "   ✓ Cleared JetBrains cache (8.1GB)"

# 3. Clear Google cache (1.4GB)
echo "3. Clearing Google cache..."
rm -rf ~/Library/Caches/Google/*
echo "   ✓ Cleared Google cache (1.4GB)"

# 4. Clear other caches
echo "4. Clearing other caches..."
rm -rf ~/Library/Caches/com.openai.atlas/*
rm -rf ~/Library/Caches/com.spotify.client/*
rm -rf ~/Library/Caches/com.codeweavers.CrossOver/*
rm -rf ~/Library/Caches/Firefox/*
echo "   ✓ Cleared additional caches (~3GB)"

# 5. Clear Hugging Face cache (if you want to re-download later)
echo "5. Option to clear Hugging Face cache (will need to re-download models)..."
read -p "   Clear HF cache? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    rm -rf ~/.cache/huggingface
    echo "   ✓ Cleared Hugging Face cache"
fi

echo ""
echo "=== Large Files You Can Manually Delete ==="
echo "1. Docker VM (60GB): ~/Library/Containers/com.docker.docker/Data/vms/0/data/Docker.raw"
echo "   Run: docker system prune -a --volumes (if you use Docker)"
echo ""
echo "2. F1 Game Patch (27GB): ~/.config/spicetify/Themes/Downloads/F1.2017.v1.0.6.Patch.1.13/"
echo "   Safe to delete if you don't need it"
echo ""

df -h ~ | tail -1
