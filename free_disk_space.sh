#!/bin/bash
# Script to free up disk space for SDXL models

echo "=== Freeing Up Disk Space for SDXL Models ==="
echo ""

# Check current disk space
echo "Current disk space:"
df -h ~ | tail -1
echo ""

# Option 1: Clear Hugging Face cache
echo "1. Clearing Hugging Face cache..."
CACHE_SIZE=$(du -sh ~/.cache/huggingface 2>/dev/null | cut -f1)
if [ -d ~/.cache/huggingface ]; then
    rm -rf ~/.cache/huggingface
    echo "   ✓ Freed: $CACHE_SIZE"
else
    echo "   No cache found"
fi
echo ""

# Option 2: Clear pip cache
echo "2. Clearing pip cache..."
PIP_CACHE_SIZE=$(du -sh ~/Library/Caches/pip 2>/dev/null | cut -f1)
if [ -d ~/Library/Caches/pip ]; then
    rm -rf ~/Library/Caches/pip
    echo "   ✓ Freed: $PIP_CACHE_SIZE"
else
    echo "   No pip cache found"
fi
echo ""

# Option 3: Clear system caches
echo "3. Clearing system caches..."
SYSTEM_CACHE_SIZE=$(du -sh ~/Library/Caches 2>/dev/null | cut -f1)
echo "   System cache size: $SYSTEM_CACHE_SIZE"
echo "   (Be careful with system caches - only clear if you know what you're doing)"
echo ""

# Option 4: Check for large files
echo "4. Finding large files (>1GB) in home directory..."
find ~ -type f -size +1G -exec ls -lh {} \; 2>/dev/null | head -10
echo ""

# Final disk space
echo "Final disk space:"
df -h ~ | tail -1
echo ""
echo "=== Done ==="
echo ""
echo "If you still need more space, consider:"
echo "  - Moving models to external drive"
echo "  - Using cloud storage"
echo "  - Setting HF_HOME to external drive in app.py"

