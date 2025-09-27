#!/bin/bash
"""
Cleanup script to remove Unsloth cache and temporary files
Run this before switching between single-GPU and multi-GPU training
"""

echo "Cleaning Unsloth cache and temporary files..."

# Remove Unsloth cache directories
if [ -d "/workspace/.cache/unsloth" ]; then
    echo "Removing Unsloth cache..."
    rm -rf /workspace/.cache/unsloth
fi

if [ -d "$HOME/.cache/unsloth" ]; then
    echo "Removing local Unsloth cache..."
    rm -rf $HOME/.cache/unsloth
fi

# Clear HuggingFace cache if specified
if [ "$1" == "--hf-cache" ]; then
    echo "Clearing HuggingFace cache..."
    if [ -d "/workspace/hub" ]; then
        rm -rf /workspace/hub/*
    fi
fi

# Clear PyTorch cache
python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null

# Remove output directories if specified
if [ "$1" == "--all" ] || [ "$2" == "--all" ]; then
    echo "Removing output directories..."
    rm -rf ./outputs/*
    rm -rf ./logs/*
fi

# Clear Python cache
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -name "*.pyc" -delete 2>/dev/null

echo "Cache cleanup completed!"
echo ""
echo "Usage:"
echo "  bash cleanup_cache.sh              # Basic cleanup"
echo "  bash cleanup_cache.sh --hf-cache   # Include HuggingFace cache"  
echo "  bash cleanup_cache.sh --all        # Remove all outputs and logs"
