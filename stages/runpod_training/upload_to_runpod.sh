#!/bin/bash
# RunPod Data Upload Script

echo "📤 RUNPOD DATA UPLOAD FOR INDIVIDUAL ITEM TRAINING"
echo "=================================================="

# Check if RunPod CLI is installed
if ! command -v runpod &> /dev/null; then
    echo "❌ RunPod CLI not found. Install with:"
    echo "pip install runpod"
    exit 1
fi

# Upload training files
echo "📦 Uploading training configuration..."
runpod upload individual_item_config.yaml
runpod upload train_individual_items.py
runpod upload Dockerfile

# Upload base model (your 99.5% accuracy model)
echo "📦 Uploading custom base model..."
if [ -f "../data/models/custom_food_detection.pt" ]; then
    runpod upload ../data/models/custom_food_detection.pt custom_food_detection.pt
    echo "✅ Custom model uploaded"
else
    echo "⚠️ Custom model not found - will use YOLOv8n as base"
fi

# Upload dataset (if exists)
if [ -d "individual_item_dataset" ]; then
    echo "📦 Uploading training dataset..."
    runpod upload individual_item_dataset/
    echo "✅ Dataset uploaded"
else
    echo "⚠️ No dataset found - follow DATA_PREPARATION_GUIDE.md first"
fi

echo ""
echo "🚀 Upload complete! Next steps:"
echo "1. Start RunPod instance with GPU"
echo "2. Run: python train_individual_items.py"
echo "3. Monitor training progress"
echo "4. Download enhanced model when complete"
