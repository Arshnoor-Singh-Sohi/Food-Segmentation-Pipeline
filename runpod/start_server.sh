#!/bin/bash
# RunPod startup script

echo "🚀 Starting MealLens Metadata Extraction Services..."

# Navigate to project directory
cd /workspace/food-segmentation-pipeline

# Install requirements
pip install -r runpod/requirements_gpu.txt

# Build databases if they don't exist
if [ ! -f "data/databases/nutrition/nutrition_expanded.db" ]; then
    echo "📊 Building databases..."
    python scripts/build_all_databases.py
fi

# Download models if needed
echo "📦 Checking models..."
python -c "
from transformers import AutoModelForImageClassification
try:
    model = AutoModelForImageClassification.from_pretrained('nateraw/food')
    print('✅ Food classifier ready')
except:
    print('❌ Failed to load food classifier')
"

# Start Jupyter Lab in background
echo "📓 Starting Jupyter Lab..."
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root &

# Start API server
echo "🌐 Starting API server..."
python src/api/metadata_api.py &

echo "✅ All services started!"
echo "Jupyter: http://localhost:8888"
echo "API: http://localhost:5000"

# Keep container running
tail -f /dev/null