"""Simple test without emojis."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from models.fast_yolo_segmentation import FastFoodSegmentation
    
    processor = FastFoodSegmentation(model_size='n')
    print("SUCCESS: YOLO processor initialized!")
    print("Ready to process images!")
    
except Exception as e:
    print(f"ERROR: {e}")