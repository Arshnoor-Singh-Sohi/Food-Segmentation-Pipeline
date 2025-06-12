#!/usr/bin/env python3
"""
Direct Segmentation Training - No Config Dependencies
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from ultralytics import YOLO

def train_direct():
    """Train segmentation directly with hardcoded parameters"""
    
    print("DIRECT SEGMENTATION TRAINING")
    print("Using hardcoded CPU parameters to avoid config issues")
    print("=" * 50)
    
    # Check dataset
    dataset_path = Path("data/training/food_training/existing_images_dataset.yaml")
    if not dataset_path.exists():
        print(f"ERROR: Dataset not found at {dataset_path}")
        print("Please run dataset preparation first!")
        return
    
    print(f"Dataset: {dataset_path}")
    
    # Load model
    model = YOLO('yolov8n-seg.pt')
    print("Loaded YOLOv8 segmentation model")
    
    # Direct training with explicit CPU parameters
    results = model.train(
        data=str(dataset_path),
        epochs=25,
        batch=4,
        device='cpu',  # Explicit CPU - no ambiguity
        project='direct_segmentation',
        name='food_seg_cpu',
        patience=10,
        workers=1,
        task='segment',
        lr0=0.001,
        optimizer='AdamW',
        verbose=True
    )
    
    print("Training completed!")
    return results

if __name__ == "__main__":
    train_direct()
