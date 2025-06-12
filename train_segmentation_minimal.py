#!/usr/bin/env python3
"""
Minimal Segmentation Training - Bypasses Config Issues
This script trains segmentation with hardcoded correct parameters
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ultralytics import YOLO

def train_segmentation_minimal():
    """Train segmentation with minimal, correct parameters"""
    
    print("MINIMAL SEGMENTATION TRAINING")
    print("=" * 40)
    
    # Check dataset exists
    dataset_path = Path("data/training/food_training/existing_images_dataset.yaml")
    if not dataset_path.exists():
        print(f"[ERROR] Dataset not found: {dataset_path}")
        print("Run the dataset preparation first!")
        return
    
    print(f"[DATASET] Using: {dataset_path}")
    
    # Load segmentation model
    print("[MODEL] Loading YOLOv8 segmentation model...")
    model = YOLO('yolov8n-seg.pt')
    
    # Training arguments with CORRECT parameter names
    train_args = {
        'data': str(dataset_path),
        'epochs': 25,  # Reduced for faster completion
        'batch': 4,    # CORRECT: 'batch' not 'batch_size'
        'device': 'cpu',
        'project': 'segmentation_training',
        'name': 'minimal_food_seg',
        'patience': 10,
        'workers': 1,
        'task': 'segment',
        # Basic augmentation
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'lr0': 0.001,
        'optimizer': 'AdamW'
    }
    
    print("[TRAINING] Starting segmentation training...")
    print("Parameters:")
    for key, value in train_args.items():
        print(f"  {key}: {value}")
    
    try:
        results = model.train(**train_args)
        print("[SUCCESS] Segmentation training completed!")
        return results
    except Exception as e:
        print(f"[ERROR] Training failed: {e}")
        return None

if __name__ == "__main__":
    train_segmentation_minimal()
