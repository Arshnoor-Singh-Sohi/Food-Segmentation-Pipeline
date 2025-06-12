#!/usr/bin/env python3
"""
Segmentation Training with Fixed Dataset
Uses the properly formatted segmentation labels
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from ultralytics import YOLO

def train_segmentation_fixed():
    """Train segmentation with properly formatted dataset"""
    
    print("SEGMENTATION TRAINING - FIXED DATASET")
    print("Using properly formatted segmentation labels")
    print("=" * 50)
    
    # Use the segmentation-specific dataset YAML
    dataset_path = Path("data/training/food_training/food_segmentation_dataset.yaml")
    if not dataset_path.exists():
        print(f"ERROR: Segmentation dataset not found at {dataset_path}")
        print("Please run fix_segmentation_dataset.py first!")
        return
    
    print(f"Dataset: {dataset_path}")
    
    # Load segmentation model
    model = YOLO('yolov8n-seg.pt')
    print("Loaded YOLOv8 segmentation model")
    
    # Training with segmentation dataset
    results = model.train(
        data=str(dataset_path),
        epochs=25,
        batch=4,
        device='cpu',
        project='segmentation_training_fixed',
        name='food_seg_fixed',
        patience=10,
        workers=1,
        task='segment',
        lr0=0.001,
        optimizer='AdamW',
        verbose=True,
        # Segmentation-specific parameters
        mask_ratio=4,
        overlap_mask=True
    )
    
    print("Segmentation training completed!")
    print("Model saved in: segmentation_training_fixed/food_seg_fixed/weights/best.pt")
    return results

if __name__ == "__main__":
    train_segmentation_fixed()
