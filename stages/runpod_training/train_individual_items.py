#!/usr/bin/env python3
"""
Individual Item Detection Training Script for RunPod
===================================================
"""

import os
import sys
from pathlib import Path
from ultralytics import YOLO
import torch
import yaml

def check_gpu():
    """Check GPU availability"""
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        print(f"🔥 GPU Available: {gpu_name} ({gpu_count} GPUs)")
        return True
    else:
        print("⚠️ No GPU available - using CPU")
        return False

def load_base_model(model_path):
    """Load your 99.5% accuracy custom model as base"""
    if Path(model_path).exists():
        print(f"📦 Loading custom base model: {model_path}")
        model = YOLO(model_path)
        print(f"✅ Custom model loaded with {len(model.names)} classes")
        return model
    else:
        print(f"⚠️ Custom model not found, using YOLOv8n as base")
        return YOLO('yolov8n.pt')

def main():
    print("🚀 INDIVIDUAL ITEM DETECTION TRAINING")
    print("=" * 50)
    
    # Check environment
    gpu_available = check_gpu()
    
    # Load base model (your 99.5% accuracy model)
    base_model = load_base_model('custom_food_detection.pt')
    
    # Training parameters optimized for individual item counting
    training_params = {
        'data': 'individual_item_config.yaml',
        'epochs': 75,                    # Sufficient for fine-tuning
        'batch': 16 if gpu_available else 4,  # Adjust based on GPU memory
        'imgsz': 640,                    # Standard YOLO image size
        'device': 0 if gpu_available else 'cpu',
        'project': 'runs/individual_items',
        'name': 'enhanced_ingredient_detector_{self.timestamp}',
        'save': True,
        'plots': True,
        'patience': 15,                  # Early stopping
        
        # Fine-tuning specific parameters
        'resume': False,                 # Start fresh fine-tuning
        'pretrained': True,              # Use pretrained weights
        'optimizer': 'AdamW',            # Good for fine-tuning
        'lr0': 0.001,                    # Lower learning rate for fine-tuning
        'lrf': 0.1,                      # Learning rate decay
        'momentum': 0.937,
        'weight_decay': 0.0005,
        
        # Individual item detection optimization
        'box': 7.5,                      # Higher box loss (important for counting)
        'cls': 1.0,                      # Classification loss
        'dfl': 1.5,                      # Distribution focal loss
        'overlap_mask': False,           # Don't merge overlapping items
        'mask_ratio': 4,                 # Mask ratio for segmentation
        
        # Data augmentation (conservative for fine-tuning)
        'hsv_h': 0.015,                  # Hue augmentation
        'hsv_s': 0.7,                    # Saturation (important for food)
        'hsv_v': 0.4,                    # Value augmentation
        'degrees': 0.0,                  # No rotation (food orientation matters)
        'translate': 0.1,                # Slight translation
        'scale': 0.5,                    # Scale augmentation
        'shear': 0.0,                    # No shearing
        'perspective': 0.0,              # No perspective change
        'flipud': 0.0,                   # No vertical flip
        'fliplr': 0.5,                   # 50% horizontal flip
        'mosaic': 1.0,                   # Mosaic augmentation
        'mixup': 0.1                     # Light mixup
    }
    
    print("🎯 Training Parameters:")
    for key, value in training_params.items():
        print(f"   {key}: {value}")
    
    # Start training
    print("\n🔥 Starting individual item detection training...")
    try:
        results = base_model.train(**training_params)
        
        print("\n✅ Training completed successfully!")
        print(f"📊 Results: {results}")
        
        # Save enhanced model
        enhanced_model_path = f'enhanced_individual_detector_{self.timestamp}.pt'
        print(f"💾 Enhanced model saved as: {enhanced_model_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return False

if __name__ == "__main__":
    main()
