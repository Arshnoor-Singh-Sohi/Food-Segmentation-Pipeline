#!/usr/bin/env python3
"""
Fine-tune model specifically for ingredient counting
"""

from ultralytics import YOLO
import yaml
from pathlib import Path

def create_ingredient_dataset_config():
    """Create dataset config for ingredient counting"""
    
    config = {
        'path': 'data/ingredient_dataset',
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        
        # Classes for individual ingredients
        'names': {
            0: 'banana_single',
            1: 'apple_single',
            2: 'orange_single',
            3: 'carrot_single',
            4: 'tomato_single',
            5: 'strawberry_single',
            6: 'grape_cluster',
            7: 'broccoli_floret'
        }
    }
    
    # Save config
    config_path = Path('data/ingredient_counting_config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    print(f"✅ Created config: {config_path}")
    return config_path

def prepare_training_data():
    """Instructions for preparing training data"""
    
    print("\n📚 PREPARING TRAINING DATA FOR INGREDIENT COUNTING")
    print("="*50)
    print("\nYou need to:")
    print("1. Collect images with individual ingredients")
    print("2. Label each banana/apple/etc. separately")
    print("3. Use tools like Roboflow or LabelImg")
    print("\nExample structure:")
    print("data/ingredient_dataset/")
    print("├── images/")
    print("│   ├── train/")
    print("│   │   ├── bananas_001.jpg")
    print("│   │   └── bananas_001.txt  (YOLO labels)")
    print("│   └── val/")
    print("└── labels/")
    
    print("\nLabel format (YOLO):")
    print("class_id x_center y_center width height")
    print("0 0.5 0.5 0.2 0.3  (for a single banana)")

def fine_tune_for_counting(base_model_path: str = None):
    """Fine-tune model for ingredient counting"""
    
    # Load base model
    if base_model_path and Path(base_model_path).exists():
        print(f"🔧 Starting from custom model: {base_model_path}")
        model = YOLO(base_model_path)
    else:
        print("🔧 Starting from YOLOv8n")
        model = YOLO('yolov8n.pt')
    
    # Training parameters for counting
    training_args = {
        'data': 'data/ingredient_counting_config.yaml',
        'epochs': 100,
        'imgsz': 640,
        'batch': 16,
        'name': 'ingredient_counter',
        'project': 'runs/ingredient_counting',
        'patience': 20,
        'save': True,
        'plots': True,
        
        # Specific for counting
        'box': 7.5,  # Higher box loss weight
        'cls': 1.5,  # Higher classification weight
        'overlap_mask': False,  # Don't merge overlapping detections
        'agnostic_nms': False,  # Keep class-specific NMS
    }
    
    print("\n🚀 Training parameters:")
    for key, value in training_args.items():
        print(f"  {key}: {value}")
    
    # Note: This would actually train if data exists
    print("\n⚠️  To actually train:")
    print("1. Prepare dataset as shown above")
    print("2. Run: model.train(**training_args)")
    
    return model

def main():
    print("🎯 INGREDIENT COUNTING MODEL SETUP")
    print("="*50)
    
    # Create config
    config_path = create_ingredient_dataset_config()
    
    # Show data preparation steps
    prepare_training_data()
    
    # Show how to fine-tune
    fine_tune_for_counting()
    
    print("\n✅ Setup complete!")
    print("\nNext steps:")
    print("1. Prepare labeled data with individual items")
    print("2. Run actual training")
    print("3. Use new model for accurate counting")

if __name__ == "__main__":
    main()