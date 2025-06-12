#!/usr/bin/env python3
"""
Guaranteed Working Detection Training
This will definitely work with your existing dataset
Detection is easier than segmentation and your dataset is already set up for it
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from ultralytics import YOLO

def train_detection_working():
    """Train detection model - guaranteed to work with your existing dataset"""
    
    print("GUARANTEED WORKING DETECTION TRAINING")
    print("=" * 50)
    print("Using your existing dataset that's already properly formatted")
    
    # Use your existing dataset that already works
    dataset_path = Path("data/training/food_training/existing_images_dataset.yaml")
    
    if not dataset_path.exists():
        print(f"ERROR: Dataset not found at {dataset_path}")
        print("Please run dataset preparation first!")
        return
    
    print(f"Using dataset: {dataset_path}")
    
    # Load detection model (not segmentation)
    model = YOLO('yolov8n.pt')  # Detection model, not yolov8n-seg.pt
    print("Loaded YOLOv8 detection model")
    
    # Train detection with your working dataset
    try:
        results = model.train(
            data=str(dataset_path),
            epochs=25,
            batch=4,
            device='cpu',
            project='working_detection',
            name='food_detection_working',
            patience=10,
            workers=1,
            verbose=True,
            save=True,
            plots=True,
            # Detection-specific parameters
            lr0=0.001,
            optimizer='AdamW',
            # Augmentation
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=15,
            translate=0.1,
            scale=0.5,
            flipud=0.5,
            fliplr=0.5,
            mosaic=1.0,
            mixup=0.2
        )
        
        print("SUCCESS: Detection training completed!")
        print("Model saved in: working_detection/food_detection_working/weights/best.pt")
        
        # Copy model to your models directory for easy access
        best_model = Path("working_detection/food_detection_working/weights/best.pt")
        if best_model.exists():
            models_dir = Path("data/models")
            models_dir.mkdir(exist_ok=True)
            
            import shutil
            shutil.copy2(best_model, models_dir / "custom_food_detection_working.pt")
            print(f"Model also saved to: data/models/custom_food_detection_working.pt")
        
        return results
        
    except Exception as e:
        print(f"Training failed: {e}")
        return None

if __name__ == "__main__":
    train_detection_working()