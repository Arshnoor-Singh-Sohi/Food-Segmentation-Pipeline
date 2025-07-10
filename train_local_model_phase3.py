#!/usr/bin/env python3
"""
Phase 3: Local Model Training
===========================

Train local model using GenAI-generated dataset
Dr. Niaki's Phase 3 implementation
"""

from ultralytics import YOLO
from pathlib import Path

def train_local_model():
    """Train local model with GenAI-generated data"""
    print("🚀 PHASE 3: LOCAL MODEL TRAINING")
    print("Using GenAI-generated perfect dataset")
    
    # Load base model
    model = YOLO('yolov8n.pt')
    
    # Training parameters optimized for individual food items
    results = model.train(
        data='data/training_dataset_phase2/dataset.yaml',
        epochs=50,
        batch=8,
        imgsz=640,
        device='cpu',  # Change to '0' for GPU
        project='data/models',
        name='genai_trained_local_model',
        save=True,
        plots=True,
        
        # Optimized for individual item detection
        box=7.5,      # Higher box loss weight
        cls=1.0,      # Classification loss
        dfl=1.5,      # Distribution focal loss
        
        # Conservative augmentation
        hsv_s=0.7,    # Saturation for food freshness
        mosaic=1.0,   # Mosaic augmentation
        mixup=0.1     # Light mixup
    )
    
    print("✅ Training completed!")
    print(f"📊 Results: {results}")
    print("💾 Model saved to: data/models/genai_trained_local_model/")
    
    return results

if __name__ == "__main__":
    train_local_model()
