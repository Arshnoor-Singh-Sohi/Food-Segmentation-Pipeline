#!/usr/bin/env python3
"""
Quick start training script - run this to begin training your first custom model
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Import your training modules (we'll create these next)
from src.training.food_yolo_trainer import FoodYOLOTrainer
from src.training.food_dataset_preparer import FoodDatasetPreparer

def main():
    """Start training with minimal setup"""
    
    print("[FOOD] Starting Food YOLO Training Pipeline")
    print("=" * 50)
    
    # Step 1: Prepare a small dataset for testing
    print("📦 Preparing test dataset...")
    preparer = FoodDatasetPreparer()
    dataset_yaml, food_categories = preparer.create_sample_dataset()
    
    # Step 2: Start training
    print("[RUN] Starting training...")
    trainer = FoodYOLOTrainer(dataset_yaml)
    model, results = trainer.train_food_detector(epochs=10)  # Quick test run
    
    print("[OK] Training complete! Check the results in food_training_runs/")
    print(f"[STATS] Trained on {len(food_categories)} food categories")
    
    # Step 3: Test the model
    test_image = "data/input/image1.jpg"
    if Path(test_image).exists():
        print(f"[TEST] Testing on {test_image}...")
        results = model(test_image)
        print(f"[TARGET] Detected {len(results[0].boxes)} items")
    
    print("[SUCCESS] Your first custom food model is ready!")

if __name__ == "__main__":
    main()
