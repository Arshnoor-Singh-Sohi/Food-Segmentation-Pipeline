#!/usr/bin/env python3
"""
Setup script to integrate YOLO training into existing food segmentation pipeline
Run this once to create the training infrastructure
"""

import os
import shutil
from pathlib import Path
import yaml

def create_training_structure():
    """Create the training directory structure"""
    
    # Define the new directories we need
    directories = [
        "src/training",
        "src/evaluation", 
        "data/training/raw_datasets",
        "data/training/food_training",
        "data/training/annotations",
        "data/training/splits",
        "data/trained_models/experiments",
        "scripts/training_scripts"
    ]
    
    # Create directories
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        
        # Create __init__.py files for Python packages
        if dir_path.startswith("src/"):
            init_file = Path(dir_path) / "__init__.py"
            init_file.touch()
    
    print("[OK] Training directory structure created")

def create_training_config():
    """Create training configuration files"""
    
    # Training configuration
    training_config = {
        'model': {
            'base_model': 'yolov8n.pt',
            'input_size': 640,
            'task': 'detect'  # or 'segment' for segmentation
        },
        'training': {
            'epochs': 50,  # Start small for testing
            'batch_size': 16,
            'patience': 20,
            'device': 'auto',
            'workers': 4,
            'project': 'food_training_runs',
            'name': 'food_detector_v1'
        },
        'augmentation': {
            'hsv_h': 0.015,
            'hsv_s': 0.7,
            'hsv_v': 0.4,
            'degrees': 15,
            'translate': 0.1,
            'scale': 0.5,
            'flipud': 0.5,
            'fliplr': 0.5,
            'mosaic': 1.0,
            'mixup': 0.2
        },
        'optimization': {
            'optimizer': 'AdamW',
            'lr0': 0.001,
            'lrf': 0.01,
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'warmup_epochs': 3.0,
            'box': 7.5,
            'cls': 0.5,
            'dfl': 1.5
        }
    }
    
    # Dataset configuration
    dataset_config = {
        'datasets': {
            'food101_subset': {
                'name': 'Food-101 Subset',
                'num_classes': 20,  # Start small
                'description': 'Subset of Food-101 for quick training',
                'categories': [
                    'apple_pie', 'beef_tartare', 'caesar_salad', 'chicken_curry',
                    'chocolate_cake', 'club_sandwich', 'fish_and_chips', 'french_fries',
                    'fried_rice', 'hamburger', 'ice_cream', 'pizza', 'ramen',
                    'spaghetti_carbonara', 'steak', 'sushi', 'tacos', 'tiramisu'
                ]
            }
        },
        'splits': {
            'train': 0.7,
            'val': 0.2, 
            'test': 0.1
        },
        'image_processing': {
            'min_size': 416,
            'max_size': 1024,
            'quality_threshold': 0.7
        }
    }
    
    # Save configurations
    config_dir = Path("config")
    config_dir.mkdir(exist_ok=True)
    
    with open(config_dir / "training_config.yaml", 'w') as f:
        yaml.dump(training_config, f, default_flow_style=False)
        
    with open(config_dir / "dataset_config.yaml", 'w') as f:
        yaml.dump(dataset_config, f, default_flow_style=False)
    
    print("[OK] Configuration files created")

def update_requirements():
    """Add training dependencies to requirements.txt"""
    
    training_deps = [
        "# Training dependencies",
        "ultralytics>=8.0.0",
        "torch>=2.0.0", 
        "torchvision>=0.15.0",
        "scikit-learn>=1.3.0",
        "tqdm>=4.65.0",
        "seaborn>=0.12.0",
        "wandb",  # For experiment tracking (optional)
        ""
    ]
    
    requirements_file = Path("requirements.txt")
    
    # Read existing requirements
    existing_deps = []
    if requirements_file.exists():
        with open(requirements_file, 'r') as f:
            existing_deps = f.read().splitlines()
    
    # Add training dependencies if not already present
    for dep in training_deps:
        if dep.startswith("#") or dep == "":
            existing_deps.append(dep)
        elif not any(dep.split(">=")[0] in line for line in existing_deps):
            existing_deps.append(dep)
    
    # Write updated requirements
    with open(requirements_file, 'w') as f:
        f.write("\n".join(existing_deps))
    
    print("[OK] Requirements.txt updated with training dependencies")

def create_quick_start_script():
    """Create a simple script to start training immediately"""
    
    quick_start = '''#!/usr/bin/env python3
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
'''
    
    with open("scripts/quick_start_training.py", 'w', encoding='utf-8') as f:
        f.write(quick_start)

    
    # Make it executable
    os.chmod("scripts/quick_start_training.py", 0o755)
    
    print("[OK] Quick start script created")

def main():
    """Setup the complete training infrastructure"""
    
    print("[TOOL] Setting up YOLO training integration...")
    print("=" * 50)
    
    create_training_structure()
    create_training_config() 
    update_requirements()
    create_quick_start_script()
    
    print("\n[SUCCESS] Setup complete!")
if __name__ == "__main__":
    main()