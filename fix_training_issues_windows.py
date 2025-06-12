#!/usr/bin/env python3
"""
Training Issues Troubleshooter - Windows Compatible Version
Run this script to automatically fix common training setup problems
NO UNICODE CHARACTERS - Pure ASCII for Windows compatibility
"""

import os
import shutil
from pathlib import Path
import yaml

def fix_training_setup():
    """Fix common training setup issues automatically"""
    
    print("TRAINING SETUP TROUBLESHOOTER")
    print("=" * 50)
    print("Automatically fixing common training issues...")
    
    fixes_applied = []
    
    # Fix 1: Create missing directories
    required_dirs = [
        "src/training",
        "src/evaluation", 
        "data/training/raw_datasets",
        "data/training/food_training",
        "data/training/annotations",
        "data/trained_models",
        "config",
        "data/input"
    ]
    
    print("\nCreating missing directories...")
    for dir_path in required_dirs:
        path = Path(dir_path)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            print(f"[CREATED] {dir_path}")
            fixes_applied.append(f"Created directory: {dir_path}")
        
        # Create __init__.py for Python packages
        if dir_path.startswith("src/"):
            init_file = path / "__init__.py"
            if not init_file.exists():
                init_file.touch()
                print(f"[CREATED] {init_file}")
    
    # Fix 2: Create training configuration if missing
    config_dir = Path("config")
    training_config_path = config_dir / "training_config.yaml"
    
    if not training_config_path.exists():
        print("\nCreating training configuration...")
        
        training_config = {
            'model': {
                'base_model': 'yolov8n.pt',
                'input_size': 640
            },
            'training': {
                'epochs': 50,
                'batch': 8,  # FIXED: Use 'batch' not 'batch_size'
                'patience': 20,
                'device': 'auto',
                'workers': 2,
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
        
        try:
            with open(training_config_path, 'w', encoding='utf-8') as f:
                yaml.dump(training_config, f, default_flow_style=False)
            
            print(f"[CREATED] {training_config_path}")
            fixes_applied.append("Created training configuration")
        except Exception as e:
            print(f"[ERROR] Could not create config: {e}")
    
    # Fix 3: Check for images in data/input
    input_dir = Path("data/input")
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    existing_images = []
    
    for ext in image_extensions:
        existing_images.extend(list(input_dir.glob(f"*{ext}")))
        existing_images.extend(list(input_dir.glob(f"*{ext.upper()}")))
    
    if len(existing_images) == 0:
        print(f"\n[WARNING] No images found in {input_dir}")
        print("IMPORTANT: You need to add food images to data/input/ before training")
        print("   - Copy your food photos to data/input/")
        print("   - Supported formats: jpg, jpeg, png, bmp, tiff")
        print("   - At least 10-20 images recommended for meaningful training")
        
        # Create a simple README
        readme_path = input_dir / "README.txt"
        try:
            with open(readme_path, 'w', encoding='utf-8') as f:
                f.write("""FOOD IMAGES DIRECTORY
====================

This directory should contain your food images for training.

What to put here:
- Food photos (any format: jpg, png, etc.)
- At least 10-20 images for testing  
- 100+ images for good training results
- Diverse foods and angles work best

Examples of good training images:
- Pizza slices on plates
- Sandwiches cut in half
- Salads in bowls
- Various foods from different angles
- Different lighting conditions

After adding images, run:
python scripts/train_custom_food_model.py --mode check_setup
""")
            fixes_applied.append("Created README in data/input")
        except Exception as e:
            print(f"[WARNING] Could not create README: {e}")
    else:
        print(f"[FOUND] {len(existing_images)} images in {input_dir}")
    
    # Fix 4: Update requirements.txt if needed
    requirements_path = Path("requirements.txt")
    training_requirements = [
        "ultralytics>=8.0.0",
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "opencv-python>=4.8.0",
        "scikit-learn>=1.3.0",
        "tqdm>=4.65.0",
        "seaborn>=0.12.0"
    ]
    
    if requirements_path.exists():
        try:
            with open(requirements_path, 'r', encoding='utf-8') as f:
                existing_reqs = f.read()
        except UnicodeDecodeError:
            # Fallback for different encoding
            try:
                with open(requirements_path, 'r', encoding='cp1252') as f:
                    existing_reqs = f.read()
            except Exception as e:
                print(f"[WARNING] Could not read requirements.txt: {e}")
                existing_reqs = ""
        
        missing_reqs = []
        for req in training_requirements:
            package_name = req.split(">=")[0]
            if package_name not in existing_reqs:
                missing_reqs.append(req)
        
        if missing_reqs:
            print(f"\nAdding missing requirements...")
            try:
                with open(requirements_path, 'a', encoding='utf-8') as f:
                    f.write("\n# Training dependencies\n")
                    for req in missing_reqs:
                        f.write(f"{req}\n")
                        print(f"[ADDED] {req}")
                fixes_applied.append("Updated requirements.txt")
            except Exception as e:
                print(f"[WARNING] Could not update requirements.txt: {e}")
    
    # Summary
    print("\n" + "=" * 50)
    print("TROUBLESHOOTING COMPLETE")
    
    if fixes_applied:
        print("[SUCCESS] Applied fixes:")
        for fix in fixes_applied:
            print(f"   - {fix}")
    else:
        print("[SUCCESS] No issues found - everything looks good!")
    
    # Next steps
    print("\nNEXT STEPS:")
    if len(existing_images) == 0:
        print("1. Add food images to data/input/")
        print("2. Install requirements: pip install -r requirements.txt")
        print("3. Check setup: python scripts/train_custom_food_model.py --mode check_setup")
    else:
        print("1. Install requirements: pip install -r requirements.txt")
        print("2. Check setup: python scripts/train_custom_food_model.py --mode check_setup")
        print("3. Start training: python scripts/train_custom_food_model.py --mode quick_test")
    
    return len(fixes_applied) > 0

if __name__ == "__main__":
    print("Running training setup troubleshooter (Windows Compatible)...")
    
    try:
        fixes_needed = fix_training_setup()
        
        print("\nReady to proceed with training setup!")
        
        if fixes_needed:
            print("Fixes were applied. Please follow the next steps above.")
        else:
            print("Everything looks good. You can proceed with training!")
            
    except Exception as e:
        print(f"\n[ERROR] Troubleshooter failed: {e}")
        print("Please check the error and try again.")