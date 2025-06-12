#!/usr/bin/env python3
"""
Training Issues Troubleshooter
Run this script to automatically fix common training setup problems
"""

import os
import shutil
from pathlib import Path
import yaml

def fix_training_setup():
    """Fix common training setup issues automatically"""
    
    print("🔧 TRAINING SETUP TROUBLESHOOTER")
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
    
    print("\n📁 Creating missing directories...")
    for dir_path in required_dirs:
        path = Path(dir_path)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            print(f"✅ Created: {dir_path}")
            fixes_applied.append(f"Created directory: {dir_path}")
        
        # Create __init__.py for Python packages
        if dir_path.startswith("src/"):
            init_file = path / "__init__.py"
            if not init_file.exists():
                init_file.touch()
                print(f"✅ Created: {init_file}")
    
    # Fix 2: Create training configuration if missing
    config_dir = Path("config")
    training_config_path = config_dir / "training_config.yaml"
    
    if not training_config_path.exists():
        print("\n⚙️  Creating training configuration...")
        
        training_config = {
            'model': {
                'base_model': 'yolov8n.pt',
                'input_size': 640
            },
            'training': {
                'epochs': 50,
                'batch': 8,  # FIXED: Use 'batch' not 'batch_size' for YOLO compatibility
                'patience': 20,
                'device': 'auto',
                'workers': 2,  # Conservative
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
        
        with open(training_config_path, 'w') as f:
            yaml.dump(training_config, f, default_flow_style=False)
        
        print(f"✅ Created: {training_config_path}")
        fixes_applied.append("Created training configuration")
    
    # Fix 3: Create sample images if data/input is empty
    input_dir = Path("data/input")
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    existing_images = []
    
    for ext in image_extensions:
        existing_images.extend(list(input_dir.glob(f"*{ext}")))
        existing_images.extend(list(input_dir.glob(f"*{ext.upper()}")))
    
    if len(existing_images) == 0:
        print(f"\n📸 No images found in {input_dir}")
        print("💡 IMPORTANT: You need to add food images to data/input/ before training")
        print("   - Copy your food photos to data/input/")
        print("   - Supported formats: jpg, jpeg, png, bmp, tiff")
        print("   - At least 10-20 images recommended for meaningful training")
        
        # Create a README in the input directory
        readme_path = input_dir / "README.txt"
        with open(readme_path, 'w') as f:
            f.write("""
FOOD IMAGES DIRECTORY
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
        
        fixes_applied.append("Created README in data/input (you need to add images)")
    else:
        print(f"✅ Found {len(existing_images)} images in {input_dir}")
    
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
        with open(requirements_path, 'r') as f:
            existing_reqs = f.read()
        
        missing_reqs = []
        for req in training_requirements:
            package_name = req.split(">=")[0]
            if package_name not in existing_reqs:
                missing_reqs.append(req)
        
        if missing_reqs:
            print(f"\n📦 Adding missing requirements...")
            with open(requirements_path, 'a') as f:
                f.write("\n# Training dependencies\n")
                for req in missing_reqs:
                    f.write(f"{req}\n")
                    print(f"✅ Added: {req}")
            fixes_applied.append("Updated requirements.txt")
    
    # Summary
    print("\n" + "=" * 50)
    print("🏁 TROUBLESHOOTING COMPLETE")
    
    if fixes_applied:
        print("✅ Applied fixes:")
        for fix in fixes_applied:
            print(f"   - {fix}")
    else:
        print("✅ No issues found - everything looks good!")
    
    # Next steps
    print("\n💡 NEXT STEPS:")
    if len(existing_images) == 0:
        print("1. 🖼️  Add food images to data/input/")
        print("2. 📦 Install requirements: pip install -r requirements.txt")
        print("3. 🔍 Check setup: python scripts/train_custom_food_model.py --mode check_setup")
    else:
        print("1. 📦 Install requirements: pip install -r requirements.txt")
        print("2. 🔍 Check setup: python scripts/train_custom_food_model.py --mode check_setup")
        print("3. 🚀 Start training: python scripts/train_custom_food_model.py --mode quick_test")
    
    return len(fixes_applied) > 0

def create_sample_images_readme():
    """Create helpful documentation about images"""
    
    readme_content = """
# Food Training Images Guide

## Quick Start
1. Add 10-20 food images to this directory (data/input/)
2. Run: python scripts/train_custom_food_model.py --mode check_setup
3. If setup check passes, run: python scripts/train_custom_food_model.py --mode quick_test

## Image Requirements
- **Format**: JPG, PNG, BMP, or TIFF
- **Size**: Any size (will be automatically resized)
- **Minimum**: 10 images for testing, 50+ for good results
- **Content**: Clear photos of food items

## Good Training Images
✅ Food clearly visible
✅ Good lighting
✅ Different angles and portions
✅ Various food types
✅ Some foods on plates/bowls

## Avoid
❌ Very blurry images
❌ Extremely dark photos
❌ Images with no food visible
❌ Duplicate identical photos

## Examples of Image Names
- pizza_slice_1.jpg
- burger_and_fries.png
- salad_bowl.jpg
- pasta_dinner.jpg
- sandwich_lunch.png

The training system will automatically create labels for your images,
so you don't need to manually annotate anything!
"""
    
    readme_path = Path("data/input/TRAINING_IMAGES_GUIDE.md")
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    
    print(f"📖 Created image guide: {readme_path}")

if __name__ == "__main__":
    print("🔧 Running training setup troubleshooter...")
    
    fixes_needed = fix_training_setup()
    create_sample_images_readme()
    
    print("\n🎯 Ready to proceed with training setup!")
    
    if fixes_needed:
        print("💡 Fixes were applied. Please follow the next steps above.")
    else:
        print("💡 Everything looks good. You can proceed with training!")