#!/usr/bin/env python3
"""
Definitive Fix for batch_size vs batch Issue
This will fix the YOLO parameter issue once and for all
"""

import os
import yaml
from pathlib import Path

def fix_config_file():
    """Fix the training config file to use correct YOLO parameters"""
    config_path = Path("config/training_config.yaml")
    
    print("FIXING BATCH_SIZE ISSUE")
    print("=" * 40)
    
    # Create the correct config with proper YOLO parameter names
    correct_config = {
        'model': {
            'base_model': 'yolov8n.pt',
            'input_size': 640
        },
        'training': {
            'epochs': 50,
            'batch': 8,  # CORRECT: Use 'batch' not 'batch_size'
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
    
    # Backup existing config if it exists
    if config_path.exists():
        backup_path = config_path.with_suffix('.yaml.backup')
        try:
            import shutil
            shutil.copy2(config_path, backup_path)
            print(f"[BACKUP] Saved existing config to {backup_path}")
        except Exception as e:
            print(f"[WARNING] Could not backup config: {e}")
    
    # Write the correct config
    try:
        config_path.parent.mkdir(exist_ok=True)
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(correct_config, f, default_flow_style=False)
        
        print(f"[FIXED] Created corrected config: {config_path}")
        print("[SUCCESS] batch_size -> batch parameter fixed!")
        return True
        
    except Exception as e:
        print(f"[ERROR] Could not write config: {e}")
        return False

def verify_config():
    """Verify the config file has correct parameters"""
    config_path = Path("config/training_config.yaml")
    
    if not config_path.exists():
        print("[ERROR] Config file not found!")
        return False
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Check for the problematic parameter
        training_section = config.get('training', {})
        
        if 'batch_size' in training_section:
            print("[ERROR] Found 'batch_size' in config - this will cause errors!")
            return False
        elif 'batch' in training_section:
            print(f"[SUCCESS] Found correct 'batch' parameter: {training_section['batch']}")
            return True
        else:
            print("[WARNING] No batch parameter found in config")
            return False
            
    except Exception as e:
        print(f"[ERROR] Could not read config: {e}")
        return False

def create_minimal_training_script():
    """Create a minimal training script that bypasses config issues"""
    script_content = '''#!/usr/bin/env python3
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
'''
    
    script_path = Path("train_segmentation_minimal.py")
    try:
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(script_content)
        print(f"[CREATED] Minimal training script: {script_path}")
        return True
    except Exception as e:
        print(f"[ERROR] Could not create script: {e}")
        return False

def main():
    """Main fix function"""
    print("DEFINITIVE BATCH_SIZE FIX")
    print("=" * 50)
    
    # Step 1: Fix config file
    print("\nStep 1: Fixing config file...")
    config_fixed = fix_config_file()
    
    # Step 2: Verify config
    print("\nStep 2: Verifying config...")
    config_valid = verify_config()
    
    # Step 3: Create minimal training script as backup
    print("\nStep 3: Creating minimal training script...")
    script_created = create_minimal_training_script()
    
    # Summary
    print("\n" + "=" * 50)
    print("FIX SUMMARY")
    print("=" * 50)
    
    if config_fixed and config_valid:
        print("[SUCCESS] Config file fixed and verified!")
        print("\nNow try running:")
        print("python scripts/train_custom_food_model.py --mode segmentation --epochs 25")
        
    elif script_created:
        print("[BACKUP] Created minimal training script!")
        print("\nIf main script still fails, try:")
        print("python train_segmentation_minimal.py")
    
    else:
        print("[ERROR] Could not fix the issue")
        print("Please check file permissions and try again")

if __name__ == "__main__":
    main()