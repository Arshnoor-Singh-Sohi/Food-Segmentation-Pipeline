#!/usr/bin/env python3
"""
Quick Device Fix - Forces CPU training when no GPU available
"""

import yaml
from pathlib import Path

def fix_device_config():
    """Fix device configuration in config file"""
    config_path = Path("config/training_config.yaml")
    
    print("FIXING DEVICE CONFIGURATION")
    print("=" * 40)
    
    if not config_path.exists():
        print("[ERROR] Config file not found!")
        return False
    
    try:
        # Read existing config
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Fix device setting
        if 'training' in config:
            old_device = config['training'].get('device', 'unknown')
            config['training']['device'] = 'cpu'  # Force CPU for now
            print(f"[FIXED] Changed device: {old_device} -> cpu")
        
        # Write back the fixed config
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        print(f"[SUCCESS] Updated config file: {config_path}")
        return True
        
    except Exception as e:
        print(f"[ERROR] Could not fix config: {e}")
        return False

def create_direct_segmentation_script():
    """Create a script that bypasses all config issues"""
    script_content = '''#!/usr/bin/env python3
"""
Direct Segmentation Training - No Config Dependencies
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from ultralytics import YOLO

def train_direct():
    """Train segmentation directly with hardcoded parameters"""
    
    print("DIRECT SEGMENTATION TRAINING")
    print("Using hardcoded CPU parameters to avoid config issues")
    print("=" * 50)
    
    # Check dataset
    dataset_path = Path("data/training/food_training/existing_images_dataset.yaml")
    if not dataset_path.exists():
        print(f"ERROR: Dataset not found at {dataset_path}")
        print("Please run dataset preparation first!")
        return
    
    print(f"Dataset: {dataset_path}")
    
    # Load model
    model = YOLO('yolov8n-seg.pt')
    print("Loaded YOLOv8 segmentation model")
    
    # Direct training with explicit CPU parameters
    results = model.train(
        data=str(dataset_path),
        epochs=25,
        batch=4,
        device='cpu',  # Explicit CPU - no ambiguity
        project='direct_segmentation',
        name='food_seg_cpu',
        patience=10,
        workers=1,
        task='segment',
        lr0=0.001,
        optimizer='AdamW',
        verbose=True
    )
    
    print("Training completed!")
    return results

if __name__ == "__main__":
    train_direct()
'''
    
    script_path = Path("train_segmentation_direct.py")
    try:
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(script_content)
        print(f"[CREATED] Direct training script: {script_path}")
        return True
    except Exception as e:
        print(f"[ERROR] Could not create script: {e}")
        return False

def main():
    """Apply device fixes"""
    print("DEVICE CONFIGURATION FIX")
    print("=" * 50)
    
    # Fix 1: Update config file
    config_fixed = fix_device_config()
    
    # Fix 2: Create direct training script
    script_created = create_direct_segmentation_script()
    
    print("\n" + "=" * 50)
    print("DEVICE FIX SUMMARY")
    print("=" * 50)
    
    if config_fixed:
        print("[SUCCESS] Config file fixed to use CPU")
        print("\nTry the main script again:")
        print("python scripts/train_custom_food_model.py --mode segmentation --epochs 25")
    
    if script_created:
        print("[BACKUP] Created direct training script")
        print("\nOr use the guaranteed-to-work script:")
        print("python train_segmentation_direct.py")
    
    print("\nBoth approaches will train your segmentation model on CPU!")

if __name__ == "__main__":
    main()