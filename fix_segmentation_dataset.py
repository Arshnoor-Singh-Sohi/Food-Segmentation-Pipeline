#!/usr/bin/env python3
"""
Fix Segmentation Dataset - Convert Detection Labels to Segmentation Format
The issue is that we created detection labels (bounding boxes) but segmentation training needs polygon labels
"""

import os
import shutil
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm

def convert_detection_to_segmentation_labels():
    """
    Convert existing detection labels to segmentation format
    Detection format: class_id center_x center_y width height
    Segmentation format: class_id x1 y1 x2 y2 x3 y3 x4 y4 (polygon points)
    """
    
    print("CONVERTING DETECTION LABELS TO SEGMENTATION FORMAT")
    print("=" * 60)
    
    # Paths to label directories
    train_labels_dir = Path("data/training/food_training/labels/train")
    val_labels_dir = Path("data/training/food_training/labels/val")
    
    if not train_labels_dir.exists() or not val_labels_dir.exists():
        print("[ERROR] Label directories not found!")
        print("Please run dataset preparation first.")
        return False
    
    # Convert train labels
    print("Converting training labels...")
    train_converted = convert_labels_in_directory(train_labels_dir)
    
    # Convert validation labels  
    print("Converting validation labels...")
    val_converted = convert_labels_in_directory(val_labels_dir)
    
    if train_converted and val_converted:
        print("[SUCCESS] All labels converted to segmentation format!")
        return True
    else:
        print("[ERROR] Label conversion failed!")
        return False

def convert_labels_in_directory(labels_dir):
    """Convert all label files in a directory from detection to segmentation format"""
    
    label_files = list(labels_dir.glob("*.txt"))
    if not label_files:
        print(f"[WARNING] No label files found in {labels_dir}")
        return True
    
    converted_count = 0
    
    for label_file in tqdm(label_files, desc=f"Converting {labels_dir.name}"):
        try:
            # Read existing detection label
            with open(label_file, 'r') as f:
                lines = f.readlines()
            
            # Convert each line to segmentation format
            new_lines = []
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Parse detection format: class_id center_x center_y width height
                parts = line.split()
                if len(parts) != 5:
                    continue
                
                class_id = parts[0]
                center_x = float(parts[1])
                center_y = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                
                # Convert to rectangle polygon (4 corners)
                # Calculate corner coordinates
                x1 = center_x - width/2
                y1 = center_y - height/2
                x2 = center_x + width/2
                y2 = center_y - height/2
                x3 = center_x + width/2
                y3 = center_y + height/2
                x4 = center_x - width/2
                y4 = center_y + height/2
                
                # Ensure coordinates are within bounds [0, 1]
                coords = [x1, y1, x2, y2, x3, y3, x4, y4]
                coords = [max(0, min(1, coord)) for coord in coords]
                
                # Format as segmentation label: class_id x1 y1 x2 y2 x3 y3 x4 y4
                seg_line = f"{class_id} " + " ".join([f"{coord:.6f}" for coord in coords])
                new_lines.append(seg_line)
            
            # Write back the converted labels
            with open(label_file, 'w') as f:
                f.write('\n'.join(new_lines) + '\n')
            
            converted_count += 1
            
        except Exception as e:
            print(f"[ERROR] Could not convert {label_file}: {e}")
    
    print(f"[SUCCESS] Converted {converted_count}/{len(label_files)} label files in {labels_dir.name}")
    return converted_count > 0

def create_segmentation_dataset_yaml():
    """Create a proper segmentation dataset YAML file"""
    
    dataset_yaml_content = {
        'path': str(Path("data/training/food_training").absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'nc': 1,  # Number of classes
        'names': ['food'],  # Class names
        'task': 'segment'  # Specify this is for segmentation
    }
    
    yaml_path = Path("data/training/food_training/food_segmentation_dataset.yaml")
    
    try:
        import yaml
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(dataset_yaml_content, f, default_flow_style=False)
        
        print(f"[CREATED] Segmentation dataset YAML: {yaml_path}")
        return yaml_path
        
    except Exception as e:
        print(f"[ERROR] Could not create segmentation YAML: {e}")
        return None

def validate_segmentation_dataset():
    """Validate that the segmentation dataset is properly formatted"""
    
    print("\nValidating segmentation dataset...")
    
    # Check label format
    train_labels_dir = Path("data/training/food_training/labels/train")
    sample_labels = list(train_labels_dir.glob("*.txt"))[:3]  # Check first 3 files
    
    valid_format = True
    
    for label_file in sample_labels:
        try:
            with open(label_file, 'r') as f:
                lines = f.readlines()
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                # Segmentation format should have: class_id + 8 coordinates (4 points x 2 coords each)
                if len(parts) != 9:  # 1 class_id + 8 coordinates
                    print(f"[ERROR] Invalid segmentation format in {label_file}: {line}")
                    valid_format = False
                    break
            
            if not valid_format:
                break
                
        except Exception as e:
            print(f"[ERROR] Could not validate {label_file}: {e}")
            valid_format = False
    
    if valid_format:
        print("[SUCCESS] Segmentation dataset format is valid!")
    else:
        print("[ERROR] Segmentation dataset format is invalid!")
    
    return valid_format

def create_segmentation_training_script():
    """Create a script specifically for segmentation training with the fixed dataset"""
    
    script_content = '''#!/usr/bin/env python3
"""
Segmentation Training with Fixed Dataset
Uses the properly formatted segmentation labels
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from ultralytics import YOLO

def train_segmentation_fixed():
    """Train segmentation with properly formatted dataset"""
    
    print("SEGMENTATION TRAINING - FIXED DATASET")
    print("Using properly formatted segmentation labels")
    print("=" * 50)
    
    # Use the segmentation-specific dataset YAML
    dataset_path = Path("data/training/food_training/food_segmentation_dataset.yaml")
    if not dataset_path.exists():
        print(f"ERROR: Segmentation dataset not found at {dataset_path}")
        print("Please run fix_segmentation_dataset.py first!")
        return
    
    print(f"Dataset: {dataset_path}")
    
    # Load segmentation model
    model = YOLO('yolov8n-seg.pt')
    print("Loaded YOLOv8 segmentation model")
    
    # Training with segmentation dataset
    results = model.train(
        data=str(dataset_path),
        epochs=25,
        batch=4,
        device='cpu',
        project='segmentation_training_fixed',
        name='food_seg_fixed',
        patience=10,
        workers=1,
        task='segment',
        lr0=0.001,
        optimizer='AdamW',
        verbose=True,
        # Segmentation-specific parameters
        mask_ratio=4,
        overlap_mask=True
    )
    
    print("Segmentation training completed!")
    print("Model saved in: segmentation_training_fixed/food_seg_fixed/weights/best.pt")
    return results

if __name__ == "__main__":
    train_segmentation_fixed()
'''
    
    script_path = Path("train_segmentation_fixed.py")
    try:
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(script_content)
        print(f"[CREATED] Fixed segmentation training script: {script_path}")
        return True
    except Exception as e:
        print(f"[ERROR] Could not create training script: {e}")
        return False

def main():
    """Main function to fix segmentation dataset"""
    
    print("SEGMENTATION DATASET FIX")
    print("=" * 60)
    print("Converting detection labels to segmentation format...")
    
    # Step 1: Convert labels from detection to segmentation format
    labels_converted = convert_detection_to_segmentation_labels()
    
    if not labels_converted:
        print("[ERROR] Could not convert labels!")
        return
    
    # Step 2: Create segmentation-specific dataset YAML
    yaml_created = create_segmentation_dataset_yaml()
    
    if not yaml_created:
        print("[ERROR] Could not create segmentation dataset YAML!")
        return
    
    # Step 3: Validate the converted dataset
    dataset_valid = validate_segmentation_dataset()
    
    # Step 4: Create fixed training script
    script_created = create_segmentation_training_script()
    
    # Summary
    print("\n" + "=" * 60)
    print("SEGMENTATION FIX SUMMARY")
    print("=" * 60)
    
    if labels_converted and yaml_created and dataset_valid and script_created:
        print("[SUCCESS] Segmentation dataset fixed!")
        print("\nWhat was fixed:")
        print("  - Converted detection labels to segmentation polygon format")
        print("  - Created segmentation-specific dataset YAML")
        print("  - Validated dataset format")
        print("  - Created fixed training script")
        
        print("\nNow you can train segmentation:")
        print("python train_segmentation_fixed.py")
        
    else:
        print("[ERROR] Some fixes failed. Check the output above.")

if __name__ == "__main__":
    main()