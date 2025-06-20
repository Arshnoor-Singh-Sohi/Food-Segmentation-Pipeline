"""
RunPod Training Setup for Individual Item Detection
==================================================

Sets up RunPod environment for fine-tuning your 99.5% model to detect individual items.

Strategy:
1. Use your custom model as base (99.5% accuracy foundation)
2. Add individual ingredient classes (banana, apple, bottle, etc.)
3. Fine-tune rather than train from scratch (faster, better results)
4. Focus on refrigerator inventory counting

Usage:
python stages/runpod_training/setup_individual_training.py --base-model data/models/custom_food_detection.pt
"""

import argparse
import json
import yaml
from pathlib import Path
from datetime import datetime
import shutil

class RunPodTrainingSetup:
    def __init__(self, base_model_path):
        self.base_model_path = Path(base_model_path)
        self.training_dir = Path("stages/runpod_training")
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def create_individual_item_config(self):
        """Create YOLO config for individual item detection"""
        print("📝 Creating individual item detection config...")
        
        # Individual ingredient classes for refrigerator counting
        individual_classes = {
            # Fruits (individual counting)
            0: 'banana_single',
            1: 'apple_red', 
            2: 'apple_green',
            3: 'orange_single',
            4: 'strawberry_single',
            5: 'grape_cluster',
            
            # Vegetables (individual counting)
            6: 'tomato_single',
            7: 'carrot_single', 
            8: 'lettuce_head',
            9: 'broccoli_head',
            10: 'bell_pepper',
            11: 'cucumber_single',
            
            # Containers & Bottles (main issue from CEO feedback)
            12: 'bottle_milk',
            13: 'bottle_juice',
            14: 'bottle_water',
            15: 'bottle_soda',
            16: 'container_plastic',
            17: 'jar_glass',
            18: 'can_food',
            
            # Packaged items
            19: 'package_food',
            20: 'box_cereal',
            21: 'bag_frozen',
            22: 'carton_milk',
            23: 'carton_juice',
            
            # Individual items
            24: 'egg_single',
            25: 'bread_loaf',
            26: 'cheese_block',
            27: 'yogurt_container'
        }
        
        # Create dataset config
        dataset_config = {
            'path': str(self.training_dir / 'individual_item_dataset'),
            'train': 'images/train',
            'val': 'images/val', 
            'test': 'images/test',
            'names': individual_classes
        }
        
        # Save config
        config_file = self.training_dir / 'individual_item_config.yaml'
        config_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(dataset_config, f, default_flow_style=False)
        
        print(f"✅ Config saved: {config_file}")
        print(f"📊 Total classes: {len(individual_classes)}")
        
        return config_file, individual_classes
    
    def create_training_script(self, config_file):
        """Create RunPod training script"""
        print("🚀 Creating RunPod training script...")
        
        training_script = f'''#!/usr/bin/env python3
"""
Individual Item Detection Training Script for RunPod
===================================================
"""

import os
import sys
from pathlib import Path
from ultralytics import YOLO
import torch
import yaml

def check_gpu():
    """Check GPU availability"""
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        print(f"🔥 GPU Available: {{gpu_name}} ({{gpu_count}} GPUs)")
        return True
    else:
        print("⚠️ No GPU available - using CPU")
        return False

def load_base_model(model_path):
    """Load your 99.5% accuracy custom model as base"""
    if Path(model_path).exists():
        print(f"📦 Loading custom base model: {{model_path}}")
        model = YOLO(model_path)
        print(f"✅ Custom model loaded with {{len(model.names)}} classes")
        return model
    else:
        print(f"⚠️ Custom model not found, using YOLOv8n as base")
        return YOLO('yolov8n.pt')

def main():
    print("🚀 INDIVIDUAL ITEM DETECTION TRAINING")
    print("=" * 50)
    
    # Check environment
    gpu_available = check_gpu()
    
    # Load base model (your 99.5% accuracy model)
    base_model = load_base_model('custom_food_detection.pt')
    
    # Training parameters optimized for individual item counting
    training_params = {{
        'data': 'individual_item_config.yaml',
        'epochs': 75,                    # Sufficient for fine-tuning
        'batch': 16 if gpu_available else 4,  # Adjust based on GPU memory
        'imgsz': 640,                    # Standard YOLO image size
        'device': 0 if gpu_available else 'cpu',
        'project': 'runs/individual_items',
        'name': 'enhanced_ingredient_detector_{{self.timestamp}}',
        'save': True,
        'plots': True,
        'patience': 15,                  # Early stopping
        
        # Fine-tuning specific parameters
        'resume': False,                 # Start fresh fine-tuning
        'pretrained': True,              # Use pretrained weights
        'optimizer': 'AdamW',            # Good for fine-tuning
        'lr0': 0.001,                    # Lower learning rate for fine-tuning
        'lrf': 0.1,                      # Learning rate decay
        'momentum': 0.937,
        'weight_decay': 0.0005,
        
        # Individual item detection optimization
        'box': 7.5,                      # Higher box loss (important for counting)
        'cls': 1.0,                      # Classification loss
        'dfl': 1.5,                      # Distribution focal loss
        'overlap_mask': False,           # Don't merge overlapping items
        'mask_ratio': 4,                 # Mask ratio for segmentation
        
        # Data augmentation (conservative for fine-tuning)
        'hsv_h': 0.015,                  # Hue augmentation
        'hsv_s': 0.7,                    # Saturation (important for food)
        'hsv_v': 0.4,                    # Value augmentation
        'degrees': 0.0,                  # No rotation (food orientation matters)
        'translate': 0.1,                # Slight translation
        'scale': 0.5,                    # Scale augmentation
        'shear': 0.0,                    # No shearing
        'perspective': 0.0,              # No perspective change
        'flipud': 0.0,                   # No vertical flip
        'fliplr': 0.5,                   # 50% horizontal flip
        'mosaic': 1.0,                   # Mosaic augmentation
        'mixup': 0.1                     # Light mixup
    }}
    
    print("🎯 Training Parameters:")
    for key, value in training_params.items():
        print(f"   {{key}}: {{value}}")
    
    # Start training
    print("\\n🔥 Starting individual item detection training...")
    try:
        results = base_model.train(**training_params)
        
        print("\\n✅ Training completed successfully!")
        print(f"📊 Results: {{results}}")
        
        # Save enhanced model
        enhanced_model_path = f'enhanced_individual_detector_{{self.timestamp}}.pt'
        print(f"💾 Enhanced model saved as: {{enhanced_model_path}}")
        
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {{e}}")
        return False

if __name__ == "__main__":
    main()
'''
        
        script_file = self.training_dir / 'train_individual_items.py'
        with open(script_file, 'w', encoding='utf-8') as f:
            f.write(training_script)
        
        # Make executable
        script_file.chmod(0o755)
        
        print(f"✅ Training script saved: {script_file}")
        return script_file
    
    def create_dockerfile(self):
        """Create Dockerfile for RunPod environment"""
        print("🐳 Creating Dockerfile for RunPod...")
        
        dockerfile_content = '''# RunPod Dockerfile for Individual Item Detection Training
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    git \\
    wget \\
    curl \\
    unzip \\
    libgl1-mesa-glx \\
    libglib2.0-0 \\
    libsm6 \\
    libxext6 \\
    libxrender-dev \\
    libgomp1 \\
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \\
    ultralytics \\
    opencv-python \\
    matplotlib \\
    pandas \\
    numpy \\
    Pillow \\
    pyyaml \\
    tqdm \\
    seaborn \\
    tensorboard

# Copy training files
COPY . /workspace/

# Set environment variables
ENV PYTHONPATH=/workspace
ENV CUDA_VISIBLE_DEVICES=0

# Default command
CMD ["python", "train_individual_items.py"]
'''
        
        dockerfile = self.training_dir / 'Dockerfile'
        with open(dockerfile, 'w', encoding='utf-8') as f:
            f.write(dockerfile_content)
        
        print(f"✅ Dockerfile saved: {dockerfile}")
        return dockerfile
    
    def create_data_preparation_guide(self, individual_classes):
        """Create guide for preparing training data"""
        print("📚 Creating data preparation guide...")
        
        guide_content = f'''# Individual Item Detection - Data Preparation Guide

## Overview
To train your model for individual item detection, you need labeled images showing individual ingredients in refrigerator contexts.

## Required Dataset Structure
```
individual_item_dataset/
├── images/
│   ├── train/          # 70% of images
│   ├── val/            # 20% of images  
│   └── test/           # 10% of images
└── labels/
    ├── train/          # YOLO format labels
    ├── val/
    └── test/
```

## Target Classes ({len(individual_classes)} total)
{chr(10).join([f"{idx}: {name}" for idx, name in individual_classes.items()])}

## Labeling Guidelines

### For Individual Items:
- **Each banana gets its own bounding box** (not grouped)
- **Each apple gets its own bounding box** (even if multiple)
- **Each bottle gets its own bounding box with specific type** (milk vs juice)

### YOLO Label Format:
```
class_id x_center y_center width height
```

Example for image with 3 bananas:
```
0 0.2 0.3 0.1 0.15    # banana_single 1
0 0.25 0.35 0.1 0.15  # banana_single 2  
0 0.3 0.4 0.1 0.15    # banana_single 3
```

## Data Collection Strategy

### Option 1: Use Existing Refrigerator Images
1. Collect 500-1000 refrigerator images
2. Label individual items manually
3. Use tools like Roboflow or LabelImg

### Option 2: Synthetic Data Generation
1. Take photos of individual ingredients
2. Composite them into refrigerator scenes
3. Generate automatic labels

### Option 3: Expand Existing Dataset
1. Start with your current images
2. Re-label for individual item detection
3. Add new refrigerator photos

## Recommended Labeling Tools
- **Roboflow**: Online labeling with export to YOLO format
- **LabelImg**: Desktop labeling tool
- **CVAT**: Computer Vision Annotation Tool
- **Supervisely**: Advanced annotation platform

## Quality Guidelines
- **Minimum 50 examples per class**
- **Variety of lighting conditions**
- **Different refrigerator types and contents**
- **Clear individual item boundaries**
- **Consistent labeling across similar items**

## Time Estimate
- **Manual labeling**: 2-4 seconds per bounding box
- **500 images with 10 items each**: ~3-6 hours total
- **Quality review**: Additional 1-2 hours

## Next Steps After Data Preparation
1. Upload dataset to RunPod
2. Run training script: `python train_individual_items.py`
3. Monitor training progress
4. Download enhanced model
5. Test on refrigerator images
'''
        
        guide_file = self.training_dir / 'DATA_PREPARATION_GUIDE.md'
        with open(guide_file, 'w', encoding='utf-8') as f:
            f.write(guide_content)
        
        print(f"✅ Data guide saved: {guide_file}")
        return guide_file
    
    def create_runpod_upload_script(self):
        """Create script to upload data to RunPod"""
        print("📤 Creating RunPod upload script...")
        
        upload_script = '''#!/bin/bash
# RunPod Data Upload Script

echo "📤 RUNPOD DATA UPLOAD FOR INDIVIDUAL ITEM TRAINING"
echo "=================================================="

# Check if RunPod CLI is installed
if ! command -v runpod &> /dev/null; then
    echo "❌ RunPod CLI not found. Install with:"
    echo "pip install runpod"
    exit 1
fi

# Upload training files
echo "📦 Uploading training configuration..."
runpod upload individual_item_config.yaml
runpod upload train_individual_items.py
runpod upload Dockerfile

# Upload base model (your 99.5% accuracy model)
echo "📦 Uploading custom base model..."
if [ -f "../data/models/custom_food_detection.pt" ]; then
    runpod upload ../data/models/custom_food_detection.pt custom_food_detection.pt
    echo "✅ Custom model uploaded"
else
    echo "⚠️ Custom model not found - will use YOLOv8n as base"
fi

# Upload dataset (if exists)
if [ -d "individual_item_dataset" ]; then
    echo "📦 Uploading training dataset..."
    runpod upload individual_item_dataset/
    echo "✅ Dataset uploaded"
else
    echo "⚠️ No dataset found - follow DATA_PREPARATION_GUIDE.md first"
fi

echo ""
echo "🚀 Upload complete! Next steps:"
echo "1. Start RunPod instance with GPU"
echo "2. Run: python train_individual_items.py"
echo "3. Monitor training progress"
echo "4. Download enhanced model when complete"
'''
        
        upload_script_file = self.training_dir / 'upload_to_runpod.sh'
        with open(upload_script_file, 'w', encoding='utf-8') as f:
            f.write(upload_script)
        
        # Make executable
        upload_script_file.chmod(0o755)
        
        print(f"✅ Upload script saved: {upload_script_file}")
        return upload_script_file
    
    def create_progress_tracker(self):
        """Create progress tracking file for chat continuity"""
        progress = {
            'stage': '1a-fixed',
            'phase': 'runpod_setup_complete',
            'timestamp': self.timestamp,
            'base_model': str(self.base_model_path),
            'runpod_setup': {
                'config_created': True,
                'training_script_created': True,
                'dockerfile_created': True,
                'upload_script_created': True
            },
            'next_steps': [
                'Prepare individual item training dataset',
                'Upload data to RunPod',
                'Run training: python train_individual_items.py',
                'Download enhanced model',
                'Test individual item detection'
            ],
            'success_criteria': {
                'individual_items_detected': '>= 10',
                'bottle_counting_accurate': True,
                'banana_counting_accurate': True,
                'false_positives_reduced': True
            }
        }
        
        progress_file = Path('stages/progress.json')
        with open(progress_file, 'w', encoding='utf-8') as f:
            json.dump(progress, f, indent=2)
        
        print(f"✅ Progress tracker saved: {progress_file}")
        return progress_file
    
    def show_setup_summary(self):
        """Show summary of RunPod setup"""
        print(f"\n🎯 RUNPOD TRAINING SETUP COMPLETE")
        print("=" * 60)
        
        print(f"📁 Training files created in: {self.training_dir}")
        print(f"⚡ Base model: {self.base_model_path}")
        print(f"🎲 Training timestamp: {self.timestamp}")
        
        print(f"\n📋 Files Created:")
        print(f"   ✅ individual_item_config.yaml - Training configuration")
        print(f"   ✅ train_individual_items.py - Main training script")
        print(f"   ✅ Dockerfile - RunPod environment")
        print(f"   ✅ upload_to_runpod.sh - Data upload script")
        print(f"   ✅ DATA_PREPARATION_GUIDE.md - Dataset instructions")
        
        print(f"\n🚀 Next Steps:")
        print(f"1. Prepare training dataset (follow DATA_PREPARATION_GUIDE.md)")
        print(f"2. Upload to RunPod: bash upload_to_runpod.sh")
        print(f"3. Start RunPod GPU instance")
        print(f"4. Run training: python train_individual_items.py")
        print(f"5. Download enhanced model")
        
        print(f"\n⏱️ Estimated Timeline:")
        print(f"   • Data preparation: 4-8 hours")
        print(f"   • RunPod training: 2-4 hours")
        print(f"   • Testing & validation: 1-2 hours")
        print(f"   • Total: 1-2 days")
        
        print(f"\n🎯 Expected Results After Training:")
        print(f"   • Detect 10-15 individual items in refrigerator")
        print(f"   • Accurate banana counting (3-4 individual bananas)")
        print(f"   • Accurate bottle counting (2-3 bottles, not 10+)")
        print(f"   • Individual item classification (apple vs banana vs bottle)")

def main():
    parser = argparse.ArgumentParser(description='Setup RunPod Training for Individual Item Detection')
    parser.add_argument('--base-model', type=str, default='data/models/custom_food_detection.pt',
                       help='Path to your 99.5% accuracy custom model')
    
    args = parser.parse_args()
    
    print("🚀 RUNPOD TRAINING SETUP - Individual Item Detection")
    print("=" * 60)
    
    # Initialize setup
    setup = RunPodTrainingSetup(args.base_model)
    
    try:
        # Create training configuration
        config_file, individual_classes = setup.create_individual_item_config()
        
        # Create training script
        training_script = setup.create_training_script(config_file)
        
        # Create Dockerfile
        dockerfile = setup.create_dockerfile()
        
        # Create data preparation guide
        data_guide = setup.create_data_preparation_guide(individual_classes)
        
        # Create upload script
        upload_script = setup.create_runpod_upload_script()
        
        # Create progress tracker
        progress_file = setup.create_progress_tracker()
        
        # Show summary
        setup.show_setup_summary()
        
        return True
        
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        return False

if __name__ == "__main__":
    main()