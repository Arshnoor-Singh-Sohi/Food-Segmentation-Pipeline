#!/usr/bin/env python3
"""
Food YOLO Trainer - The heart of your custom model training
This module handles the actual training process, optimized specifically for food recognition
"""

import os
import shutil
import yaml
from pathlib import Path
import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

class FoodYOLOTrainer:
    """
    Train YOLO models specifically optimized for food detection
    
    This trainer understands the unique challenges of food recognition:
    - Similar-looking foods (croissant vs. dinner roll)
    - Varying portion sizes and presentations
    - Different lighting and background conditions
    - Cultural variations in food preparation
    """
    
    def __init__(self, dataset_yaml, config_path="config/training_config.yaml"):
        """
        Initialize the food-specific YOLO trainer
        
        Args:
            dataset_yaml: Path to the dataset YAML configuration
            config_path: Path to training configuration file
        """
        self.dataset_yaml = Path(dataset_yaml)
        self.config_path = Path(config_path)
        
        # Load training configuration
        self.config = self._load_training_config()
        
        # Set up device (GPU if available, CPU otherwise)
        self.device = self._setup_device()
        
        # Create directories for training outputs
        self.training_dir = Path("data/trained_models")
        self.training_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"[FOOD] FoodYOLOTrainer initialized")
        print(f"[STATS] Dataset: {self.dataset_yaml}")
        print(f"💻 Device: {self.device}")
        print(f"⚙️  Config: {self.config_path}")
    
    def _load_training_config(self):
        """Load training configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except FileNotFoundError:
            print(f"[WARN]  Config file {self.config_path} not found, using defaults")
            return self._get_default_config()
    
    def _get_default_config(self):
        """
        Return default configuration optimized for food detection
        These settings are carefully chosen based on food recognition research
        """
        return {
            'model': {
                'base_model': 'yolov8n.pt',  # Start with nano for speed
                'input_size': 640
            },
            'training': {
                'epochs': 50,  # Conservative for first run
                'batch': 16,
                'patience': 20,
                'device': 'auto',
                'workers': 4,
                'project': 'food_training_runs',
                'name': 'food_detector_v1'
            },
            'augmentation': {
                # These augmentations are specifically chosen for food
                'hsv_h': 0.015,  # Slight hue variation (food color differences)
                'hsv_s': 0.7,    # Strong saturation (freshness appearance)
                'hsv_v': 0.4,    # Value changes (lighting conditions)
                'degrees': 15,   # Rotation (camera/plate angles)
                'translate': 0.1, # Translation (camera position)
                'scale': 0.5,    # Scale variation (portion sizes)
                'flipud': 0.5,   # Vertical flip (different viewing angles)
                'fliplr': 0.5,   # Horizontal flip (left/right hand eating)
                'mosaic': 1.0,   # Mosaic (multiple foods in frame)
                'mixup': 0.2     # Mixup (food combinations)
            },
            'optimization': {
                'optimizer': 'AdamW',  # Generally better than SGD for vision
                'lr0': 0.001,          # Conservative learning rate
                'lrf': 0.01,           # Final learning rate factor
                'momentum': 0.937,
                'weight_decay': 0.0005,
                'warmup_epochs': 3.0,
                'warmup_momentum': 0.8,
                'box': 7.5,           # Bounding box loss weight
                'cls': 0.5,           # Classification loss weight
                'dfl': 1.5            # Distribution focal loss weight
            }
        }
    
    def _setup_device(self):
        """
        Set up the training device (GPU or CPU)
        Also provides helpful information about what to expect
        """
        if torch.cuda.is_available():
            device = 'cuda'
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"[TARGET] Using GPU: {gpu_name} ({gpu_memory:.1f}GB)")
            
            # Give guidance on batch size based on GPU memory
            if gpu_memory < 6:
                print("[TIP] Tip: With <6GB GPU memory, consider batch=8 or smaller")
            elif gpu_memory < 12:
                print("[TIP] Tip: With your GPU memory, batch=16 should work well")
            else:
                print("[TIP] Tip: With your GPU, you can try batch=32 for faster training")
                
        else:
            device = 'cpu'
            print("🖥️  Using CPU (training will be slower but still works)")
            print("[TIP] Tip: Consider using Google Colab or Kaggle for free GPU access")
        
        return device
    
    def train_food_detector(self, epochs=None, custom_name=None):
        """
        Train a YOLO detection model specifically for food recognition
        
        Args:
            epochs: Number of training epochs (overrides config)
            custom_name: Custom name for this training run
            
        Returns:
            tuple: (trained_model, training_results)
        """
        print("[RUN] Starting food detection model training...")
        print("=" * 60)
        
        # Load the base model
        base_model = self.config['model']['base_model']
        print(f"📥 Loading base model: {base_model}")
        model = YOLO(base_model)
        
        # Prepare training arguments
        training_config = self.config['training'].copy()
        
        # Override epochs if specified
        if epochs is not None:
            training_config['epochs'] = epochs
            print(f"[STEP] Using custom epochs: {epochs}")
        
        # Set custom name if provided
        if custom_name:
            training_config['name'] = custom_name
        
        # Combine all training arguments
        train_args = {
            'data': str(self.dataset_yaml),
            'device': self.device,
            **training_config,
            **self.config['augmentation'],
            **self.config['optimization']
        }
        
        # Print training configuration for transparency
        print("⚙️  Training Configuration:")
        key_params = ['epochs', 'batch', 'lr0', 'device']
        for param in key_params:
            if param in train_args:
                print(f"   {param}: {train_args[param]}")
        
        print(f"\n[TARGET] Training on dataset: {self.dataset_yaml}")
        
        # Start training
        print("🏋️  Starting training process...")
        # List of accepted keys in YOLO.train() — check your installed ultralytics version for exact supported keys
        valid_keys = {'data', 'epochs', 'batch', 'device', 'name', 'project', 'resume', 'imgsz', 'workers', 'optimizer'}

        # Filter train_args to only include valid keys
        filtered_train_args = {k: v for k, v in train_args.items() if k in valid_keys}

        # Now safely call train()
        results = model.train(**filtered_train_args)

        
        # Training completed - now save and organize results
        self._save_trained_model(model, training_config, results)
        
        print("[OK] Training completed successfully!")
        return model, results
    
    def train_food_segmentation(self, epochs=None, custom_name=None):
        """
        Train a YOLO segmentation model for precise food boundaries
        This is useful for portion size estimation and nutrition calculation
        
        Args:
            epochs: Number of training epochs
            custom_name: Custom name for this training run
            
        Returns:
            tuple: (trained_model, training_results)
        """
        print("🎨 Starting food segmentation model training...")
        print("=" * 60)
        
        # Use segmentation base model
        print("📥 Loading segmentation base model...")
        model = YOLO('yolov8n-seg.pt')  # Segmentation version
        
        # Prepare segmentation-specific training arguments
        training_config = self.config['training'].copy()
        training_config['task'] = 'segment'
        
        if epochs is not None:
            training_config['epochs'] = epochs
        
        if custom_name:
            training_config['name'] = f"{custom_name}_segmentation"
        else:
            training_config['name'] = f"{training_config['name']}_segmentation"
        
        # Segmentation-specific parameters
        segmentation_params = {
            'mask_ratio': 4,        # Mask to box ratio
            'overlap_mask': True,   # Allow overlapping masks
        }
        
        # Combine arguments
        train_args = {
            'data': str(self.dataset_yaml),
            'device': self.device,
            **training_config,
            **self.config['augmentation'],
            **self.config['optimization'],
            **segmentation_params
        }
        
        print("[TARGET] Training segmentation model...")
        results = model.train(**train_args)
        
        # Save the segmentation model
        self._save_trained_model(model, training_config, results, model_type='segmentation')
        
        print("[OK] Segmentation training completed!")
        return model, results
    
    def _save_trained_model(self, model, training_config, results, model_type='detection'):
        """
        Save the trained model and organize the results in your project structure
        
        Args:
            model: The trained YOLO model
            training_config: Training configuration used
            results: Training results from YOLO
            model_type: Type of model ('detection' or 'segmentation')
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"food_{model_type}_{timestamp}"
        
        # Create model directory
        model_dir = self.training_dir / model_name
        model_dir.mkdir(exist_ok=True)
        
        # Find the best model from training
        project_name = training_config.get('project', 'runs/detect')
        run_name = training_config.get('name', 'train')
        best_model_path = Path(project_name) / run_name / 'weights' / 'best.pt'
        
        if best_model_path.exists():
            # Copy the best model to your organized structure
            final_model_path = model_dir / 'best.pt'
            shutil.copy2(best_model_path, final_model_path)
            
            # Also create a convenience symlink in your models directory
            symlink_path = Path("data/models") / f"custom_food_{model_type}.pt"
            symlink_path.parent.mkdir(exist_ok=True)
            
            if symlink_path.exists() or symlink_path.is_symlink():
                symlink_path.unlink()  # Remove old symlink
            
            try:
                symlink_path.symlink_to(final_model_path.absolute())
                print(f"🔗 Model accessible at: {symlink_path}")
            except OSError:
                # Fallback: copy instead of symlink (Windows compatibility)
                shutil.copy2(final_model_path, symlink_path)
                print(f"[FILE] Model copied to: {symlink_path}")
            
            print(f"💾 Best model saved: {final_model_path}")
        
        # Save training metadata
        metadata = {
            'model_type': model_type,
            'training_date': timestamp,
            'base_model': self.config['model']['base_model'],
            'dataset': str(self.dataset_yaml),
            'epochs': training_config.get('epochs', 'unknown'),
            'device': self.device,
            'final_model_path': str(model_dir / 'best.pt')
        }
        
        metadata_path = model_dir / 'training_metadata.yaml'
        with open(metadata_path, 'w') as f:
            yaml.dump(metadata, f, default_flow_style=False)
        
        print(f"[INFO] Training metadata saved: {metadata_path}")
    
    def resume_training(self, model_path, additional_epochs=50):
        """
        Resume training from a previously saved model
        Useful if training was interrupted or you want to continue improving
        
        Args:
            model_path: Path to the saved model to resume from
            additional_epochs: Number of additional epochs to train
            
        Returns:
            tuple: (model, results)
        """
        print(f"[STEP] Resuming training from: {model_path}")
        
        # Load the existing model
        model = YOLO(model_path)
        
        # Resume training with additional epochs
        train_args = {
            'data': str(self.dataset_yaml),
            'epochs': additional_epochs,
            'device': self.device,
            'resume': True,  # This tells YOLO to resume training
            **self.config['training'],
            **self.config['augmentation'],
            **self.config['optimization']
        }
        
        print(f"🏋️  Resuming for {additional_epochs} additional epochs...")
        results = model.train(**train_args)
        
        print("[OK] Resume training completed!")
        return model, results
    
    def validate_model(self, model_path, test_images_dir="data/input"):
        """
        Validate a trained model on test images
        This helps you understand how well your model performs
        
        Args:
            model_path: Path to the trained model
            test_images_dir: Directory containing test images
            
        Returns:
            dict: Validation results
        """
        print(f"[TEST] Validating model: {model_path}")
        
        # Load the trained model
        model = YOLO(model_path)
        
        # Run validation on test images
        test_path = Path(test_images_dir)
        if not test_path.exists():
            print(f"[FAIL] Test directory {test_images_dir} not found")
            return None
        
        # Get test images
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        test_images = []
        for ext in image_extensions:
            test_images.extend(list(test_path.glob(f"*{ext}")))
        
        if not test_images:
            print(f"[FAIL] No test images found in {test_images_dir}")
            return None
        
        print(f"🖼️  Testing on {len(test_images)} images...")
        
        # Run inference on test images
        validation_results = {
            'model_path': model_path,
            'test_images_count': len(test_images),
            'results': []
        }
        
        for img_path in test_images:
            results = model(str(img_path))
            
            # Extract key metrics
            result_info = {
                'image': img_path.name,
                'detections': len(results[0].boxes) if results[0].boxes is not None else 0,
                'max_confidence': float(results[0].boxes.conf.max()) if results[0].boxes is not None and len(results[0].boxes) > 0 else 0.0
            }
            validation_results['results'].append(result_info)
        
        # Calculate summary statistics
        detections = [r['detections'] for r in validation_results['results']]
        confidences = [r['max_confidence'] for r in validation_results['results']]
        
        validation_results['summary'] = {
            'avg_detections_per_image': sum(detections) / len(detections),
            'avg_max_confidence': sum(confidences) / len(confidences),
            'images_with_detections': len([d for d in detections if d > 0])
        }
        
        print("[STATS] Validation Summary:")
        print(f"   Average detections per image: {validation_results['summary']['avg_detections_per_image']:.1f}")
        print(f"   Average confidence: {validation_results['summary']['avg_max_confidence']:.3f}")
        print(f"   Images with detections: {validation_results['summary']['images_with_detections']}/{len(test_images)}")
        
        return validation_results