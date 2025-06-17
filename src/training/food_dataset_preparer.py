#!/usr/bin/env python3
"""
Fixed Food Dataset Preparer - Complete working version
Save this as: src/training/food_dataset_preparer.py
"""

import os
import json
import shutil
from pathlib import Path
import yaml
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

class FoodDatasetPreparer:
    """
    Prepare food datasets for YOLO training
    FIXED VERSION: Recursively finds all images and creates proper training data
    """
    
    def __init__(self, base_dir="data/training"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Dataset paths
        self.raw_datasets_dir = self.base_dir / "raw_datasets"
        self.food_training_dir = self.base_dir / "food_training"
        self.annotations_dir = self.base_dir / "annotations"
        
        # YOLO format structure
        self.images_dir = self.food_training_dir / "images"
        self.labels_dir = self.food_training_dir / "labels"
        
        for dir_path in [self.raw_datasets_dir, self.food_training_dir, 
                        self.annotations_dir, self.images_dir, self.labels_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def create_sample_dataset_with_real_data(self, source_dir="data/input", num_classes=5):
        """
        Create a dataset using your existing images with automatically generated labels
        Now searches recursively for ALL images in subdirectories
        
        Args:
            source_dir: Directory containing your existing food images
            num_classes: Number of food categories to create
            
        Returns:
            tuple: (dataset_yaml_path, food_categories)
        """
        print("🥪 Creating sample food dataset with REAL training data...")
        
        source_path = Path(source_dir)
        if not source_path.exists():
            print(f"[FAIL] Source directory {source_dir} not found!")
            print("[TIP] Please put some food images in data/input/ first")
            return None, None
        
        # Find ALL images recursively in your input directory
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        image_files = []
        
        print(f"🔍 Searching for images in {source_dir} (including subdirectories)...")
        
        # Search recursively using **/* pattern
        for ext in image_extensions:
            # Recursive search with **/*
            image_files.extend(list(source_path.rglob(f"*{ext}")))
            image_files.extend(list(source_path.rglob(f"*{ext.upper()}")))
        
        # Remove duplicates and filter out hidden files
        image_files = list(set([f for f in image_files if not f.name.startswith('.')]))
        
        if not image_files:
            print(f"[FAIL] No images found in {source_dir} (searched recursively)!")
            print("[TIP] Please add some food images to data/input/ directory")
            print(f"[TIP] Looking for extensions: {', '.join(image_extensions)}")
            return None, None
        
        print(f"📸 Found {len(image_files)} images total!")
        
        # Show directory breakdown
        dir_counts = {}
        for img in image_files:
            try:
                relative_dir = str(img.parent.relative_to(source_path))
            except ValueError:
                relative_dir = str(img.parent)
            dir_counts[relative_dir] = dir_counts.get(relative_dir, 0) + 1
        
        print("📂 Images by directory:")
        for dir_name, count in sorted(dir_counts.items()):
            print(f"   {dir_name}: {count} images")
        
        # Simple food categories for initial training
        food_categories = ['food']  # Start with just one class: "food"
        
        # Create YOLO directory structure
        train_images = self.images_dir / "train"
        val_images = self.images_dir / "val"
        train_labels = self.labels_dir / "train"
        val_labels = self.labels_dir / "val"
        
        for dir_path in [train_images, val_images, train_labels, val_labels]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Split images into train/val
        train_files, val_files = train_test_split(
            image_files, test_size=0.2, random_state=42
        )
        
        print(f"📊 Splitting: {len(train_files)} train, {len(val_files)} validation")
        
        # Copy images and create simple labels
        self._copy_images_and_create_labels(train_files, train_images, train_labels)
        self._copy_images_and_create_labels(val_files, val_images, val_labels)
        
        # Create dataset YAML
        dataset_yaml = {
            'path': str(self.food_training_dir.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'nc': len(food_categories),
            'names': food_categories
        }
        
        yaml_path = self.food_training_dir / "food_dataset.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump(dataset_yaml, f, default_flow_style=False)
        
        print(f"[LOG] Dataset configuration saved to: {yaml_path}")
        print(f"🏷️  Food categories: {', '.join(food_categories)}")
        print(f"[OK] Dataset created successfully with {len(image_files)} images!")
        
        return yaml_path, food_categories
    
    def create_from_existing_images_smart(self, source_dir="data/input"):
        """
        SMART VERSION: Use your existing images with better automatic labeling
        This tries to be smarter about creating bounding boxes
        
        Args:
            source_dir: Directory containing your existing food images
            
        Returns:
            str: Path to the dataset YAML file
        """
        source_path = Path(source_dir)
        if not source_path.exists():
            print(f"[WARN]  Source directory {source_dir} not found")
            return None
            
        print(f"📂 Creating smart dataset from existing images in {source_dir}")
        
        # Find all image files recursively
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        image_files = []
        
        print(f"🔍 Searching for images recursively...")
        
        for ext in image_extensions:
            image_files.extend(list(source_path.rglob(f"*{ext}")))
            image_files.extend(list(source_path.rglob(f"*{ext.upper()}")))
        
        # Remove duplicates and filter out hidden files
        image_files = list(set([f for f in image_files if not f.name.startswith('.')]))
        
        if not image_files:
            print(f"[FAIL] No images found in {source_dir}")
            return None
        
        print(f"🖼️  Found {len(image_files)} images")
        
        # Split into train/val sets
        train_files, val_files = train_test_split(
            image_files, test_size=0.2, random_state=42
        )
        
        # Create directory structure
        train_images = self.images_dir / "train"
        val_images = self.images_dir / "val"
        train_labels = self.labels_dir / "train"
        val_labels = self.labels_dir / "val"
        
        for dir_path in [train_images, val_images, train_labels, val_labels]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Process images with smarter labeling
        self._smart_copy_and_label(train_files, train_images, train_labels)
        self._smart_copy_and_label(val_files, val_images, val_labels)
        
        print(f"[OK] Smart dataset prepared: {len(train_files)} train, {len(val_files)} val")
        
        # Create dataset YAML
        dataset_yaml = {
            'path': str(self.food_training_dir.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'nc': 1,  # Just one class: food
            'names': ['food']
        }
        
        yaml_path = self.food_training_dir / "existing_images_dataset.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump(dataset_yaml, f, default_flow_style=False)
        
        return yaml_path
    
    def _copy_images_and_create_labels(self, image_files, images_dir, labels_dir):
        """
        Copy images and create simple YOLO labels for them
        Handles filename conflicts by adding numbers
        """
        for img_file in tqdm(image_files, desc=f"Processing images for {images_dir.name}"):
            # Handle potential filename conflicts
            dest_img = images_dir / img_file.name
            counter = 1
            original_stem = img_file.stem
            
            while dest_img.exists():
                new_name = f"{original_stem}_{counter}{img_file.suffix}"
                dest_img = images_dir / new_name
                counter += 1
            
            # Copy image
            shutil.copy2(img_file, dest_img)
            
            # Create corresponding label file
            label_file = labels_dir / f"{dest_img.stem}.txt"
            
            # Create a simple label covering most of the image
            with open(label_file, 'w') as f:
                # Class 0 (food), centered, covering 80% of image
                f.write("0 0.5 0.5 0.8 0.8\n")
    
    def _smart_copy_and_label(self, image_files, images_dir, labels_dir):
        """
        Copy images and create smarter labels using basic image processing
        """
        for img_file in tqdm(image_files, desc=f"Smart processing {images_dir.name}"):
            try:
                # Handle filename conflicts
                dest_img = images_dir / img_file.name
                counter = 1
                original_stem = img_file.stem
                
                while dest_img.exists():
                    new_name = f"{original_stem}_{counter}{img_file.suffix}"
                    dest_img = images_dir / new_name
                    counter += 1
                
                # Copy image
                shutil.copy2(img_file, dest_img)
                
                # Try to create a smarter bounding box
                image = cv2.imread(str(img_file))
                if image is None:
                    # Fallback to simple label
                    self._create_simple_label(labels_dir / f"{dest_img.stem}.txt")
                    continue
                
                # Use simple image processing to find likely food regions
                bbox = self._find_food_region(image)
                
                # Create label file
                label_file = labels_dir / f"{dest_img.stem}.txt"
                with open(label_file, 'w') as f:
                    # Format: class_id center_x center_y width height
                    f.write(f"0 {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")
                    
            except Exception as e:
                print(f"[WARN]  Error processing {img_file.name}: {e}")
                # Create fallback simple label
                dest_img = images_dir / img_file.name
                self._create_simple_label(labels_dir / f"{dest_img.stem}.txt")
    
    def _find_food_region(self, image):
        """
        Use basic image processing to estimate where food might be
        """
        height, width = image.shape[:2]
        
        # Convert to different color spaces
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Look for regions with food-like colors
        food_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        
        # Brown/tan colors (bread, meat)
        lower_brown = np.array([10, 50, 50])
        upper_brown = np.array([25, 255, 255])
        brown_mask = cv2.inRange(hsv, lower_brown, upper_brown)
        
        # Red colors (tomatoes, sauces)
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 50, 50])
        upper_red2 = np.array([180, 255, 255])
        red_mask = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)
        
        # Green colors (vegetables)
        lower_green = np.array([40, 50, 50])
        upper_green = np.array([80, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Combine masks
        food_mask = brown_mask + red_mask + green_mask
        
        # Find the largest contour (likely food region)
        contours, _ = cv2.findContours(food_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Get the largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Convert to normalized YOLO format
            center_x = (x + w/2) / width
            center_y = (y + h/2) / height
            norm_width = w / width
            norm_height = h / height
            
            # Make sure the bounding box is reasonable
            center_x = max(0.1, min(0.9, center_x))
            center_y = max(0.1, min(0.9, center_y))
            norm_width = max(0.2, min(0.8, norm_width))
            norm_height = max(0.2, min(0.8, norm_height))
            
            return (center_x, center_y, norm_width, norm_height)
        
        else:
            # Fallback: assume food is in the center of the image
            return (0.5, 0.5, 0.7, 0.7)
    
    def _create_simple_label(self, label_path):
        """Create a simple fallback label"""
        with open(label_path, 'w') as f:
            f.write("0 0.5 0.5 0.7 0.7\n")
    
    def debug_image_search(self, source_dir="data/input"):
        """Debug function to see what images are found"""
        source_path = Path(source_dir)
        if not source_path.exists():
            print(f"Directory {source_dir} doesn't exist")
            return
            
        print(f"🔍 Debugging image search in: {source_path.absolute()}")
        
        # Check what's in the directory
        all_files = list(source_path.rglob("*"))
        print(f"📁 Total files/folders found: {len(all_files)}")
        
        # Group by extension
        extensions = {}
        for file in all_files:
            if file.is_file():
                ext = file.suffix.lower()
                extensions[ext] = extensions.get(ext, 0) + 1
        
        print("📋 Files by extension:")
        for ext, count in sorted(extensions.items()):
            print(f"   {ext}: {count} files")
        
        # Look specifically for image extensions
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        image_files = []
        
        for ext in image_extensions:
            found = list(source_path.rglob(f"*{ext}"))
            found.extend(list(source_path.rglob(f"*{ext.upper()}")))
            image_files.extend(found)
        
        image_files = list(set(image_files))
        print(f"🖼️  Total image files found: {len(image_files)}")
        
        if image_files:
            print("📂 First 10 image paths:")
            for img in image_files[:10]:
                print(f"   {img}")
        
        return image_files
    
    def validate_dataset(self, dataset_yaml_path):
        """Validate that the dataset has actual training data"""
        print("🔍 Validating dataset...")
        
        if not dataset_yaml_path or not Path(dataset_yaml_path).exists():
            return {
                'valid': False,
                'issues': ['Dataset YAML file not found'],
                'stats': {}
            }
        
        with open(dataset_yaml_path, 'r') as f:
            dataset_config = yaml.safe_load(f)
        
        dataset_path = Path(dataset_config['path'])
        
        validation_results = {
            'valid': True,
            'issues': [],
            'stats': {}
        }
        
        # Check directories and count files
        train_images = dataset_path / dataset_config['train']
        val_images = dataset_path / dataset_config['val']
        train_labels = dataset_path / 'labels' / 'train'
        val_labels = dataset_path / 'labels' / 'val'
        
        for name, path in [
            ('train_images', train_images),
            ('val_images', val_images),
            ('train_labels', train_labels),
            ('val_labels', val_labels)
        ]:
            if not path.exists():
                validation_results['issues'].append(f"Missing directory: {path}")
                validation_results['valid'] = False
                validation_results['stats'][name] = 0
            else:
                # Count files
                if 'images' in name:
                    file_count = len([f for f in path.glob('*') if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']])
                else:
                    file_count = len([f for f in path.glob('*.txt')])
                validation_results['stats'][name] = file_count
                
                if file_count == 0:
                    validation_results['issues'].append(f"No files in {name}: {path}")
                    validation_results['valid'] = False
        
        # Check that images and labels match
        if validation_results['stats'].get('train_images', 0) != validation_results['stats'].get('train_labels', 0):
            validation_results['issues'].append("Mismatch between train images and labels")
            validation_results['valid'] = False
            
        if validation_results['stats'].get('val_images', 0) != validation_results['stats'].get('val_labels', 0):
            validation_results['issues'].append("Mismatch between val images and labels")
            validation_results['valid'] = False
        
        # Check minimum data requirements
        total_images = validation_results['stats'].get('train_images', 0) + validation_results['stats'].get('val_images', 0)
        if total_images < 2:
            validation_results['issues'].append(f"Not enough training data: only {total_images} images")
            validation_results['valid'] = False
        
        # Print validation results
        if validation_results['valid']:
            print("[OK] Dataset validation passed!")
            print("[STATS] Dataset statistics:")
            for key, value in validation_results['stats'].items():
                print(f"   {key}: {value} files")
        else:
            print("[FAIL] Dataset validation failed!")
            print("[ERROR] Issues found:")
            for issue in validation_results['issues']:
                print(f"   - {issue}")
        
        return validation_results