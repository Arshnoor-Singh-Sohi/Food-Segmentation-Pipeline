"""Fast YOLO-based food segmentation and analysis."""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import json
import time
from pathlib import Path
from datetime import datetime
import pandas as pd
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)

class FastFoodSegmentation:
    """Fast food segmentation using YOLO - No 10-minute waits!"""
    
    # In your FastFoodSegmentation class
    def __init__(self, model_size='n'):
        self.model_size = model_size
        
        # Test multiple models in order of preference
        models_to_try = [
            ('yolov8n-seg.pt', 'YOLOv8 Segmentation'),
            ('yolov8n.pt', 'YOLOv8 Detection'), 
            (f'yolov8{model_size}-seg.pt', f'YOLOv8{model_size} Segmentation')
        ]
        
        for model_name, description in models_to_try:
            try:
                self.model = YOLO(model_name)
                print(f"[OK] Loaded: {description}")
                
                # Test the model quickly
                import tempfile
                import numpy as np
                from PIL import Image
                
                # Create a small test image
                test_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
                test_path = 'test_image.jpg'
                Image.fromarray(test_img).save(test_path)
                
                # Quick test
                results = self.model(test_path, verbose=False)
                print(f"[STATS] Model working: {len(results)} results")
                
                # Clean up
                import os
                if os.path.exists(test_path):
                    os.remove(test_path)
                
                break  # Success, use this model
                
            except Exception as e:
                print(f"[FAIL] {model_name} failed: {e}")
                continue
        
                    
        # Nutrition database (simplified)
        self.nutrition_db = {
            'apple': {'cal_per_100g': 52, 'protein': 0.3, 'carbs': 14, 'fat': 0.2},
            'banana': {'cal_per_100g': 89, 'protein': 1.1, 'carbs': 23, 'fat': 0.3},
            'orange': {'cal_per_100g': 47, 'protein': 0.9, 'carbs': 12, 'fat': 0.1},
            'pizza': {'cal_per_100g': 266, 'protein': 11, 'carbs': 33, 'fat': 10},
            'sandwich': {'cal_per_100g': 250, 'protein': 12, 'carbs': 30, 'fat': 8},
            'cake': {'cal_per_100g': 350, 'protein': 5, 'carbs': 55, 'fat': 12},
            'donut': {'cal_per_100g': 400, 'protein': 6, 'carbs': 45, 'fat': 20},
            'hot_dog': {'cal_per_100g': 290, 'protein': 11, 'carbs': 3, 'fat': 26},
            'broccoli': {'cal_per_100g': 34, 'protein': 2.8, 'carbs': 7, 'fat': 0.4},
            'carrot': {'cal_per_100g': 41, 'protein': 0.9, 'carbs': 10, 'fat': 0.2},
            'cup': {'cal_per_100g': 0, 'protein': 0, 'carbs': 0, 'fat': 0},
            'bowl': {'cal_per_100g': 0, 'protein': 0, 'carbs': 0, 'fat': 0},
        }
        
        print(f"Initialized FastFoodSegmentation with YOLOv8{model_size}-seg")
    
    def process_single_image(self, image_path: str, save_visualization: bool = True) -> Dict[str, Any]:
        """Process a single image - Fast processing in seconds!"""
        start_time = time.time()
        
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Run YOLO segmentation
            results = self.model(image_rgb, conf=0.25, iou=0.45)
            
            # Process results
            food_items = self._extract_food_items(results[0], image_rgb.shape)
            
            # Calculate totals
            total_nutrition = self._calculate_total_nutrition(food_items)
            
            processing_time = time.time() - start_time
            
            # Create result structure
            result = {
                'image_info': {
                    'path': image_path,
                    'filename': Path(image_path).name,
                    'size': f"{image_rgb.shape[1]}x{image_rgb.shape[0]}",
                    'processing_time_seconds': round(processing_time, 2)
                },
                'analysis_summary': {
                    'total_items_detected': len(food_items),
                    'food_items_count': len([item for item in food_items if item['is_food']]),
                    'non_food_items_count': len([item for item in food_items if not item['is_food']]),
                    'avg_confidence': round(np.mean([item['confidence'] for item in food_items]), 3) if food_items else 0,
                    'timestamp': datetime.now().isoformat()
                },
                'food_items': food_items,
                'nutrition_totals': total_nutrition,
                'model_info': {
                    'model_type': f'YOLOv8{self.model_size}-seg',
                    'confidence_threshold': 0.25,
                    'iou_threshold': 0.45
                }
            }
            
            # Save visualization if requested
            if save_visualization:
                viz_path = self._create_visualization(image_rgb, food_items, image_path)
                result['visualization_path'] = viz_path
            
            print(f"Processed {Path(image_path).name} in {processing_time:.2f}s - {len(food_items)} items found")
            return result
            
        except Exception as e:
            print(f"Failed to process {image_path}: {e}")
            return {
                'image_info': {'path': image_path, 'filename': Path(image_path).name},
                'error': str(e),
                'processing_time_seconds': time.time() - start_time,
                'timestamp': datetime.now().isoformat()
            }
    
    def _extract_food_items(self, result, image_shape) -> List[Dict[str, Any]]:
        """Extract food items from YOLO results."""
        food_items = []
        
        if result.masks is None or len(result.masks) == 0:
            return food_items
        
        for i, (box, mask, conf, cls) in enumerate(zip(
            result.boxes.xyxy.cpu().numpy(),
            result.masks.data.cpu().numpy(),
            result.boxes.conf.cpu().numpy(),
            result.boxes.cls.cpu().numpy()
        )):
            class_id = int(cls)
            class_name = self.model.names.get(class_id, f"class_{class_id}")
            
            # Calculate mask properties
            mask_area = int(np.sum(mask))  # Convert to int
            image_area = image_shape[0] * image_shape[1]
            area_percentage = float((mask_area / image_area) * 100)  # Convert to float
            
            # Estimate portion size and nutrition
            portion_grams = self._estimate_portion_size(mask_area, class_name)
            nutrition = self._calculate_nutrition(class_name, portion_grams)
            
            # Determine if it's food or not
            is_food = self._is_food_item(class_name)
            
            food_item = {
                'id': i,
                'name': class_name,
                'confidence': float(conf),  # Convert numpy float32 to Python float
                'is_food': is_food,
                'bbox': {
                    'x1': int(box[0]), 'y1': int(box[1]),
                    'x2': int(box[2]), 'y2': int(box[3]),
                    'width': int(box[2] - box[0]),
                    'height': int(box[3] - box[1])
                },
                'mask_info': {
                    'area_pixels': mask_area,
                    'area_percentage': round(area_percentage, 2)
                },
                'portion_estimate': {
                    'estimated_grams': portion_grams,
                    'confidence': 'rough_estimate'
                },
                'nutrition': nutrition
            }
            
            food_items.append(food_item)
        
        return food_items

    def _estimate_portion_size(self, mask_area: int, class_name: str) -> int:
        """Estimate portion size in grams based on mask area."""
        base_density = 0.5  # grams per pixel (rough average)
        
        # Food-specific adjustments
        density_multipliers = {
            'apple': 1.2, 'banana': 1.0, 'orange': 1.1,
            'pizza': 0.8, 'cake': 0.7, 'bread': 0.6,
            'broccoli': 0.3, 'carrot': 0.9
        }
        
        multiplier = density_multipliers.get(class_name, 1.0)
        estimated_grams = int(mask_area * base_density * multiplier)
        
        return max(10, min(estimated_grams, 1000))  # Reasonable bounds
    
    def _calculate_nutrition(self, class_name: str, portion_grams: int) -> Dict[str, float]:
        """Calculate nutrition for the estimated portion."""
        nutrition_data = self.nutrition_db.get(class_name, {
            'cal_per_100g': 150, 'protein': 5, 'carbs': 20, 'fat': 3
        })
        
        scale_factor = portion_grams / 100
        
        return {
            'calories': round(nutrition_data['cal_per_100g'] * scale_factor, 1),
            'protein_g': round(nutrition_data['protein'] * scale_factor, 1),
            'carbs_g': round(nutrition_data['carbs'] * scale_factor, 1),
            'fat_g': round(nutrition_data['fat'] * scale_factor, 1),
            'portion_grams': portion_grams
        }
    
    def _is_food_item(self, class_name: str) -> bool:
        """Determine if detected item is actually food."""
        food_items = {
            'apple', 'banana', 'orange', 'sandwich', 'pizza', 'cake',
            'donut', 'hot_dog', 'broccoli', 'carrot'
        }
        
        non_food_items = {
            'cup', 'bowl', 'fork', 'knife', 'spoon', 'bottle', 'wine_glass'
        }
        
        if class_name in food_items:
            return True
        elif class_name in non_food_items:
            return False
        else:
            return 'food' in class_name.lower() or 'fruit' in class_name.lower()
    
    def _calculate_total_nutrition(self, food_items: List[Dict]) -> Dict[str, float]:
        """Calculate total nutrition from all food items."""
        totals = {'calories': 0, 'protein_g': 0, 'carbs_g': 0, 'fat_g': 0}
        
        for item in food_items:
            if item['is_food']:
                nutrition = item['nutrition']
                for key in totals:
                    totals[key] += nutrition.get(key, 0)
        
        return {k: round(v, 1) for k, v in totals.items()}
    
    def _create_visualization(self, image_rgb: np.ndarray, food_items: List[Dict], image_path: str) -> str:
        """Create and save visualization."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Original image with bounding boxes
        ax1.imshow(image_rgb)
        ax1.set_title(f"Detected Items ({len(food_items)} total)")
        ax1.axis('off')
        
        colors = plt.cm.Set3(np.linspace(0, 1, 12))
        
        for i, item in enumerate(food_items):
            bbox = item['bbox']
            color = colors[i % len(colors)]
            
            # Draw bounding box
            rect = plt.Rectangle(
                (bbox['x1'], bbox['y1']), bbox['width'], bbox['height'],
                linewidth=2, edgecolor=color, facecolor='none'
            )
            ax1.add_patch(rect)
            
            # Add label
            label = f"{item['name']} ({item['confidence']:.2f})"
            ax1.text(bbox['x1'], bbox['y1']-10, label, 
                    color=color, fontsize=10, weight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))
        
        # Nutrition breakdown
        food_only = [item for item in food_items if item['is_food']]
        if food_only:
            calories = [item['nutrition']['calories'] for item in food_only]
            names = [item['name'] for item in food_only]
            
            ax2.bar(names, calories, color=colors[:len(food_only)])
            ax2.set_title('Calories by Food Item')
            ax2.set_ylabel('Calories')
            ax2.tick_params(axis='x', rotation=45)
        else:
            ax2.text(0.5, 0.5, 'No food items detected', 
                    ha='center', va='center', transform=ax2.transAxes, fontsize=14)
            ax2.set_title('No Food Items Found')
        
        # Save visualization
        output_dir = Path("data/output/yolo_results/visualizations")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        viz_filename = f"{Path(image_path).stem}_analysis.png"
        viz_path = output_dir / viz_filename
        
        plt.tight_layout()
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return str(viz_path)