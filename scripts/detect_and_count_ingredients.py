#!/usr/bin/env python3
"""
Detect and count simple ingredients (like banana clusters)
Fixed version with better error handling and custom model support
"""

import cv2
import numpy as np
from pathlib import Path
import argparse
from ultralytics import YOLO
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datetime import datetime
import json
import torch
import warnings
warnings.filterwarnings('ignore')

class IngredientCounter:
    """Detect and count individual ingredients in images"""
    
    def __init__(self, model_path: str = None, confidence_threshold: float = 0.5, use_cpu: bool = False):
        """
        Initialize ingredient counter
        
        Args:
            model_path: Path to YOLO model (None for default)
            confidence_threshold: Minimum confidence for detections
            use_cpu: Force CPU usage (helps with some errors)
        """
        # Set device
        if use_cpu:
            self.device = 'cpu'
        else:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        print(f"🔧 Using device: {self.device}")
        
        # Load model
        try:
            if model_path and Path(model_path).exists():
                print(f"🎯 Loading custom model: {model_path}")
                self.model = YOLO(model_path)
                self.model_type = "custom"
            else:
                print("📦 Loading default YOLOv8 model")
                # Use yolov8n.pt for faster loading and less memory
                self.model = YOLO('yolov8n.pt')  # Changed to nano model
                self.model_type = "default"
                
            # Move model to device
            self.model.to(self.device)
            
        except Exception as e:
            print(f"⚠️ Error loading model: {e}")
            print("Trying CPU mode...")
            self.device = 'cpu'
            self.model = YOLO('yolov8n.pt')
            self.model_type = "default"
        
        self.confidence_threshold = confidence_threshold
        
        # Ingredient mapping for common fruits/vegetables
        self.ingredient_classes = {
            # COCO dataset classes that are ingredients
            46: 'banana',
            47: 'apple',
            49: 'orange',
            50: 'broccoli',
            51: 'carrot',
            52: 'hot dog',  # Can be counted as ingredient
            53: 'pizza',
            54: 'donut',
            55: 'cake',
            # Add more as needed
        }
        
        # Text-based ingredient detection
        self.ingredient_keywords = [
            'banana', 'apple', 'orange', 'carrot', 'broccoli',
            'tomato', 'potato', 'strawberry', 'grape', 'berry',
            'vegetable', 'fruit', 'food'
        ]
    
    def detect_and_count(self, image_path: str):
        """
        Detect and count ingredients in image
        
        Returns:
            dict: Detection results with counts
        """
        print(f"\n🔍 Analyzing: {image_path}")
        
        # Verify image exists
        if not Path(image_path).exists():
            return {
                'status': 'error',
                'message': f'Image not found: {image_path}',
                'counts': {},
                'detections': []
            }
        
        # Read image
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError("Could not read image")
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Error reading image: {e}',
                'counts': {},
                'detections': []
            }
        
        # Run detection with error handling
        try:
            # Run inference with explicit device setting
            results = self.model.predict(
                source=image_path,
                conf=self.confidence_threshold,
                device=self.device,
                verbose=False
            )
        except RuntimeError as e:
            if "could not create a primitive" in str(e) or "CUDA" in str(e):
                print("⚠️ GPU error detected, switching to CPU...")
                self.device = 'cpu'
                self.model.to('cpu')
                results = self.model.predict(
                    source=image_path,
                    conf=self.confidence_threshold,
                    device='cpu',
                    verbose=False
                )
            else:
                raise e
        
        if not results or len(results) == 0:
            return {
                'status': 'no_detections',
                'counts': {},
                'detections': [],
                'model_used': self.model_type
            }
        
        # Process detections
        detections = []
        ingredient_counts = {}
        
        result = results[0]
        
        if result.boxes is not None and len(result.boxes) > 0:
            for i, box in enumerate(result.boxes):
                # Get detection info
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                confidence = float(box.conf)
                class_id = int(box.cls)
                
                # Get class name - handle both custom and default models
                if hasattr(result, 'names') and result.names:
                    class_name = result.names.get(class_id, f"class_{class_id}")
                else:
                    # For COCO model
                    class_name = self.ingredient_classes.get(class_id, f"class_{class_id}")
                
                # Check if this is an ingredient
                ingredient_type = self._classify_ingredient(class_name, class_id)
                
                if ingredient_type:
                    detection = {
                        'id': i,
                        'ingredient': ingredient_type,
                        'original_class': class_name,
                        'confidence': confidence,
                        'bbox': {
                            'x1': int(x1), 'y1': int(y1),
                            'x2': int(x2), 'y2': int(y2)
                        },
                        'center': {
                            'x': int((x1 + x2) / 2),
                            'y': int((y1 + y2) / 2)
                        },
                        'area': int((x2 - x1) * (y2 - y1))
                    }
                    
                    detections.append(detection)
                    
                    # Update count
                    if ingredient_type not in ingredient_counts:
                        ingredient_counts[ingredient_type] = 0
                    ingredient_counts[ingredient_type] += 1
        
        # Check for clusters
        cluster_adjustments = self._detect_clusters(detections, image.shape)
        
        return {
            'status': 'success',
            'image_path': image_path,
            'image_size': {'width': image.shape[1], 'height': image.shape[0]},
            'counts': ingredient_counts,
            'total_items': sum(ingredient_counts.values()),
            'detections': detections,
            'cluster_info': cluster_adjustments,
            'model_used': self.model_type,
            'device_used': self.device
        }
    
    def _classify_ingredient(self, class_name: str, class_id: int) -> str:
        """Map detected class to ingredient type"""
        # For custom food model
        if self.model_type == "custom":
            # Your custom model likely detects food items directly
            class_lower = class_name.lower()
            for keyword in self.ingredient_keywords:
                if keyword in class_lower:
                    return class_lower
            return None
        
        # For default COCO model
        if class_id in self.ingredient_classes:
            return self.ingredient_classes[class_id]
        
        # Check class name for ingredient keywords
        class_lower = class_name.lower()
        for keyword in self.ingredient_keywords:
            if keyword in class_lower:
                return class_lower
        
        return None
    
    def _detect_clusters(self, detections: list, image_shape: tuple) -> dict:
        """Detect clusters of ingredients"""
        cluster_adjustments = {}
        
        # Group by ingredient type
        by_ingredient = {}
        for det in detections:
            ingredient = det['ingredient']
            if ingredient not in by_ingredient:
                by_ingredient[ingredient] = []
            by_ingredient[ingredient].append(det)
        
        # Analyze banana clusters specifically
        if 'banana' in by_ingredient:
            bananas = by_ingredient['banana']
            if len(bananas) >= 2:
                # Check for clustering pattern
                centers = [b['center'] for b in bananas]
                distances = []
                
                for i in range(len(centers)):
                    for j in range(i + 1, len(centers)):
                        dist = np.sqrt((centers[i]['x'] - centers[j]['x'])**2 + 
                                     (centers[i]['y'] - centers[j]['y'])**2)
                        distances.append(dist)
                
                # If bananas are close together, it's likely a bunch
                avg_distance = np.mean(distances) if distances else 0
                avg_banana_width = np.mean([b['bbox']['x2'] - b['bbox']['x1'] for b in bananas])
                
                if avg_distance < avg_banana_width * 2:  # Close together
                    # Estimate hidden bananas in bunch
                    visible_count = len(bananas)
                    estimated_hidden = int(visible_count * 1.5)  # Conservative estimate
                    cluster_adjustments['banana'] = estimated_hidden
                    print(f"  🍌 Detected banana cluster: {visible_count} visible, "
                          f"estimated {estimated_hidden} additional")
        
        return cluster_adjustments
    
    def visualize_counts(self, image_path: str, results: dict, save_path: str = None):
        """Create visualization with counts"""
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Left: Image with detections
        ax1.imshow(image_rgb)
        ax1.set_title(f"Ingredient Detection ({results['model_used']} model)", 
                     fontsize=16, fontweight='bold')
        ax1.axis('off')
        
        # Color map for different ingredients
        colors = plt.cm.rainbow(np.linspace(0, 1, max(len(results['counts']), 1)))
        color_map = {}
        
        for i, (ingredient, _) in enumerate(results['counts'].items()):
            color_map[ingredient] = colors[i]
        
        # Draw detections
        for detection in results['detections']:
            bbox = detection['bbox']
            ingredient = detection['ingredient']
            
            # Get color
            color = color_map.get(ingredient, 'red')
            
            # Draw box
            rect = patches.Rectangle(
                (bbox['x1'], bbox['y1']),
                bbox['x2'] - bbox['x1'],
                bbox['y2'] - bbox['y1'],
                linewidth=3,
                edgecolor=color,
                facecolor='none'
            )
            ax1.add_patch(rect)
            
            # Add label
            label = f"{ingredient} ({detection['confidence']:.2f})"
            ax1.text(
                bbox['x1'], bbox['y1'] - 5,
                label,
                color='white',
                fontsize=10,
                weight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.8)
            )
        
        # Right: Summary
        ax2.axis('off')
        
        # Summary text
        summary_text = "📊 INGREDIENT COUNT SUMMARY\n" + "="*30 + "\n\n"
        
        if results['counts']:
            for ingredient, count in sorted(results['counts'].items()):
                emoji = {'banana': '🍌', 'apple': '🍎', 'orange': '🍊', 
                        'carrot': '🥕', 'broccoli': '🥦'}.get(ingredient, '🥗')
                summary_text += f"{emoji} {ingredient.capitalize()}: {count}\n"
                
                if results.get('cluster_info', {}).get(ingredient):
                    additional = results['cluster_info'][ingredient]
                    total = count + additional
                    summary_text += f"   📦 Cluster: ~{total} total estimated\n"
                
                summary_text += "\n"
            
            summary_text += f"\n📋 TOTAL DETECTED: {results['total_items']}"
        else:
            summary_text += "No ingredients detected\n\n"
            summary_text += "Tips:\n"
            summary_text += "• Try adjusting confidence threshold\n"
            summary_text += "• Ensure good lighting in image\n"
            summary_text += "• Use custom trained model for better results"
        
        summary_text += f"\n\n🔧 Model: {results['model_used']}"
        summary_text += f"\n💻 Device: {results['device_used']}"
        
        ax2.text(
            0.1, 0.9,
            summary_text,
            transform=ax2.transAxes,
            fontsize=14,
            verticalalignment='top',
            fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=1", facecolor='lightyellow', alpha=0.8)
        )
        
        plt.suptitle(f"Ingredient Detection - {Path(image_path).name}", 
                    fontsize=18, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ Saved visualization: {save_path}")
        
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Detect and count ingredients")
    parser.add_argument('--image', type=str, required=True, help='Path to image')
    parser.add_argument('--model', type=str, help='Path to custom YOLO model')
    parser.add_argument('--confidence', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--output-dir', type=str, default='data/output/ingredient_counts')
    parser.add_argument('--cpu', action='store_true', help='Force CPU usage')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize counter
    counter = IngredientCounter(
        model_path=args.model,
        confidence_threshold=args.confidence,
        use_cpu=args.cpu
    )
    
    # Run detection
    results = counter.detect_and_count(args.image)
    
    if results['status'] == 'success':
        # Generate outputs
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = Path(args.image).stem
        
        # Save visualization
        viz_path = output_dir / f"{base_name}_count_{timestamp}.png"
        counter.visualize_counts(args.image, results, viz_path)
        
        # Print summary
        print("\n" + "="*50)
        print("📊 RESULTS")
        print("="*50)
        
        if results['counts']:
            for ingredient, count in results['counts'].items():
                print(f"{ingredient.capitalize()}: {count}")
        else:
            print("No ingredients detected")
            
        print(f"\nTotal items: {results['total_items']}")
    else:
        print(f"❌ Error: {results.get('message', 'Unknown error')}")


if __name__ == "__main__":
    main()