"""
Stage 1A: Detection Fixer
=========================

Place this file in: stages/stage1a_detection_fixes/detection_fixer.py

Focuses specifically on Stage 1A objectives:
1. Fix false positive bottle detection
2. Fix banana quantity counting (no more "whole on 3")  
3. Improve item classification accuracy
4. Fix portion vs complete dish display
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import json

try:
    from ultralytics import YOLO
except ImportError:
    print("⚠️ Ultralytics not found. Install with: pip install ultralytics")

class Stage1ADetectionFixer:
    """
    Stage 1A Detection Fixer - Addresses specific detection issues
    """
    
    def __init__(self):
        self.model = None
        self.load_best_model()
        
        # Stage 1A: Enhanced confidence thresholds to reduce false positives
        self.confidence_thresholds = {
            'bottle': 0.65,     # Higher threshold - CEO feedback: too many bottles
            'banana': 0.35,     # Lower threshold - better detection of individual bananas
            'apple': 0.4,
            'food': 0.3,
            'pizza': 0.5,
            'default': 0.25
        }
        
        # Stage 1A: Bottle validation parameters (main issue)
        self.bottle_validation = {
            'min_area_pixels': 800,      # Minimum size for real bottles
            'max_area_pixels': 40000,    # Maximum size for real bottles
            'min_aspect_ratio': 0.15,    # Bottles are typically tall
            'max_aspect_ratio': 1.0,     # Bottles shouldn't be too wide
            'min_confidence': 0.65       # High confidence required for bottles
        }
        
        # Stage 1A: Complete dishes vs individual items classification
        self.complete_dishes = {
            'pizza', 'pasta', 'salad', 'burger', 'sandwich', 'soup',
            'curry', 'stew', 'lasagna', 'casserole', 'stir fry', 'omelet'
        }
        
        self.storage_indicators = {
            'bottle', 'container', 'jar', 'package', 'carton', 'box',
            'refrigerator', 'fridge', 'shelf'
        }

    def load_best_model(self):
        """Load the best available model for detection"""
        model_options = [
            'data/models/custom_food_detection.pt',  # Custom 99.5% model first
            'yolov8n.pt',
            'yolov8s.pt'
        ]
        
        for model_path in model_options:
            try:
                self.model = YOLO(model_path)
                print(f"✅ Loaded model: {model_path}")
                return
            except Exception as e:
                continue
        
        raise Exception("❌ No YOLO model could be loaded")

    def analyze_image(self, image_path):
        """
        Stage 1A: Main analysis function that fixes the specific issues
        """
        print(f"🔍 Stage 1A Analysis: {Path(image_path).name}")
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Could not load image: {image_path}")
            return None
        
        # Step 1: Run YOLO detection
        results = self.model(image)
        
        # Step 2: Extract raw detections
        raw_detections = self.extract_raw_detections(results)
        print(f"📊 Raw YOLO detections: {len(raw_detections)}")
        
        # Step 3: Apply Stage 1A fixes
        enhanced_detections = self.apply_stage1a_fixes(raw_detections, image.shape)
        print(f"📊 Enhanced detections: {len(enhanced_detections)}")
        
        # Step 4: Count items intelligently (banana cluster fix)
        item_counts = self.count_items_with_cluster_analysis(enhanced_detections)
        
        # Step 5: Classify food context (portion vs individual fix)
        food_classification = self.classify_food_context(enhanced_detections)
        
        return {
            'image_path': image_path,
            'raw_detections': raw_detections,
            'enhanced_detections': enhanced_detections,
            'item_counts': item_counts,
            'food_type_classification': food_classification,
            'stage1a_fixes_applied': {
                'bottle_validation': True,
                'banana_cluster_analysis': True,
                'enhanced_confidence_thresholds': True,
                'food_context_classification': True
            }
        }

    def extract_raw_detections(self, yolo_results):
        """Extract all raw detections from YOLO results"""
        raw_detections = []
        
        for result in yolo_results:
            if result.boxes is not None:
                for box in result.boxes:
                    detection = {
                        'class_name': self.model.names[int(box.cls)],
                        'confidence': float(box.conf),
                        'bbox': box.xyxy[0].cpu().numpy().tolist()
                    }
                    raw_detections.append(detection)
        
        return raw_detections

    def apply_stage1a_fixes(self, raw_detections, image_shape):
        """
        Apply Stage 1A specific fixes to reduce false positives
        """
        enhanced_detections = []
        
        for detection in raw_detections:
            class_name = detection['class_name']
            confidence = detection['confidence']
            bbox = detection['bbox']
            
            # Stage 1A Fix 1: Enhanced confidence thresholds
            threshold = self.confidence_thresholds.get(class_name, 
                                                     self.confidence_thresholds['default'])
            if confidence < threshold:
                continue
            
            # Stage 1A Fix 2: Special bottle validation (main issue)
            if class_name == 'bottle':
                if not self.validate_bottle_detection(bbox, confidence):
                    continue  # Filter out false positive bottles
            
            # Stage 1A Fix 3: General size validation
            if not self.validate_detection_size(bbox, image_shape):
                continue
            
            enhanced_detections.append(detection)
        
        return enhanced_detections

    def validate_bottle_detection(self, bbox, confidence):
        """
        Stage 1A: Enhanced bottle validation to fix false positives
        CEO feedback: "too many bottles count"
        """
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        area = width * height
        aspect_ratio = width / height if height > 0 else 0
        
        # Rule 1: Size constraints
        if area < self.bottle_validation['min_area_pixels']:
            return False  # Too small to be a real bottle
        if area > self.bottle_validation['max_area_pixels']:
            return False  # Too large to be a real bottle
        
        # Rule 2: Shape constraints (bottles are typically tall)
        if aspect_ratio < self.bottle_validation['min_aspect_ratio']:
            return False  # Too thin
        if aspect_ratio > self.bottle_validation['max_aspect_ratio']:
            return False  # Too wide
        
        # Rule 3: Confidence requirement
        if confidence < self.bottle_validation['min_confidence']:
            return False  # Low confidence bottles likely false positives
        
        return True

    def validate_detection_size(self, bbox, image_shape):
        """General size validation for all detections"""
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        area = width * height
        
        # Image area
        image_area = image_shape[0] * image_shape[1]
        relative_size = area / image_area
        
        # Filter out very small detections (noise) and very large ones (background)
        if relative_size < 0.0005:  # Too small
            return False
        if relative_size > 0.8:     # Too large (probably background)
            return False
        
        return True

    def count_items_with_cluster_analysis(self, detections):
        """
        Stage 1A: Fix banana counting with cluster analysis
        CEO feedback: banana quantity showing "whole on 3" instead of individual count
        """
        item_counts = {}
        
        # Group detections by class
        by_class = {}
        for detection in detections:
            class_name = detection['class_name']
            if class_name not in by_class:
                by_class[class_name] = []
            by_class[class_name].append(detection)
        
        # Count each class with special handling
        for class_name, class_detections in by_class.items():
            detected_objects = len(class_detections)
            
            if class_name == 'banana':
                # Stage 1A: Special banana cluster analysis
                estimated_total = self.analyze_banana_cluster(class_detections)
                notes = f"Cluster analysis: {detected_objects} detected → {estimated_total} estimated bananas"
            else:
                estimated_total = detected_objects
                notes = f"Direct count: {detected_objects} objects"
            
            item_counts[class_name] = {
                'detected_objects': detected_objects,
                'estimated_total': estimated_total,
                'notes': notes
            }
        
        return item_counts

    def analyze_banana_cluster(self, banana_detections):
        """
        Stage 1A: Banana cluster analysis to fix counting
        """
        if not banana_detections:
            return 0
        
        total_estimated = 0
        
        for detection in banana_detections:
            bbox = detection['bbox']
            x1, y1, x2, y2 = bbox
            width = x2 - x1
            height = y2 - y1
            area = width * height
            
            # Estimate number of bananas based on detection size
            if area > 8000:  # Large detection suggests cluster
                estimated_in_cluster = min(int(area / 2500), 6)  # Max 6 per cluster
                total_estimated += max(estimated_in_cluster, 1)
            elif area > 4000:  # Medium detection might be 2-3 bananas
                total_estimated += 2
            else:  # Small detection is probably 1 banana
                total_estimated += 1
        
        return total_estimated

    def classify_food_context(self, detections):
        """
        Stage 1A: Fix portion vs complete dish classification display
        """
        if not detections:
            return {
                'type': 'unknown',
                'confidence': 0.0,
                'explanation': 'No detections found',
                'display_as': 'UNKNOWN'
            }
        
        food_names = [d['class_name'] for d in detections]
        detection_count = len(detections)
        
        # Check for storage context (refrigerator, multiple bottles, etc.)
        storage_score = sum(1 for name in food_names 
                          if any(indicator in name.lower() 
                               for indicator in self.storage_indicators))
        
        # Check for complete dish indicators
        dish_score = sum(1 for name in food_names if name in self.complete_dishes)
        
        # Stage 1A: Enhanced classification logic
        if storage_score > 0 or detection_count > 5:
            # Storage context or many items = individual items
            return {
                'type': 'individual_items',
                'confidence': 1.0,
                'explanation': f'Storage context with {detection_count} items',
                'display_as': 'INDIVIDUAL ITEMS'
            }
        elif dish_score > 0 and detection_count <= 3:
            # Clear dish indicators with few items = complete dish
            return {
                'type': 'complete_dish', 
                'confidence': 0.9,
                'explanation': f'Complete dish detected ({dish_score} dish indicators)',
                'display_as': 'COMPLETE DISH'
            }
        else:
            # Default to individual items for ambiguous cases
            return {
                'type': 'individual_items',
                'confidence': 0.7,
                'explanation': f'Ambiguous context, defaulting to individual items',
                'display_as': 'INDIVIDUAL ITEMS'
            }

    def create_visualization(self, image_path, results, output_dir, timestamp):
        """Create Stage 1A visualization showing before/after fixes"""
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create side-by-side comparison
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # Left: Raw detections
        ax1.imshow(image_rgb)
        ax1.set_title('Before Stage 1A Fixes (Raw YOLO)', fontsize=16, fontweight='bold')
        
        for detection in results['raw_detections']:
            bbox = detection['bbox']
            x1, y1, x2, y2 = bbox
            
            # Red boxes for raw detections
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                               fill=False, color='red', linewidth=2, alpha=0.7)
            ax1.add_patch(rect)
            
            label = f"{detection['class_name']}: {detection['confidence']:.2f}"
            ax1.text(x1, y1-5, label, fontsize=9, color='red',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        ax1.set_xlabel(f"Raw Detections: {len(results['raw_detections'])}", fontsize=12)
        ax1.axis('off')
        
        # Right: Enhanced detections
        ax2.imshow(image_rgb)
        ax2.set_title('After Stage 1A Fixes (Enhanced)', fontsize=16, fontweight='bold')
        
        for detection in results['enhanced_detections']:
            bbox = detection['bbox']
            x1, y1, x2, y2 = bbox
            class_name = detection['class_name']
            
            # Green boxes for enhanced detections
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                               fill=False, color='green', linewidth=2, alpha=0.8)
            ax2.add_patch(rect)
            
            # Add count information
            item_info = results['item_counts'].get(class_name, {})
            estimated = item_info.get('estimated_total', 1)
            
            label = f"{class_name}: {detection['confidence']:.2f}\nEst: {estimated}"
            ax2.text(x1, y1-10, label, fontsize=9, color='green',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        ax2.set_xlabel(f"Enhanced Detections: {len(results['enhanced_detections'])}", 
                      fontsize=12)
        ax2.axis('off')
        
        # Add title with classification
        food_type = results['food_type_classification']
        fig.suptitle(f"Stage 1A Fixes Applied - {food_type['display_as']} "
                    f"(Confidence: {food_type['confidence']:.1%})", 
                    fontsize=18, fontweight='bold')
        
        plt.tight_layout()
        
        # Save visualization
        image_name = Path(image_path).stem
        viz_file = output_dir / f"{image_name}_stage1a_fixes_{timestamp}.png"
        plt.savefig(viz_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        return viz_file

# Test function for standalone execution
def test_stage1a():
    """Test function for Stage 1A detection fixer"""
    print("🧪 Testing Stage 1A Detection Fixer")
    
    fixer = Stage1ADetectionFixer()
    
    # Test with a sample image (adjust path as needed)
    test_image = "data/input/refrigerator.jpg"
    
    if Path(test_image).exists():
        results = fixer.analyze_image(test_image)
        if results:
            print("✅ Stage 1A test successful")
            return True
    
    print("⚠️ Test image not found or analysis failed")
    return False

if __name__ == "__main__":
    test_stage1a()