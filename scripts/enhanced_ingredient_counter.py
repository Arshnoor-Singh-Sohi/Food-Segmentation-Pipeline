#!/usr/bin/env python3
"""
Enhanced ingredient counter that handles both individual and grouped detections
"""

import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import matplotlib.pyplot as plt

class EnhancedIngredientCounter:
    """Count ingredients even when model detects groups"""
    
    def __init__(self, model_path: str = None):
        if model_path and Path(model_path).exists():
            self.model = YOLO(model_path)
            self.model_type = "custom"
        else:
            self.model = YOLO('yolov8n.pt')
            self.model_type = "default"
    
    def count_with_splitting(self, image_path: str):
        """Count ingredients with group splitting"""
        
        # Get detections
        results = self.model.predict(image_path, conf=0.25, device='cpu')
        
        if not results:
            return {'counts': {}, 'total': 0}
        
        result = results[0]
        image = cv2.imread(image_path)
        
        counts = {}
        
        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls)
                class_name = result.names.get(class_id, "unknown")
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                # Extract region
                roi = image[int(y1):int(y2), int(x1):int(x2)]
                
                # Estimate count based on detection
                if 'banana' in class_name.lower():
                    # For bananas, estimate based on size and shape
                    estimated_count = self._estimate_banana_count(roi, (x2-x1, y2-y1))
                    counts['banana'] = counts.get('banana', 0) + estimated_count
                elif any(fruit in class_name.lower() for fruit in ['apple', 'orange']):
                    # For round fruits, use circular detection
                    estimated_count = self._estimate_round_fruit_count(roi)
                    counts[class_name.lower()] = counts.get(class_name.lower(), 0) + estimated_count
                else:
                    # Default: count as 1
                    counts[class_name.lower()] = counts.get(class_name.lower(), 0) + 1
        
        return {
            'counts': counts,
            'total': sum(counts.values()),
            'method': 'enhanced_splitting'
        }
    
    def _estimate_banana_count(self, roi, bbox_size):
        """Estimate number of bananas in a detection"""
        width, height = bbox_size
        
        # Typical single banana aspect ratio
        single_banana_aspect = 4.0  # length/width
        
        # Calculate aspect ratio of detection
        aspect_ratio = max(width, height) / min(width, height)
        
        if aspect_ratio > single_banana_aspect * 1.5:
            # Likely multiple bananas side by side
            estimated = int(aspect_ratio / single_banana_aspect)
        else:
            # Use area-based estimation
            single_banana_area = 5000  # pixels, adjust based on image resolution
            detection_area = width * height
            estimated = max(1, int(detection_area / single_banana_area))
        
        # Apply edge detection to refine
        edges = cv2.Canny(roi, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # If we can see distinct contours, use that
        significant_contours = [c for c in contours if cv2.contourArea(c) > 1000]
        if len(significant_contours) > 1:
            estimated = max(estimated, len(significant_contours))
        
        return min(estimated, 6)  # Cap at 6 for bunches
    
    def _estimate_round_fruit_count(self, roi):
        """Estimate count of round fruits using circular detection"""
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Detect circles
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=30,
            param1=50,
            param2=30,
            minRadius=20,
            maxRadius=100
        )
        
        if circles is not None:
            return len(circles[0])
        
        return 1  # Default to 1 if no circles detected
    
    def visualize_enhanced_counting(self, image_path: str, save_path: str = None):
        """Visualize the enhanced counting results"""
        results = self.count_with_splitting(image_path)
        
        # Create visualization
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        plt.figure(figsize=(12, 8))
        plt.imshow(image_rgb)
        plt.title(f"Enhanced Counting ({self.model_type} model)", fontsize=16)
        
        # Add count annotation
        count_text = "Counts:\n"
        for item, count in results['counts'].items():
            count_text += f"{item}: {count}\n"
        count_text += f"\nTotal: {results['total']}"
        
        plt.text(10, 50, count_text, fontsize=12, 
                bbox=dict(boxstyle="round,pad=0.5", facecolor='yellow', alpha=0.7))
        
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        plt.close()
        
        return results


# Quick test
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        counter = EnhancedIngredientCounter(
            model_path="data/models/custom_food_detection.pt" 
            if Path("data/models/custom_food_detection.pt").exists() else None
        )
        
        results = counter.count_with_splitting(sys.argv[1])
        print(f"\nEnhanced counting results:")
        for item, count in results['counts'].items():
            print(f"  {item}: {count}")
        print(f"Total: {results['total']}")
        
        counter.visualize_enhanced_counting(sys.argv[1], "enhanced_count_result.png")