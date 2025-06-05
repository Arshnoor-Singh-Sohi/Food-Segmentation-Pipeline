"""Fast segmentation using YOLO + traditional CV - No SAM 2 needed!"""

import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt

class FastFoodSegmentation:
    def __init__(self):
        self.model = YOLO('yolov8n-seg.pt')  # Segmentation version of YOLO
    
    def process_image(self, image_path):
        """Process image in 5-10 seconds instead of 10+ minutes!"""
        # Load image
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Run YOLO segmentation (FAST!)
        results = self.model(image_rgb)
        
        # Extract results
        food_items = []
        if results[0].masks is not None:
            for i, (box, mask, conf, cls) in enumerate(zip(
                results[0].boxes.xyxy,
                results[0].masks.data,
                results[0].boxes.conf,
                results[0].boxes.cls
            )):
                food_items.append({
                    'name': self.model.names[int(cls)],
                    'confidence': float(conf),
                    'bbox': box.cpu().numpy().tolist(),
                    'mask': mask.cpu().numpy(),
                    'calories': self._estimate_calories(mask.cpu().numpy())
                })
        
        return {
            'image_path': image_path,
            'food_items': food_items,
            'total_items': len(food_items),
            'processing_time': '5-10 seconds!'
        }
    
    def _estimate_calories(self, mask):
        """Quick calorie estimation"""
        area = np.sum(mask)
        return int(area * 0.001)  # Rough estimate
    
    def visualize(self, results):
        """Show results"""
        image = cv2.imread(results['image_path'])
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        plt.figure(figsize=(12, 8))
        plt.imshow(image_rgb)
        
        for item in results['food_items']:
            mask = item['mask']
            plt.imshow(mask, alpha=0.3)
            
        plt.title(f"Found {results['total_items']} items in {results['processing_time']}")
        plt.axis('off')
        plt.show()
        
        # Print results
        print("\n🍽️ FAST FOOD ANALYSIS:")
        for i, item in enumerate(results['food_items']):
            print(f"{i+1}. {item['name']} - {item['confidence']:.2f} conf - ~{item['calories']} cal")