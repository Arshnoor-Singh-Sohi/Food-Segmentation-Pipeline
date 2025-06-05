"""Visualization utilities for food segmentation results."""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import cv2
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

class FoodVisualization:
    """Visualization tools for food analysis results."""
    
    def __init__(self):
        """Initialize visualization tools."""
        self.colors = plt.cm.Set3(np.linspace(0, 1, 12))
        
    def visualize_meal_analysis(self, results: Dict[str, Any]):
        """Create comprehensive visualization of meal analysis."""
        # Load original image
        image_path = results['image_path']
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Could not load image: {image_path}")
            return
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        food_items = results['food_items']
        
        # Create subplot layout
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f"Food Analysis: {results['meal_summary']['total_items']} items detected", fontsize=16)
        
        # 1. Original image with bounding boxes
        self._plot_detections(axes[0, 0], image_rgb, food_items, "Detected Food Items")
        
        # 2. Segmentation masks
        self._plot_segmentation_masks(axes[0, 1], image_rgb, food_items, "Segmentation Masks")
        
        # 3. Nutrition breakdown
        self._plot_nutrition_breakdown(axes[1, 0], results['total_nutrition'])
        
        # 4. Quality and confidence scores
        self._plot_quality_scores(axes[1, 1], food_items)
        
        plt.tight_layout()
        plt.show()
    
    def _plot_detections(self, ax, image, food_items, title):
        """Plot original image with detection bounding boxes."""
        ax.imshow(image)
        ax.set_title(title)
        ax.axis('off')
        
        for i, item in enumerate(food_items):
            if 'bbox' in item:
                bbox = item['bbox']
                x1, y1, x2, y2 = bbox
                
                # Draw bounding box
                rect = patches.Rectangle(
                    (x1, y1), x2-x1, y2-y1,
                    linewidth=2, edgecolor=self.colors[i % len(self.colors)],
                    facecolor='none'
                )
                ax.add_patch(rect)
                
                # Add label
                label = f"{item['name']} ({item['detection_confidence']:.2f})"
                ax.text(x1, y1-5, label, fontsize=8, 
                       color=self.colors[i % len(self.colors)], weight='bold')
    
    def _plot_segmentation_masks(self, ax, image, food_items, title):
        """Plot segmentation masks overlaid on image."""
        ax.imshow(image)
        ax.set_title(title)
        ax.axis('off')
        
        for i, item in enumerate(food_items):
            if 'mask' in item:
                mask = item['mask']
                color = self.colors[i % len(self.colors)]
                
                # Create colored mask
                colored_mask = np.zeros((*mask.shape, 4))
                colored_mask[mask] = [*color[:3], 0.6]  # Semi-transparent
                
                ax.imshow(colored_mask)
    
    def _plot_nutrition_breakdown(self, ax, nutrition_data):
        """Plot nutrition breakdown pie chart."""
        # Calculate macronutrient calories
        protein_cal = nutrition_data.get('protein', 0) * 4
        carb_cal = nutrition_data.get('carbohydrates', 0) * 4
        fat_cal = nutrition_data.get('fat', 0) * 9
        
        total_macro_cal = protein_cal + carb_cal + fat_cal
        
        if total_macro_cal > 0:
            sizes = [protein_cal, carb_cal, fat_cal]
            labels = [f'Protein\n{protein_cal:.0f} cal', 
                     f'Carbs\n{carb_cal:.0f} cal', 
                     f'Fat\n{fat_cal:.0f} cal']
            colors = ['lightcoral', 'lightskyblue', 'lightgreen']
            
            ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            ax.set_title(f'Macronutrient Breakdown\nTotal: {nutrition_data.get("calories", 0):.0f} calories')
        else:
            ax.text(0.5, 0.5, 'No nutrition data available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Nutrition Breakdown')
    
    def _plot_quality_scores(self, ax, food_items):
        """Plot quality and confidence scores."""
        if not food_items:
            ax.text(0.5, 0.5, 'No food items detected', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Quality Scores')
            return
        
        names = [item['name'][:15] for item in food_items]  # Truncate long names
        detection_scores = [item.get('detection_confidence', 0) for item in food_items]
        segmentation_scores = [item.get('segmentation_confidence', 0) for item in food_items]
        
        x = np.arange(len(names))
        width = 0.35
        
        ax.bar(x - width/2, detection_scores, width, label='Detection', alpha=0.8)
        ax.bar(x + width/2, segmentation_scores, width, label='Segmentation', alpha=0.8)
        
        ax.set_xlabel('Food Items')
        ax.set_ylabel('Confidence Score')
        ax.set_title('Detection vs Segmentation Confidence')
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)