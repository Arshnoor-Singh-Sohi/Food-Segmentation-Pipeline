
"""
Portion-Aware Segmentation System
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class PortionAwareSegmentation:
    """
    Implements intelligent segmentation based on food type:
    - Complete dishes: One unified segment
    - Individual items: Separate segments for each item
    """
    
    def __init__(self, food_classifier, measurement_system):
        """
        Initialize with food type classifier and measurement system
        
        Args:
            food_classifier: Instance of FoodTypeClassifier
            measurement_system: Instance of MeasurementUnitSystem
        """
        self.food_classifier = food_classifier
        self.measurement_system = measurement_system
        
    def process_segmentation(self, 
                           detection_results: Dict,
                           image: np.ndarray) -> Dict[str, Any]:
        """
        Process segmentation based on food type classification
        
        Args:
            detection_results: Results from YOLO detection
            image: Original image array
            
        Returns:
            Enhanced results with proper segmentation and units
        """
        # Extract food names and count
        food_items = detection_results.get('food_items', [])
        food_names = [item['name'] for item in food_items if item.get('is_food', False)]
        detection_count = len(food_names)
        
        # Classify food type
        food_type, confidence = self.food_classifier.classify_food_type(
            food_names, detection_count
        )
        
        logger.info(f"Food type classification: {food_type} (confidence: {confidence:.2%})")
        
        # Process based on classification
        if food_type == 'complete_dish':
            segmented_results = self._process_complete_dish(food_items, image)
        else:
            segmented_results = self._process_individual_items(food_items, image)
        
        # Add classification metadata
        segmented_results['food_type_classification'] = {
            'type': food_type,
            'confidence': confidence,
            'explanation': self.food_classifier.get_classification_explanation(food_type, confidence)
        }
        
        return segmented_results
    
    def _process_complete_dish(self, 
                              food_items: List[Dict], 
                              image: np.ndarray) -> Dict[str, Any]:
        """
        Process complete dish - merge all segments into one portion
        """
        if not food_items:
            return {'segments': [], 'measurement_summary': {}}
        
        # Merge all masks and bounding boxes
        merged_mask = np.zeros((image.shape[0], image.shape[1]), dtype=bool)
        min_x, min_y = float('inf'), float('inf')
        max_x, max_y = 0, 0
        
        # Collect all food items (ignore non-food)
        food_only = [item for item in food_items if item.get('is_food', False)]
        
        for item in food_only:
            # Merge masks if available
            if 'mask_info' in item or 'mask' in item:
                # Add mask to merged mask
                # (Implementation depends on your mask format)
                pass
            
            # Expand bounding box
            bbox = item['bbox']
            min_x = min(min_x, bbox['x1'])
            min_y = min(min_y, bbox['y1'])
            max_x = max(max_x, bbox['x2'])
            max_y = max(max_y, bbox['y2'])
        
        # Determine dish name (use most confident detection or aggregate)
        dish_name = self._determine_dish_name(food_only)
        
        # Create single unified segment
        unified_segment = {
            'id': 'dish_0',
            'name': dish_name,
            'type': 'complete_dish',
            'bbox': {
                'x1': int(min_x), 'y1': int(min_y),
                'x2': int(max_x), 'y2': int(max_y)
            },
            'measurement': {
                'value': 1,
                'unit': 'portion',
                'unit_full': 'portion',
                'formatted': '1 portion'
            },
            'components': food_only,  # Keep track of detected components
            'confidence': np.mean([item['confidence'] for item in food_only])
        }
        
        return {
            'segments': [unified_segment],
            'measurement_summary': {
                'total_portions': 1,
                'dish_type': dish_name,
                'components_detected': len(food_only)
            }
        }
    
    def _process_individual_items(self, 
                                 food_items: List[Dict], 
                                 image: np.ndarray) -> Dict[str, Any]:
        """
        Process individual items - keep separate segments with appropriate units
        """
        segments = []
        measurement_summary = {}
        
        # Group items by type for counting
        item_groups = {}
        
        for item in food_items:
            if not item.get('is_food', False):
                continue
            
            food_name = item['name']
            
            # Group similar items
            if food_name not in item_groups:
                item_groups[food_name] = []
            item_groups[food_name].append(item)
        
        # Process each group
        segment_id = 0
        for food_name, items in item_groups.items():
            # Get appropriate unit
            unit, unit_full = self.measurement_system.get_unit_for_food(
                food_name, 'individual_items'
            )
            
            # For countable items, create separate segments
            if unit in ['piece', 'unit', 'item', 'slice']:
                for idx, item in enumerate(items):
                    segment = {
                        'id': f'item_{segment_id}',
                        'name': food_name,
                        'type': 'individual_item',
                        'bbox': item['bbox'],
                        'measurement': {
                            'value': 1,
                            'unit': unit,
                            'unit_full': unit_full,
                            'formatted': self.measurement_system.format_measurement(1, unit)
                        },
                        'group_id': food_name,
                        'confidence': item['confidence']
                    }
                    
                    # Add mask info if available
                    if 'mask' in item:
                        segment['mask'] = item['mask']
                    
                    segments.append(segment)
                    segment_id += 1
                
                # Update summary
                measurement_summary[food_name] = {
                    'count': len(items),
                    'unit': unit,
                    'formatted': self.measurement_system.format_measurement(len(items), unit)
                }
            
            else:
                # For weight/volume based items, estimate quantity
                for item in items:
                    # Estimate weight/volume based on size
                    estimated_value = self._estimate_quantity(item, unit, image.shape)
                    
                    segment = {
                        'id': f'item_{segment_id}',
                        'name': food_name,
                        'type': 'individual_item',
                        'bbox': item['bbox'],
                        'measurement': {
                            'value': estimated_value,
                            'unit': unit,
                            'unit_full': unit_full,
                            'formatted': self.measurement_system.format_measurement(
                                estimated_value, unit
                            )
                        },
                        'confidence': item['confidence']
                    }
                    
                    segments.append(segment)
                    segment_id += 1
                
                # Sum up quantities for summary
                total_quantity = sum(self._estimate_quantity(item, unit, image.shape) 
                                   for item in items)
                measurement_summary[food_name] = {
                    'total': total_quantity,
                    'unit': unit,
                    'formatted': self.measurement_system.format_measurement(total_quantity, unit)
                }
        
        return {
            'segments': segments,
            'measurement_summary': measurement_summary
        }
    
    def _determine_dish_name(self, food_items: List[Dict]) -> str:
        """
        Determine the best name for a complete dish based on detected components
        """
        if not food_items:
            return "unknown dish"
        
        # Use the most confident detection
        most_confident = max(food_items, key=lambda x: x.get('confidence', 0))
        base_name = most_confident['name']
        
        # Check if we can identify a specific dish type
        item_names = [item['name'].lower() for item in food_items]
        
        # Pattern matching for common dishes
        if 'pizza' in base_name.lower():
            return base_name
        elif any('salad' in name for name in item_names):
            if 'chicken' in ' '.join(item_names):
                return "chicken salad"
            elif 'caesar' in ' '.join(item_names):
                return "caesar salad"
            else:
                return "mixed salad"
        elif 'burger' in base_name.lower() or 'hamburger' in base_name.lower():
            return "hamburger meal"
        else:
            # Generic naming based on components
            return f"{base_name} meal"
    
    def _estimate_quantity(self, item: Dict, unit: str, image_shape: Tuple) -> float:
        """
        Estimate quantity based on item size and unit type
        """
        # Get area from mask or bbox
        if 'mask_info' in item:
            area_pixels = item['mask_info'].get('area_pixels', 0)
        else:
            bbox = item['bbox']
            area_pixels = (bbox['x2'] - bbox['x1']) * (bbox['y2'] - bbox['y1'])
        
        # Convert to percentage of image
        total_pixels = image_shape[0] * image_shape[1]
        area_percentage = (area_pixels / total_pixels) * 100
        
        # Estimate based on unit type
        if unit == 'g':
            # Rough estimation: assume typical food photo captures ~500g total
            return area_percentage * 5  # 5g per 1% of image
        elif unit == 'ml':
            # For liquids, assume typical container sizes
            if area_percentage > 10:
                return 250  # Large container
            elif area_percentage > 5:
                return 150  # Medium container
            else:
                return 100  # Small container
        else:
            # Default estimation
            return area_percentage