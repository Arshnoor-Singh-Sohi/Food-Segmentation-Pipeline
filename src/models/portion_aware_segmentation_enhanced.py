# Save as: src/models/portion_aware_segmentation_enhanced.py
"""
Enhanced Portion-Aware Segmentation System
Fixes missing items and adds visual output capabilities
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import logging
from datetime import datetime
import json

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
                           image: np.ndarray,
                           save_visualization: bool = True,
                           output_dir: str = "data/output") -> Dict[str, Any]:
        """
        Process segmentation based on food type classification
        
        Args:
            detection_results: Results from YOLO detection
            image: Original image array
            save_visualization: Whether to save visual output
            output_dir: Directory for output files
            
        Returns:
            Enhanced results with proper segmentation and units
        """
        # Extract ALL detected items (fixing the missing item issue)
        all_detections = []
        
        # Handle different possible result formats from YOLO
        if 'food_items' in detection_results:
            all_detections = detection_results['food_items']
        elif 'detections' in detection_results:
            all_detections = detection_results['detections']
        elif isinstance(detection_results, list):
            all_detections = detection_results
            
        # Log the actual number of detections
        logger.info(f"Processing {len(all_detections)} total detections")
        
        # Filter only food items
        food_items = []
        for item in all_detections:
            # Check if it's a food item (handle different formats)
            if item.get('is_food', True):  # Default to True if not specified
                food_items.append(item)
                
        logger.info(f"Found {len(food_items)} food items to segment")
        
        # Extract food names
        food_names = []
        for item in food_items:
            name = item.get('name') or item.get('class_name') or 'unknown'
            food_names.append(name)
            
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
        
        # Add detection vs segmentation count for debugging
        segmented_results['processing_stats'] = {
            'total_detections': len(all_detections),
            'food_items_found': len(food_items),
            'segments_created': len(segmented_results['segments'])
        }
        
        # Save visualization if requested
        if save_visualization:
            visual_path = self._create_visualization(
                image, segmented_results, output_dir
            )
            segmented_results['visualization_path'] = str(visual_path)
        
        return segmented_results
    
    def _process_individual_items(self, 
                                 food_items: List[Dict], 
                                 image: np.ndarray) -> Dict[str, Any]:
        """
        Process individual items - keep separate segments with appropriate units
        FIXED: Ensures all items are processed
        """
        segments = []
        measurement_summary = {}
        
        # Process EACH item individually first (don't group yet)
        segment_id = 0
        item_tracker = {}  # Track items by name for grouping
        
        for idx, item in enumerate(food_items):
            # Extract food name
            food_name = item.get('name') or item.get('class_name') or f'item_{idx}'
            
            # Get appropriate unit
            unit, unit_full = self.measurement_system.get_unit_for_food(
                food_name, 'individual_items'
            )
            
            # Extract bounding box
            bbox = self._extract_bbox(item)
            
            # Create segment for this item
            segment = {
                'id': f'item_{segment_id}',
                'name': food_name,
                'type': 'individual_item',
                'bbox': bbox,
                'measurement': {
                    'value': 1,
                    'unit': unit,
                    'unit_full': unit_full,
                    'formatted': self.measurement_system.format_measurement(1, unit)
                },
                'confidence': item.get('confidence', 0.5),
                'original_index': idx  # Track original detection index
            }
            
            # Add mask info if available
            if 'mask' in item:
                segment['mask'] = item['mask']
            elif 'segmentation' in item:
                segment['mask'] = item['segmentation']
            
            segments.append(segment)
            segment_id += 1
            
            # Track for summary
            if food_name not in item_tracker:
                item_tracker[food_name] = []
            item_tracker[food_name].append(segment)
        
        # Create measurement summary
        for food_name, item_segments in item_tracker.items():
            count = len(item_segments)
            # Get unit from first item
            unit = item_segments[0]['measurement']['unit']
            
            measurement_summary[food_name] = {
                'count': count,
                'unit': unit,
                'formatted': self.measurement_system.format_measurement(count, unit)
            }
        
        logger.info(f"Created {len(segments)} segments from {len(food_items)} food items")
        
        return {
            'segments': segments,
            'measurement_summary': measurement_summary
        }
    
    def _extract_bbox(self, item: Dict) -> Dict[str, int]:
        """
        Extract bounding box from various possible formats
        """
        # Try different bbox formats
        if 'bbox' in item and isinstance(item['bbox'], dict):
            return item['bbox']
        elif 'bbox' in item and isinstance(item['bbox'], (list, tuple)):
            # Convert [x, y, w, h] or [x1, y1, x2, y2] to dict
            bbox = item['bbox']
            if len(bbox) == 4:
                # Check if it's xywh or xyxy format
                if bbox[2] > 100 and bbox[3] > 100:  # Likely x1y1x2y2
                    return {
                        'x1': int(bbox[0]), 'y1': int(bbox[1]),
                        'x2': int(bbox[2]), 'y2': int(bbox[3])
                    }
                else:  # Likely xywh
                    return {
                        'x1': int(bbox[0]), 'y1': int(bbox[1]),
                        'x2': int(bbox[0] + bbox[2]), 'y2': int(bbox[1] + bbox[3])
                    }
        elif 'box' in item:
            # Handle 'box' key
            return self._extract_bbox({'bbox': item['box']})
        else:
            # Default bbox if none found
            logger.warning(f"No bbox found for item: {item.get('name', 'unknown')}")
            return {'x1': 0, 'y1': 0, 'x2': 100, 'y2': 100}
    
    def _create_visualization(self, 
                            image: np.ndarray, 
                            results: Dict,
                            output_dir: str) -> Path:
        """
        Create visualization with bounding boxes and labels
        This is what your CEO requested!
        """
        # Create a copy for visualization
        vis_image = image.copy()
        
        # Define colors for different types
        colors = {
            'complete_dish': (255, 100, 0),  # Orange for dishes
            'individual_item': (0, 255, 0),   # Green for individual items
        }
        
        # Font settings
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 2
        
        # Draw each segment
        for segment in results['segments']:
            bbox = segment['bbox']
            seg_type = segment['type']
            color = colors.get(seg_type, (255, 255, 255))
            
            # Draw bounding box
            cv2.rectangle(
                vis_image,
                (bbox['x1'], bbox['y1']),
                (bbox['x2'], bbox['y2']),
                color, 2
            )
            
            # Prepare label with measurement
            label = f"{segment['name']}: {segment['measurement']['formatted']}"
            
            # Get text size for background
            (text_width, text_height), baseline = cv2.getTextSize(
                label, font, font_scale, font_thickness
            )
            
            # Draw background for text
            cv2.rectangle(
                vis_image,
                (bbox['x1'], bbox['y1'] - text_height - 10),
                (bbox['x1'] + text_width + 10, bbox['y1']),
                color, -1
            )
            
            # Draw text
            cv2.putText(
                vis_image,
                label,
                (bbox['x1'] + 5, bbox['y1'] - 5),
                font, font_scale, (255, 255, 255), font_thickness
            )
        
        # Add summary information
        summary_y = 30
        summary_bg_height = 100
        cv2.rectangle(
            vis_image,
            (10, 10),
            (500, summary_bg_height),
            (0, 0, 0), -1
        )
        
        # Add classification info
        cv2.putText(
            vis_image,
            f"Type: {results['food_type_classification']['type']}",
            (20, summary_y),
            font, 0.7, (255, 255, 255), 2
        )
        
        cv2.putText(
            vis_image,
            f"Total Segments: {len(results['segments'])}",
            (20, summary_y + 30),
            font, 0.7, (255, 255, 255), 2
        )
        
        # Generate unique filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(output_dir) / f"segmentation_visual_{timestamp}.jpg"
        output_path.parent.mkdir(exist_ok=True)
        
        # Save visualization
        cv2.imwrite(str(output_path), vis_image)
        logger.info(f"Saved visualization to: {output_path}")
        
        return output_path
    
    def _determine_dish_name(self, food_items: List[Dict]) -> str:
        """
        Determine the best name for a complete dish based on detected components
        """
        if not food_items:
            return "unknown dish"
        
        # Use the most confident detection
        most_confident = max(food_items, key=lambda x: x.get('confidence', 0))
        base_name = most_confident.get('name') or most_confident.get('class_name', 'unknown')
        
        # Check if we can identify a specific dish type
        item_names = [
            (item.get('name') or item.get('class_name', '')).lower() 
            for item in food_items
        ]
        
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
    
    def _process_complete_dish(self, 
                              food_items: List[Dict], 
                              image: np.ndarray) -> Dict[str, Any]:
        """
        Process complete dish - merge all segments into one portion
        """
        if not food_items:
            return {'segments': [], 'measurement_summary': {}}
        
        # Find the overall bounding box that encompasses all items
        min_x, min_y = float('inf'), float('inf')
        max_x, max_y = 0, 0
        
        for item in food_items:
            bbox = self._extract_bbox(item)
            min_x = min(min_x, bbox['x1'])
            min_y = min(min_y, bbox['y1'])
            max_x = max(max_x, bbox['x2'])
            max_y = max(max_y, bbox['y2'])
        
        # Determine dish name
        dish_name = self._determine_dish_name(food_items)
        
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
            'components': food_items,  # Keep track of detected components
            'confidence': np.mean([item.get('confidence', 0.5) for item in food_items])
        }
        
        return {
            'segments': [unified_segment],
            'measurement_summary': {
                'total_portions': 1,
                'dish_type': dish_name,
                'components_detected': len(food_items)
            }
        }