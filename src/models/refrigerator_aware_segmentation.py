# Save as: src/models/refrigerator_aware_segmentation.py
"""
Refrigerator-Aware Segmentation System
Fixes the critical issues with food detection in storage contexts
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import logging
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class RefrigeratorAwareSegmentation:
    """
    Enhanced segmentation system that properly handles refrigerators and storage contexts
    Key fixes:
    1. Doesn't filter out food items aggressively
    2. Properly identifies storage contexts
    3. Assigns appropriate units to different food types
    """
    
    def __init__(self, food_classifier, measurement_system):
        self.food_classifier = food_classifier
        self.measurement_system = measurement_system
        
        # Expanded list of food-related classes that YOLO might detect
        self.food_related_classes = {
            # Direct food items
            'banana', 'apple', 'orange', 'sandwich', 'pizza', 'hot dog', 'donut',
            'cake', 'carrot', 'broccoli', 'egg', 'bread', 'croissant', 'cookie',
            
            # Containers that likely contain food
            'bottle', 'cup', 'bowl', 'jar', 'can', 'box', 'container',
            
            # Items commonly found in refrigerators
            'milk', 'juice', 'yogurt', 'cheese', 'butter', 'eggs',
            'vegetables', 'fruits', 'meat', 'leftovers',
            
            # Generic food category
            'food', 'meal', 'dish', 'snack', 'ingredient'
        }
        
        # Items to exclude (definitely not food)
        self.non_food_items = {
            'person', 'chair', 'table', 'refrigerator', 'oven', 'microwave',
            'sink', 'counter', 'floor', 'wall', 'ceiling', 'light',
            'knife', 'fork', 'spoon', 'plate', 'napkin'
        }
    
    def is_likely_food_item(self, item_name: str, confidence: float = 0.0) -> bool:
        """
        Determine if an item is likely food or food-related
        FIXED: Much more inclusive for refrigerator contexts
        """
        item_lower = item_name.lower()
        
        # Exclude obvious non-food items
        for non_food in self.non_food_items:
            if non_food in item_lower:
                return False
        
        # Include anything that might be food-related
        for food_word in self.food_related_classes:
            if food_word in item_lower or item_lower in food_word:
                return True
        
        # In a refrigerator context, assume most unidentified items are food
        # Only exclude if confidence is very low
        if confidence < 0.2:
            return False
            
        # Default to including the item (especially in storage contexts)
        return True
    
    def process_segmentation(self, 
                           detection_results: Dict,
                           image: np.ndarray,
                           save_visualization: bool = True,
                           output_dir: str = "data/output") -> Dict[str, Any]:
        """
        Process segmentation with proper refrigerator awareness
        """
        # Extract ALL detected items
        all_detections = self._extract_all_detections(detection_results)
        logger.info(f"Processing {len(all_detections)} total detections")
        
        # Get all item names for context analysis
        all_item_names = [self._get_item_name(item) for item in all_detections]
        
        # Filter food items more intelligently
        food_items = []
        non_food_items = []
        
        for item in all_detections:
            item_name = self._get_item_name(item)
            confidence = item.get('confidence', 0.5)
            
            if self.is_likely_food_item(item_name, confidence):
                # Mark as food item
                item['is_food'] = True
                food_items.append(item)
                logger.info(f"Classified as food: {item_name} (conf: {confidence:.2f})")
            else:
                item['is_food'] = False
                non_food_items.append(item)
                logger.info(f"Classified as non-food: {item_name}")
        
        logger.info(f"Found {len(food_items)} food items out of {len(all_detections)} detections")
        
        # Extract food names for classification
        food_names = [self._get_item_name(item) for item in food_items]
        
        # Classify food type with all context
        food_type, confidence = self.food_classifier.classify_food_type(
            food_names, 
            all_item_names,  # Pass ALL detected items
            len(all_detections),
            image_context={'has_storage_items': len(non_food_items) > 0}
        )
        
        logger.info(f"Food type classification: {food_type} (confidence: {confidence:.2%})")
        
        # Process based on classification
        if food_type == 'complete_dish' and len(food_items) > 3:
            # Override if too many items for a single dish
            logger.info("Overriding to individual_items due to item count")
            food_type = 'individual_items'
            
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
        
        # Add processing statistics
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
    
    def _extract_all_detections(self, detection_results: Dict) -> List[Dict]:
        """Extract all detections from various possible formats"""
        all_detections = []
        
        # Try different result formats
        if 'results' in detection_results and isinstance(detection_results['results'], list):
            # Handle ultralytics format
            for result in detection_results['results']:
                if hasattr(result, 'boxes') and result.boxes is not None:
                    boxes = result.boxes
                    names = result.names if hasattr(result, 'names') else {}
                    
                    for i in range(len(boxes)):
                        try:
                            box = boxes[i]
                            class_id = int(box.cls[0])
                            class_name = names.get(class_id, f'class_{class_id}')
                            
                            detection = {
                                'name': class_name,
                                'confidence': float(box.conf[0]),
                                'bbox': {
                                    'x1': int(box.xyxy[0][0]),
                                    'y1': int(box.xyxy[0][1]),
                                    'x2': int(box.xyxy[0][2]),
                                    'y2': int(box.xyxy[0][3])
                                }
                            }
                            all_detections.append(detection)
                        except Exception as e:
                            logger.warning(f"Error processing box {i}: {e}")
        
        elif 'detections' in detection_results:
            all_detections = detection_results['detections']
        elif 'food_items' in detection_results:
            all_detections = detection_results['food_items']
        elif isinstance(detection_results, list):
            all_detections = detection_results
            
        return all_detections
    
    def _get_item_name(self, item: Dict) -> str:
        """Extract item name from various formats"""
        return item.get('name') or item.get('class_name') or item.get('label', 'unknown')
    
    def _process_individual_items(self, 
                                 food_items: List[Dict], 
                                 image: np.ndarray) -> Dict[str, Any]:
        """
        Process individual items with proper units for refrigerator contents
        """
        segments = []
        measurement_summary = {}
        segment_id = 0
        
        # Group items by type for better organization
        item_groups = {}
        
        for item in food_items:
            food_name = self._get_item_name(item)
            
            # Normalize food names for better grouping
            normalized_name = self._normalize_food_name(food_name)
            
            if normalized_name not in item_groups:
                item_groups[normalized_name] = []
            item_groups[normalized_name].append(item)
        
        # Process each group
        for food_name, items in item_groups.items():
            # Get appropriate unit for this food type
            unit, unit_full = self._get_refrigerator_appropriate_unit(food_name, len(items))
            
            # Create segments for each item
            for idx, item in enumerate(items):
                bbox = self._extract_bbox(item)
                
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
                    'group_name': food_name
                }
                
                segments.append(segment)
                segment_id += 1
            
            # Update measurement summary
            if unit in ['piece', 'unit', 'item']:
                total_count = len(items)
                measurement_summary[food_name] = {
                    'count': total_count,
                    'unit': unit,
                    'formatted': self.measurement_system.format_measurement(total_count, unit)
                }
            else:
                # For volume/weight items, estimate based on typical container sizes
                estimated_amount = self._estimate_container_amount(food_name, len(items))
                measurement_summary[food_name] = {
                    'total': estimated_amount,
                    'unit': unit,
                    'formatted': self.measurement_system.format_measurement(estimated_amount, unit)
                }
        
        return {
            'segments': segments,
            'measurement_summary': measurement_summary
        }
    
    def _normalize_food_name(self, food_name: str) -> str:
        """Normalize food names for better grouping"""
        name_lower = food_name.lower()
        
        # Common normalizations
        normalizations = {
            'egg': 'eggs',
            'eggs': 'eggs',
            'banana': 'banana',
            'bananas': 'banana',
            'apple': 'apple',
            'apples': 'apple',
            'milk': 'milk',
            'bottle': 'bottle',
            'jar': 'jar',
            'container': 'container',
            'juice': 'juice',
            'yogurt': 'yogurt',
            'cheese': 'cheese',
            'vegetable': 'vegetables',
            'vegetables': 'vegetables',
            'fruit': 'fruits',
            'fruits': 'fruits'
        }
        
        for key, value in normalizations.items():
            if key in name_lower:
                return value
                
        return food_name
    
    def _get_refrigerator_appropriate_unit(self, food_name: str, count: int) -> Tuple[str, str]:
        """Get appropriate units for common refrigerator items"""
        name_lower = food_name.lower()
        
        # Specific mappings for refrigerator items
        if 'egg' in name_lower:
            return 'piece', 'pieces'
        elif 'banana' in name_lower or 'apple' in name_lower:
            return 'piece', 'pieces'
        elif 'milk' in name_lower or 'juice' in name_lower:
            return 'ml', 'milliliters'
        elif 'bottle' in name_lower:
            if 'water' in name_lower:
                return 'ml', 'milliliters'
            else:
                return 'bottle', 'bottles'
        elif 'jar' in name_lower:
            return 'jar', 'jars'
        elif 'container' in name_lower or 'box' in name_lower:
            return 'container', 'containers'
        elif 'cheese' in name_lower:
            return 'g', 'grams'
        elif 'yogurt' in name_lower:
            return 'container', 'containers'
        elif 'vegetable' in name_lower or 'fruit' in name_lower:
            if count > 1:
                return 'piece', 'pieces'
            else:
                return 'bunch', 'bunches'
        else:
            # Default based on count
            if count == 1:
                return 'item', 'item'
            else:
                return 'piece', 'pieces'
    
    def _estimate_container_amount(self, food_name: str, count: int) -> float:
        """Estimate amount for containers based on typical sizes"""
        name_lower = food_name.lower()
        
        if 'milk' in name_lower:
            return 1000.0 * count  # 1L per milk container
        elif 'juice' in name_lower:
            return 500.0 * count   # 500ml per juice container
        elif 'yogurt' in name_lower:
            return 150.0 * count   # 150g per yogurt container
        elif 'cheese' in name_lower:
            return 200.0 * count   # 200g per cheese package
        else:
            return 250.0 * count   # Default estimate
    
    def _extract_bbox(self, item: Dict) -> Dict[str, int]:
        """Extract bounding box from various formats"""
        if 'bbox' in item and isinstance(item['bbox'], dict):
            return item['bbox']
        elif 'bbox' in item and isinstance(item['bbox'], (list, tuple)):
            bbox = item['bbox']
            if len(bbox) == 4:
                return {
                    'x1': int(bbox[0]), 'y1': int(bbox[1]),
                    'x2': int(bbox[2]), 'y2': int(bbox[3])
                }
        else:
            logger.warning(f"No bbox found for item: {item.get('name', 'unknown')}")
            return {'x1': 0, 'y1': 0, 'x2': 100, 'y2': 100}
    
    def _create_visualization(self, 
                            image: np.ndarray, 
                            results: Dict,
                            output_dir: str) -> Path:
        """Create visualization with all detected items and measurements"""
        vis_image = image.copy()
        
        # Colors for different items
        colors = [
            (0, 255, 0),    # Green
            (255, 0, 0),    # Blue
            (0, 0, 255),    # Red
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
            (128, 255, 0),  # Green-Yellow
            (255, 128, 0),  # Orange
            (128, 0, 255),  # Purple
            (0, 128, 255),  # Sky Blue
        ]
        
        # Group segments by name for color consistency
        name_to_color = {}
        color_idx = 0
        
        # Font settings
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 2
        
        # Draw each segment
        for segment in results['segments']:
            bbox = segment['bbox']
            name = segment['name']
            
            # Assign consistent color to each food type
            if name not in name_to_color:
                name_to_color[name] = colors[color_idx % len(colors)]
                color_idx += 1
            
            color = name_to_color[name]
            
            # Draw bounding box
            cv2.rectangle(
                vis_image,
                (bbox['x1'], bbox['y1']),
                (bbox['x2'], bbox['y2']),
                color, 2
            )
            
            # Prepare label
            label = f"{name}: {segment['measurement']['formatted']}"
            
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
        summary_bg_height = 150
        
        # Semi-transparent background
        overlay = vis_image.copy()
        cv2.rectangle(overlay, (10, 10), (600, summary_bg_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, vis_image, 0.3, 0, vis_image)
        
        # Add text
        cv2.putText(
            vis_image,
            f"Type: {results['food_type_classification']['type']}",
            (20, summary_y),
            font, 0.7, (255, 255, 255), 2
        )
        
        cv2.putText(
            vis_image,
            f"Total Items: {len(results['segments'])}",
            (20, summary_y + 30),
            font, 0.7, (255, 255, 255), 2
        )
        
        # Add item summary
        y_offset = summary_y + 60
        for food_name, measurement in results.get('measurement_summary', {}).items():
            if isinstance(measurement, dict) and 'formatted' in measurement:
                text = f"{food_name}: {measurement['formatted']}"
                cv2.putText(
                    vis_image,
                    text,
                    (20, y_offset),
                    font, 0.6, (255, 255, 255), 2
                )
                y_offset += 25
                if y_offset > summary_bg_height - 10:
                    break
        
        # Save visualization
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(output_dir) / f"refrigerator_visual_{timestamp}.jpg"
        output_path.parent.mkdir(exist_ok=True, parents=True)
        
        cv2.imwrite(str(output_path), vis_image)
        logger.info(f"Saved visualization to: {output_path}")
        
        return output_path
    
    def _process_complete_dish(self, food_items: List[Dict], image: np.ndarray) -> Dict[str, Any]:
        """Process complete dish - only when truly appropriate"""
        # This should rarely be used for refrigerator contexts
        # Keeping minimal implementation for actual dish scenarios
        
        if not food_items:
            return {'segments': [], 'measurement_summary': {}}
        
        # Create single segment only if truly a single dish
        min_x, min_y = float('inf'), float('inf')
        max_x, max_y = 0, 0
        
        for item in food_items:
            bbox = self._extract_bbox(item)
            min_x = min(min_x, bbox['x1'])
            min_y = min(min_y, bbox['y1'])
            max_x = max(max_x, bbox['x2'])
            max_y = max(max_y, bbox['y2'])
        
        dish_name = self._get_item_name(food_items[0])
        
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
            'components': food_items,
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