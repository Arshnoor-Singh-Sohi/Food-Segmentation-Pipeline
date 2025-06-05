"""Combined YOLO + SAM 2 pipeline for comprehensive food analysis."""

import numpy as np
import cv2
from typing import List, Dict, Any, Tuple, Optional
import logging
from pathlib import Path
import json
import sys

# Add src directory to Python path
current_dir = Path(__file__).parent.parent
sys.path.insert(0, str(current_dir))

# Use absolute imports instead of relative imports
from models.yolo_detector import FoodYOLODetector
from models.sam2_predictor import FoodSAM2Predictor
from utils.nutrition_db import NutritionDatabase
from preprocessing.food_preprocessor import FoodImagePreprocessor

logger = logging.getLogger(__name__)


class FoodAnalysisPipeline:
    """Complete food analysis pipeline combining YOLO detection and SAM 2 segmentation."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the complete food analysis pipeline."""
        self.config = config
        
        # Initialize components
        self.yolo_detector = FoodYOLODetector(config['models']['yolo'])
        self.sam2_predictor = FoodSAM2Predictor(config['models']['sam2'])
        self.preprocessor = FoodImagePreprocessor(config.get('preprocessing', {}))
        self.nutrition_db = NutritionDatabase(config.get('nutrition', {}))
        
        # Processing parameters
        self.quality_threshold = config['processing'].get('quality_threshold', 0.7)
        self.min_area_ratio = config['processing'].get('min_area_ratio', 0.01)
        self.max_area_ratio = config['processing'].get('max_area_ratio', 0.8)
        
        logger.info("Food analysis pipeline initialized successfully")
    
    def analyze_meal_image(
        self, 
        image_path: str, 
        interactive_points: Optional[List[Tuple[int, int]]] = None,
        save_results: bool = True
    ) -> Dict[str, Any]:
        """
        Complete meal analysis: detection → segmentation → nutrition calculation.
        
        Args:
            image_path: Path to the food image
            interactive_points: Optional list of (x, y) points for manual segmentation
            save_results: Whether to save results to output directory
        
        Returns:
            Complete analysis results including nutrition information
        """
        try:
            # Load and preprocess image
            image = self.preprocessor.load_and_preprocess(image_path)
            original_image = cv2.imread(image_path)
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            
            # Step 1: Detect food items with YOLO
            logger.info("Step 1: Detecting food items with YOLO")
            detections = self.yolo_detector.detect_food_items(image)
            
            # Step 2: Precise segmentation with SAM 2
            logger.info("Step 2: Performing precise segmentation with SAM 2")
            self.sam2_predictor.set_image(image)
            
            # Process each detection
            food_items = []
            for i, detection in enumerate(detections):
                try:
                    # Use YOLO bounding box as SAM 2 prompt
                    bbox = detection['bbox']
                    masks, scores, _ = self.sam2_predictor.predict_with_box(
                        bbox, multimask_output=True
                    )
                    
                    # Select best mask
                    best_mask_idx = self.sam2_predictor.select_best_mask(
                        masks, scores, image
                    )
                    
                    # Calculate portion size and nutrition
                    portion_info = self._estimate_portion_size(
                        masks[best_mask_idx], image, detection['class_name']
                    )
                    
                    nutrition_info = self.nutrition_db.get_nutrition_info(
                        detection['class_name'], portion_info['portion_size']
                    )
                    
                    food_item = {
                        'id': i,
                        'name': detection['class_name'],
                        'detection_confidence': detection['confidence'],
                        'segmentation_confidence': scores[best_mask_idx],
                        'bbox': bbox,
                        'mask': masks[best_mask_idx],
                        'portion_info': portion_info,
                        'nutrition': nutrition_info,
                        'food_analysis': detection.get('food_analysis', {}),
                        'quality_score': self._calculate_quality_score(
                            detection['confidence'], scores[best_mask_idx]
                        )
                    }
                    
                    food_items.append(food_item)
                    
                except Exception as e:
                    logger.warning(f"Failed to process detection {i}: {e}")
                    continue
            
            # Step 3: Handle interactive points if provided
            if interactive_points:
                logger.info("Step 3: Processing interactive segmentation points")
                interactive_items = self._process_interactive_points(
                    image, interactive_points
                )
                food_items.extend(interactive_items)
            
            # Step 4: Generate automatic masks for missed items
            logger.info("Step 4: Generating automatic masks for missed items")
            automatic_items = self._find_missed_food_items(image, food_items)
            food_items.extend(automatic_items)
            
            # Step 5: Compile comprehensive analysis
            meal_analysis = self._compile_meal_analysis(
                food_items, original_image, image_path
            )
            
            # Save results if requested
            if save_results:
                self._save_analysis_results(meal_analysis, image_path)
            
            logger.info(f"Analysis complete: {len(food_items)} food items identified")
            return meal_analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze meal image {image_path}: {e}")
            raise
    
    def _estimate_portion_size(
        self, 
        mask: np.ndarray, 
        image: np.ndarray, 
        food_name: str
    ) -> Dict[str, Any]:
        """Estimate portion size from segmentation mask."""
        # Calculate mask area in pixels
        pixel_area = np.sum(mask)
        total_pixels = image.shape[0] * image.shape[1]
        area_ratio = pixel_area / total_pixels
        
        # Estimate real-world area (simplified approach)
        # In production, this would use camera calibration or reference objects
        estimated_area_cm2 = self._pixels_to_area(pixel_area, image.shape)
        
        # Convert to serving size based on food type
        serving_size = self._area_to_serving_size(estimated_area_cm2, food_name)
        
        return {
            'pixel_area': int(pixel_area),
            'area_ratio': float(area_ratio),
            'estimated_area_cm2': float(estimated_area_cm2),
            'portion_size': serving_size
        }
    
    def _pixels_to_area(self, pixel_area: int, image_shape: Tuple[int, int, int]) -> float:
        """Convert pixel area to real-world area estimate."""
        # Simplified conversion assuming average dinner plate is 25cm diameter
        # and occupies about 60% of image area in typical food photos
        total_pixels = image_shape[0] * image_shape[1]
        plate_area_cm2 = np.pi * (12.5 ** 2)  # 25cm diameter plate
        
        # Rough conversion factor
        pixels_per_cm2 = (total_pixels * 0.6) / plate_area_cm2
        
        return pixel_area / pixels_per_cm2
    
    def _area_to_serving_size(self, area_cm2: float, food_name: str) -> Dict[str, Any]:
        """Convert area to standard serving sizes."""
        # Food-specific conversion factors based on typical densities and shapes
        conversion_factors = {
            'apple': {'factor': 0.8, 'unit': 'medium apple', 'grams_per_unit': 150},
            'banana': {'factor': 0.6, 'unit': 'medium banana', 'grams_per_unit': 120},
            'orange': {'factor': 0.7, 'unit': 'medium orange', 'grams_per_unit': 130},
            'pizza': {'factor': 0.12, 'unit': 'slice', 'grams_per_unit': 100},
            'sandwich': {'factor': 0.4, 'unit': 'half sandwich', 'grams_per_unit': 150},
            'salad': {'factor': 2.0, 'unit': 'cups', 'grams_per_unit': 50},
            'pasta': {'factor': 1.5, 'unit': 'cups', 'grams_per_unit': 80},
            'rice': {'factor': 2.2, 'unit': 'cups', 'grams_per_unit': 90},
            'chicken': {'factor': 0.35, 'unit': 'oz', 'grams_per_unit': 28},
            'beef': {'factor': 0.35, 'unit': 'oz', 'grams_per_unit': 28},
            'broccoli': {'factor': 3.0, 'unit': 'cups', 'grams_per_unit': 30},
            'carrot': {'factor': 1.5, 'unit': 'medium carrot', 'grams_per_unit': 60}
        }
        
        food_key = food_name.lower()
        if food_key in conversion_factors:
            factor_data = conversion_factors[food_key]
            quantity = area_cm2 * factor_data['factor']
            estimated_grams = quantity * factor_data['grams_per_unit']
            
            return {
                'quantity': round(quantity, 1),
                'unit': factor_data['unit'],
                'estimated_grams': round(estimated_grams, 1)
            }
        else:
            # Default estimation for unknown foods
            estimated_grams = area_cm2 * 2.5  # Rough density estimate
            return {
                'quantity': round(area_cm2, 1),
                'unit': 'cm²',
                'estimated_grams': round(estimated_grams, 1)
            }
    
    def _process_interactive_points(
        self, 
        image: np.ndarray, 
        points: List[Tuple[int, int]]
    ) -> List[Dict[str, Any]]:
        """Process interactive segmentation points."""
        interactive_items = []
        
        for i, point in enumerate(points):
            try:
                # Convert point to required format
                point_coords = [list(point)]
                point_labels = [1]  # Positive point
                
                # Get segmentation
                masks, scores, _ = self.sam2_predictor.predict_with_points(
                    point_coords, point_labels, multimask_output=True
                )
                
                # Select best mask
                best_mask_idx = self.sam2_predictor.select_best_mask(
                    masks, scores, image, prefer_smaller=True
                )
                
                # Estimate what food this might be (simplified approach)
                food_name = f"interactive_item_{i}"
                
                # Calculate portion info
                portion_info = self._estimate_portion_size(
                    masks[best_mask_idx], image, food_name
                )
                
                interactive_item = {
                    'id': f'interactive_{i}',
                    'name': food_name,
                    'detection_confidence': 1.0,  # User-provided
                    'segmentation_confidence': scores[best_mask_idx],
                    'bbox': self._mask_to_bbox(masks[best_mask_idx]),
                    'mask': masks[best_mask_idx],
                    'portion_info': portion_info,
                    'nutrition': {'calories': 0, 'note': 'Manual segmentation - nutrition TBD'},
                    'source': 'interactive',
                    'click_point': point
                }
                
                interactive_items.append(interactive_item)
                
            except Exception as e:
                logger.warning(f"Failed to process interactive point {point}: {e}")
                continue
        
        return interactive_items
    
    def _find_missed_food_items(
        self, 
        image: np.ndarray, 
        existing_items: List[Dict]
    ) -> List[Dict[str, Any]]:
        """Find food items that might have been missed by detection."""
        try:
            # Generate automatic masks
            automatic_masks = self.sam2_predictor.generate_automatic_masks(image)
            
            # Filter out masks that overlap significantly with existing items
            existing_masks = [item['mask'] for item in existing_items]
            new_items = []
            
            for mask_data in automatic_masks:
                mask = mask_data['segmentation']
                
                # Check overlap with existing masks
                if not self._has_significant_overlap(mask, existing_masks):
                    # This might be a missed food item
                    portion_info = self._estimate_portion_size(
                        mask, image, "unknown_food"
                    )
                    
                    # Only include if it's a reasonable size for food
                    if (0.01 < portion_info['area_ratio'] < 0.3 and
                        mask_data['stability_score'] > 0.9):
                        
                        new_item = {
                            'id': f"auto_{len(new_items)}",
                            'name': 'unknown_food',
                            'detection_confidence': 0.8,  # Estimated
                            'segmentation_confidence': mask_data['stability_score'],
                            'bbox': mask_data['bbox'],
                            'mask': mask,
                            'portion_info': portion_info,
                            'nutrition': {'calories': 0, 'note': 'Unknown food item'},
                            'source': 'automatic',
                            'stability_score': mask_data['stability_score']
                        }
                        
                        new_items.append(new_item)
            
            logger.info(f"Found {len(new_items)} additional food items automatically")
            return new_items
            
        except Exception as e:
            logger.warning(f"Failed to find missed food items: {e}")
            return []
    
    def _has_significant_overlap(
        self, 
        mask: np.ndarray, 
        existing_masks: List[np.ndarray], 
        threshold: float = 0.5
    ) -> bool:
        """Check if mask has significant overlap with existing masks."""
        for existing_mask in existing_masks:
            # Calculate IoU (Intersection over Union)
            intersection = np.logical_and(mask, existing_mask)
            union = np.logical_or(mask, existing_mask)
            
            if np.sum(union) > 0:
                iou = np.sum(intersection) / np.sum(union)
                if iou > threshold:
                    return True
        
        return False
    
    def _mask_to_bbox(self, mask: np.ndarray) -> List[int]:
        """Convert mask to bounding box."""
        rows, cols = np.where(mask)
        if len(rows) == 0:
            return [0, 0, 0, 0]
        
        y1, y2 = rows.min(), rows.max()
        x1, x2 = cols.min(), cols.max()
        
        return [int(x1), int(y1), int(x2), int(y2)]
    
    def _calculate_quality_score(
        self, 
        detection_conf: float, 
        segmentation_conf: float
    ) -> float:
        """Calculate overall quality score for a food item."""
        # Weighted combination of detection and segmentation confidence
        quality_score = (detection_conf * 0.4 + segmentation_conf * 0.6)
        return float(quality_score)
    
    def _compile_meal_analysis(
        self, 
        food_items: List[Dict], 
        original_image: np.ndarray, 
        image_path: str
    ) -> Dict[str, Any]:
        """Compile comprehensive meal analysis."""
        # Calculate total nutrition
        total_nutrition = self._calculate_total_nutrition(food_items)
        
        # Generate meal summary
        meal_summary = {
            'total_items': len(food_items),
            'detected_items': len([item for item in food_items if item.get('source') != 'interactive']),
            'interactive_items': len([item for item in food_items if item.get('source') == 'interactive']),
            'automatic_items': len([item for item in food_items if item.get('source') == 'automatic']),
            'avg_quality_score': np.mean([item.get('quality_score', 0) for item in food_items]) if food_items else 0,
            'food_types': list(set(item['name'] for item in food_items))
        }
        
        return {
            'image_path': image_path,
            'image_shape': original_image.shape,
            'analysis_timestamp': self._get_timestamp(),
            'meal_summary': meal_summary,
            'total_nutrition': total_nutrition,
            'food_items': food_items,
            'processing_config': {
                'yolo_confidence': self.yolo_detector.confidence_threshold,
                'sam2_model': self.sam2_predictor.config.get('model_type'),
                'quality_threshold': self.quality_threshold
            }
        }
    
    def _calculate_total_nutrition(self, food_items: List[Dict]) -> Dict[str, float]:
        """Calculate total nutrition from all food items."""
        total_nutrition = {
            'calories': 0.0,
            'protein': 0.0,
            'carbohydrates': 0.0,
            'fat': 0.0,
            'fiber': 0.0,
            'sugar': 0.0,
            'sodium': 0.0
        }
        
        for item in food_items:
            nutrition = item.get('nutrition', {})
            for nutrient in total_nutrition:
                if nutrient in nutrition and isinstance(nutrition[nutrient], (int, float)):
                    total_nutrition[nutrient] += nutrition[nutrient]
        
        return total_nutrition
    
    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def _save_analysis_results(self, analysis: Dict[str, Any], image_path: str):
        """Save analysis results to output directory."""
        try:
            output_dir = Path(self.config['paths']['output_dir'])
            output_dir.mkdir(exist_ok=True)
            
            # Create output filename
            image_name = Path(image_path).stem
            output_file = output_dir / f"{image_name}_analysis.json"
            
            # Prepare JSON-serializable data
            json_data = self._prepare_for_json(analysis)
            
            # Save analysis
            with open(output_file, 'w') as f:
                json.dump(json_data, f, indent=2)
            
            logger.info(f"Analysis results saved to: {output_file}")
            
        except Exception as e:
            logger.warning(f"Failed to save analysis results: {e}")
    
    def _prepare_for_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare data for JSON serialization by converting numpy arrays."""
        if isinstance(data, dict):
            return {k: self._prepare_for_json(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._prepare_for_json(item) for item in data]
        elif isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, np.integer):
            return int(data)
        elif isinstance(data, np.floating):
            return float(data)
        else:
            return data