"""Enhanced SAM 2 predictor for food image segmentation."""

import torch
import numpy as np
import cv2
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import logging

import sys
sys.path.append(r'C:\temp\sam2_install\sam2')

try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
except ImportError:
    print("SAM 2 not installed. Please install from: https://github.com/facebookresearch/sam2")
    raise

logger = logging.getLogger(__name__)

class FoodSAM2Predictor:
    """Enhanced SAM 2 predictor optimized for food image segmentation."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize SAM 2 predictor with food-specific optimizations."""
        self.config = config
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.predictor = None
        self.mask_generator = None
        self.current_image = None
        
        logger.info(f"Initializing SAM 2 predictor on device: {self.device}")
        self._load_model()
    
    def _load_model(self):
        """Load SAM 2 model and initialize predictors."""
        try:
            model_cfg = self.config['config_path']
            checkpoint_path = self.config['checkpoint_path']
            
            # Check if files exist
            if not Path(checkpoint_path).exists():
                raise FileNotFoundError(f"SAM 2 checkpoint not found: {checkpoint_path}")
            
            # Build model
            self.model = build_sam2(model_cfg, checkpoint_path, device=self.device)
            
            # Initialize predictor for interactive segmentation
            self.predictor = SAM2ImagePredictor(self.model)
            
            # Initialize automatic mask generator
            self.mask_generator = SAM2AutomaticMaskGenerator(
                model=self.model,
                points_per_side=32,
                pred_iou_thresh=0.7,
                stability_score_thresh=0.92,
                crop_n_layers=1,
                min_mask_region_area=100,
                use_m2m=True
            )
            
            logger.info("SAM 2 model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load SAM 2 model: {e}")
            raise
    
    def set_image(self, image: np.ndarray) -> None:
        """Set image for segmentation."""
        if image is None or image.size == 0:
            raise ValueError("Invalid image provided")
        
        # Ensure image is in RGB format
        if len(image.shape) == 3 and image.shape[2] == 3:
            self.current_image = image
            self.predictor.set_image(image)
            logger.debug(f"Image set for segmentation: {image.shape}")
        else:
            raise ValueError(f"Image must be RGB format, got shape: {image.shape}")
    
    def predict_with_points(
        self, 
        points: List[List[int]], 
        labels: List[int],
        multimask_output: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict segmentation masks using point prompts."""
        if self.current_image is None:
            raise ValueError("No image set. Call set_image() first.")
        
        if len(points) != len(labels):
            raise ValueError("Number of points must match number of labels")
        
        point_coords = np.array(points)
        point_labels = np.array(labels)
        
        masks, scores, logits = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=multimask_output
        )
        
        logger.debug(f"Generated {len(masks)} masks with scores: {scores}")
        return masks, scores, logits
    
    def predict_with_box(
        self, 
        box: List[int],
        multimask_output: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict segmentation masks using bounding box prompt."""
        if self.current_image is None:
            raise ValueError("No image set. Call set_image() first.")
        
        box_array = np.array(box)
        
        masks, scores, logits = self.predictor.predict(
            box=box_array,
            multimask_output=multimask_output
        )
        
        logger.debug(f"Generated {len(masks)} masks from box with scores: {scores}")
        return masks, scores, logits
    
    def generate_automatic_masks(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Generate automatic masks for food items in the image."""
        if image is None or image.size == 0:
            raise ValueError("Invalid image provided")
        
        masks = self.mask_generator.generate(image)
        
        # Filter masks for food relevance
        filtered_masks = self._filter_food_masks(masks, image)
        
        logger.info(f"Generated {len(filtered_masks)} food-relevant masks from {len(masks)} total")
        return filtered_masks
    
    def _filter_food_masks(self, masks: List[Dict], image: np.ndarray) -> List[Dict]:
        """Filter masks to focus on food items."""
        filtered_masks = []
        image_area = image.shape[0] * image.shape[1]
        
        for mask_data in masks:
            mask = mask_data['segmentation']
            area = mask_data['area']
            stability_score = mask_data['stability_score']
            bbox = mask_data['bbox']
            
            # Calculate relative area
            relative_area = area / image_area
            
            # Filter criteria for food items
            if (0.005 < relative_area < 0.7 and  # Reasonable size
                stability_score > 0.85 and       # Good stability
                self._is_food_like_region(mask, image, bbox)):  # Food-like appearance
                
                filtered_masks.append(mask_data)
        
        # Sort by stability score and area
        filtered_masks.sort(key=lambda x: (x['stability_score'], x['area']), reverse=True)
        
        return filtered_masks
    
    def _is_food_like_region(self, mask: np.ndarray, image: np.ndarray, bbox: List) -> bool:
        """Heuristic check if a masked region looks like food."""
        try:
            # Extract masked region
            masked_region = image[mask]
            
            if len(masked_region) == 0:
                return False
            
            # Food items typically have:
            # 1. Moderate color variation (not too uniform)
            color_std = np.std(masked_region, axis=0).mean()
            if color_std < 8:  # Too uniform, likely background
                return False
            
            # 2. Not pure white/black (plates, shadows)
            mean_intensity = np.mean(masked_region)
            if mean_intensity < 25 or mean_intensity > 235:
                return False
            
            # 3. Reasonable aspect ratio (not too elongated)
            x, y, w, h = bbox
            aspect_ratio = max(w, h) / min(w, h)
            if aspect_ratio > 5:  # Too elongated
                return False
            
            # 4. Some texture variation (not completely smooth)
            if len(masked_region) > 100:  # Only for larger regions
                gray_region = cv2.cvtColor(masked_region.reshape(-1, 1, 3), cv2.COLOR_RGB2GRAY)
                texture_var = np.var(gray_region)
                if texture_var < 50:  # Too smooth
                    return False
            
            return True
            
        except Exception as e:
            logger.warning(f"Error in food-like region check: {e}")
            return True  # Default to keeping the mask
    
    def select_best_mask(
        self, 
        masks: np.ndarray, 
        scores: np.ndarray, 
        image: np.ndarray,
        prefer_smaller: bool = False
    ) -> int:
        """Select the best mask for food segmentation."""
        if len(masks) == 0:
            raise ValueError("No masks provided")
        
        mask_scores = []
        image_area = image.shape[0] * image.shape[1]
        
        for i, (mask, confidence) in enumerate(zip(masks, scores)):
            score = float(confidence)
            
            # Area considerations
            area_ratio = np.sum(mask) / image_area
            
            # Penalize masks that are too small or large
            if area_ratio < 0.005 or area_ratio > 0.8:
                score *= 0.3
            elif 0.01 < area_ratio < 0.5:  # Ideal range for food items
                score *= 1.2
            
            # Prefer smaller masks for individual food items if specified
            if prefer_smaller and area_ratio < 0.1:
                score *= 1.1
            
            # Check boundary quality (important for food portions)
            boundary_score = self._calculate_boundary_quality(mask)
            score *= boundary_score
            
            mask_scores.append(score)
        
        best_idx = np.argmax(mask_scores)
        logger.debug(f"Selected mask {best_idx} with adjusted score: {mask_scores[best_idx]:.3f}")
        
        return best_idx
    
    def _calculate_boundary_quality(self, mask: np.ndarray) -> float:
        """Calculate quality score based on mask boundary characteristics."""
        try:
            # Find contours
            contours, _ = cv2.findContours(
                mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            
            if not contours:
                return 0.5
            
            # Use largest contour
            main_contour = max(contours, key=cv2.contourArea)
            
            # Calculate shape metrics
            area = cv2.contourArea(main_contour)
            perimeter = cv2.arcLength(main_contour, True)
            
            if area == 0 or perimeter == 0:
                return 0.5
            
            # Compactness score (lower is better for food items)
            compactness = (perimeter ** 2) / (4 * np.pi * area)
            compactness_score = max(0.2, 1.0 / max(1.0, compactness / 3.0))
            
            # Smoothness score
            epsilon = 0.02 * perimeter
            approx = cv2.approxPolyDP(main_contour, epsilon, True)
            smoothness_score = max(0.5, 1.0 - len(approx) / 20.0)
            
            # Combined boundary score
            boundary_score = (compactness_score + smoothness_score) / 2
            
            return np.clip(boundary_score, 0.2, 1.5)
            
        except Exception as e:
            logger.warning(f"Error calculating boundary quality: {e}")
            return 1.0
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            "model_type": self.config.get('model_type', 'unknown'),
            "device": self.device,
            "checkpoint_path": self.config.get('checkpoint_path'),
            "config_path": self.config.get('config_path'),
            "is_loaded": self.model is not None
        }