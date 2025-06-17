"""
Portion size estimation module
"""

import numpy as np
from typing import Dict, Tuple, Optional

class PortionEstimator:
    """Estimate portion sizes from image data"""
    
    def __init__(self, reference_plate_diameter_cm: float = 23.0):
        self.reference_plate_cm = reference_plate_diameter_cm
        
        # Food density factors (g/cm³)
        self.density_factors = {
            'salad': 0.15,
            'soup': 0.95,
            'rice': 0.75,
            'pasta': 0.60,
            'meat': 1.05,
            'bread': 0.25,
            'vegetables': 0.40,
            'fruit': 0.65,
            'dessert': 0.55,
            'default': 0.65
        }
        
        # Typical serving sizes (grams)
        self.typical_servings = {
            'pizza': 125,
            'burger': 200,
            'salad': 150,
            'pasta': 200,
            'rice': 150,
            'steak': 225,
            'soup': 250,
            'sandwich': 150
        }
    
    def estimate_portion(self, 
                        food_type: str,
                        mask_area_pixels: int,
                        image_shape: Tuple[int, int],
                        bbox: Dict[str, int]) -> Dict[str, any]:
        """
        Estimate portion size from mask and bbox
        
        Args:
            food_type: Type of food
            mask_area_pixels: Number of pixels in segmentation mask
            image_shape: (height, width) of image
            bbox: Bounding box coordinates
            
        Returns:
            Portion estimation details
        """
        # Calculate relative size
        image_area = image_shape[0] * image_shape[1]
        mask_percentage = (mask_area_pixels / image_area) * 100
        
        # Estimate physical size (assuming typical photo distance)
        # This is a rough approximation
        estimated_area_cm2 = mask_percentage * 5  # Scale factor
        
        # Get density for food type
        density = self._get_density(food_type)
        
        # Estimate volume (assuming average thickness)
        thickness_cm = 2.5  # Average food thickness
        volume_cm3 = estimated_area_cm2 * thickness_cm
        
        # Calculate weight
        weight_g = volume_cm3 * density
        
        # Adjust based on typical servings
        if food_type.lower() in self.typical_servings:
            typical = self.typical_servings[food_type.lower()]
            # Bias towards typical serving sizes
            weight_g = (weight_g + typical) / 2
        
        # Determine serving size description
        serving_desc = self._get_serving_description(weight_g, food_type)
        
        return {
            'estimated_weight_g': round(weight_g, 1),
            'serving_description': serving_desc,
            'mask_area_percentage': round(mask_percentage, 2),
            'confidence': self._estimate_confidence(mask_percentage),
            'estimation_method': 'area_based'
        }
    
    def _get_density(self, food_type: str) -> float:
        """Get density factor for food type"""
        food_lower = food_type.lower()
        
        for food_key, density in self.density_factors.items():
            if food_key in food_lower:
                return density
        
        return self.density_factors['default']
    
    def _get_serving_description(self, weight_g: float, food_type: str) -> str:
        """Convert weight to serving description"""
        if weight_g < 50:
            size = "small"
            desc = "snack portion"
        elif weight_g < 100:
            size = "small"
            desc = "light serving"
        elif weight_g < 200:
            size = "medium"
            desc = "single serving"
        elif weight_g < 350:
            size = "large"
            desc = "full meal"
        else:
            size = "extra large"
            desc = "sharing portion"
        
        return f"{size} ({desc})"
    
    def _estimate_confidence(self, mask_percentage: float) -> str:
        """Estimate confidence based on mask quality"""
        if mask_percentage < 1:
            return "low"
        elif mask_percentage < 5:
            return "medium"
        else:
            return "high"