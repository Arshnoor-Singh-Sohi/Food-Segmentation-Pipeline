"""Nutrition database for food analysis."""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class NutritionDatabase:
    """Handle nutrition database operations."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize nutrition database."""
        self.config = config
        self.database_path = Path(config.get('database_path', 'data/nutrition_database.json'))
        self.nutrition_data = {}
        self.default_serving_size = config.get('default_serving_size', 100)
        
        self._load_database()
    
    def _load_database(self):
        """Load nutrition database from JSON file."""
        try:
            if self.database_path.exists():
                with open(self.database_path, 'r') as f:
                    self.nutrition_data = json.load(f)
                logger.info(f"Loaded nutrition database with {len(self.nutrition_data)} items")
            else:
                logger.warning(f"Nutrition database not found: {self.database_path}")
                self.nutrition_data = self._get_default_nutrition_data()
        except Exception as e:
            logger.error(f"Failed to load nutrition database: {e}")
            self.nutrition_data = self._get_default_nutrition_data()
    
    def get_nutrition_info(self, food_name: str, portion_info: Dict[str, Any]) -> Dict[str, Any]:
        """Get nutrition information for a food item."""
        food_key = food_name.lower().strip()
        
        # Get base nutrition data
        base_nutrition = self.nutrition_data.get(food_key, self._get_default_nutrition(food_name))
        
        # Calculate based on portion size
        estimated_grams = portion_info.get('estimated_grams', self.default_serving_size)
        scale_factor = estimated_grams / 100  # Base nutrition is per 100g
        
        scaled_nutrition = {}
        for nutrient, value in base_nutrition.items():
            if isinstance(value, (int, float)):
                scaled_nutrition[nutrient] = round(value * scale_factor, 1)
            else:
                scaled_nutrition[nutrient] = value
        
        # Add portion information
        scaled_nutrition['portion_info'] = portion_info
        scaled_nutrition['scaling_factor'] = scale_factor
        
        return scaled_nutrition
    
    def _get_default_nutrition(self, food_name: str) -> Dict[str, Any]:
        """Get default nutrition for unknown foods."""
        return {
            'calories_per_100g': 150,
            'protein': 5.0,
            'carbohydrates': 20.0,
            'fat': 3.0,
            'fiber': 2.0,
            'sugar': 5.0,
            'sodium': 50,
            'note': f'Estimated values for {food_name}'
        }
    
    def _get_default_nutrition_data(self) -> Dict[str, Dict]:
        """Get default nutrition database."""
        return {
            "apple": {
                "calories_per_100g": 52,
                "protein": 0.3,
                "carbohydrates": 14,
                "fat": 0.2,
                "fiber": 2.4,
                "sugar": 10.4,
                "sodium": 1
            },
            "banana": {
                "calories_per_100g": 89,
                "protein": 1.1,
                "carbohydrates": 23,
                "fat": 0.3,
                "fiber": 2.6,
                "sugar": 12,
                "sodium": 1
            },
            "orange": {
                "calories_per_100g": 47,
                "protein": 0.9,
                "carbohydrates": 12,
                "fat": 0.1,
                "fiber": 2.4,
                "sugar": 9.4,
                "sodium": 0
            }
        }