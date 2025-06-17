"""
Cuisine identification module
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

class CuisineIdentifier:
    """Identify cuisine type from food items"""
    
    def __init__(self, cuisine_db_path: str = "data/databases/cuisine_mapping/cuisine_patterns.json"):
        self.cuisine_db_path = Path(cuisine_db_path)
        self._load_cuisine_database()
        
    def _load_cuisine_database(self):
        """Load cuisine patterns database"""
        if self.cuisine_db_path.exists():
            with open(self.cuisine_db_path, 'r') as f:
                self.cuisine_data = json.load(f)
        else:
            # Fallback data
            self.cuisine_data = {
                'cuisine_indicators': {
                    'italian': {'dishes': ['pizza', 'pasta'], 'ingredients': ['basil', 'mozzarella']},
                    'japanese': {'dishes': ['sushi', 'ramen'], 'ingredients': ['soy sauce', 'wasabi']},
                    'mexican': {'dishes': ['taco', 'burrito'], 'ingredients': ['salsa', 'cilantro']},
                    'american': {'dishes': ['burger', 'hot dog'], 'ingredients': ['ketchup', 'mustard']}
                }
            }
    
    def identify_cuisine(self, food_type: str, ingredients: List[str] = None) -> Dict[str, float]:
        """
        Identify cuisine with confidence scores
        
        Returns:
            Dict of cuisine: confidence pairs
        """
        scores = {}
        food_lower = food_type.lower()
        
        for cuisine, indicators in self.cuisine_data['cuisine_indicators'].items():
            score = 0.0
            
            # Check dishes
            for dish in indicators.get('dishes', []):
                if dish in food_lower:
                    score += 0.8
                    break
            
            # Check ingredients if provided
            if ingredients:
                for ingredient in ingredients:
                    if ingredient.lower() in [i.lower() for i in indicators.get('ingredients', [])]:
                        score += 0.2
            
            # Check keywords
            for keyword in indicators.get('keywords', []):
                if keyword in food_lower:
                    score += 0.3
            
            if score > 0:
                scores[cuisine] = min(score, 1.0)
        
        # Normalize scores
        if scores:
            max_score = max(scores.values())
            if max_score > 0:
                scores = {k: v/max_score for k, v in scores.items()}
        
        return scores
    
    def get_top_cuisine(self, food_type: str, ingredients: List[str] = None) -> str:
        """Get most likely cuisine"""
        scores = self.identify_cuisine(food_type, ingredients)
        
        if scores:
            return max(scores, key=scores.get)
        
        return "international"