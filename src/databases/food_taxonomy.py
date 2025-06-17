"""
Food taxonomy database handler
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

class FoodTaxonomy:
    """Handle food categorization and hierarchy"""
    
    def __init__(self, taxonomy_path: str = "data/databases/food_taxonomy/food_hierarchy.json"):
        self.taxonomy_path = Path(taxonomy_path)
        self.taxonomy_data = {}
        self._load_taxonomy()
        
    def _load_taxonomy(self):
        """Load food taxonomy from JSON"""
        if self.taxonomy_path.exists():
            with open(self.taxonomy_path, 'r') as f:
                self.taxonomy_data = json.load(f)
        else:
            # Create default taxonomy
            self.taxonomy_data = {
                "food_categories": {
                    "fruits": ["apple", "banana", "orange"],
                    "vegetables": ["carrot", "broccoli", "spinach"],
                    "proteins": ["chicken", "beef", "fish"],
                    "grains": ["rice", "pasta", "bread"],
                    "dairy": ["milk", "cheese", "yogurt"]
                }
            }
            # Save it
            self.taxonomy_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.taxonomy_path, 'w') as f:
                json.dump(self.taxonomy_data, f, indent=2)
    
    def get_category(self, food_name: str) -> Optional[str]:
        """Get category for a food item"""
        food_lower = food_name.lower()
        
        for category, items in self.taxonomy_data.get("food_categories", {}).items():
            if isinstance(items, dict):
                # Handle nested categories
                for subcategory, subitems in items.items():
                    if any(item in food_lower for item in subitems):
                        return f"{category}/{subcategory}"
            elif isinstance(items, list):
                # Handle flat lists
                if any(item in food_lower for item in items):
                    return category
        
        return None
    
    def get_meal_type(self, food_name: str) -> Optional[str]:
        """Get meal type for a food"""
        food_lower = food_name.lower()
        
        for meal_type, items in self.taxonomy_data.get("meal_types", {}).items():
            if any(item in food_lower for item in items):
                return meal_type
        
        return None