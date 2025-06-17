"""
Allergen database handler
"""

from typing import List, Dict, Optional

class AllergenDatabase:
    """Handle allergen information for foods"""
    
    def __init__(self):
        # Common allergen mappings
        self.allergen_data = {
            'gluten': [
                'wheat', 'bread', 'pasta', 'pizza', 'cake', 'cookie',
                'donut', 'bagel', 'muffin', 'pancake', 'waffle',
                'cereal', 'cracker', 'pretzel', 'beer'
            ],
            'dairy': [
                'milk', 'cheese', 'yogurt', 'butter', 'cream',
                'ice cream', 'pizza', 'lasagna', 'cheesecake',
                'chocolate', 'latte', 'cappuccino'
            ],
            'eggs': [
                'egg', 'mayonnaise', 'cake', 'cookie', 'pasta',
                'bread', 'pancake', 'waffle', 'custard', 'meringue'
            ],
            'nuts': [
                'peanut', 'almond', 'walnut', 'cashew', 'pecan',
                'pistachio', 'hazelnut', 'macadamia', 'pad thai',
                'pesto', 'baklava', 'nutella'
            ],
            'soy': [
                'soy', 'tofu', 'edamame', 'miso', 'tempeh',
                'soy sauce', 'teriyaki', 'sushi'
            ],
            'shellfish': [
                'shrimp', 'crab', 'lobster', 'crawfish', 'prawn',
                'scallop', 'oyster', 'mussel', 'clam'
            ],
            'fish': [
                'salmon', 'tuna', 'cod', 'halibut', 'tilapia',
                'trout', 'bass', 'sushi', 'sashimi', 'fish and chips'
            ],
            'sesame': [
                'sesame', 'tahini', 'hummus', 'bagel', 'burger bun',
                'sushi', 'asian cuisine'
            ]
        }
    
    def get_allergens(self, food_name: str, ingredients: List[str] = None) -> List[str]:
        """Get potential allergens for a food"""
        allergens = []
        food_lower = food_name.lower()
        
        # Check food name
        for allergen, foods in self.allergen_data.items():
            if any(trigger in food_lower for trigger in foods):
                allergens.append(allergen)
        
        # Check ingredients if provided
        if ingredients:
            for ingredient in ingredients:
                ingredient_lower = ingredient.lower()
                for allergen, foods in self.allergen_data.items():
                    if any(trigger in ingredient_lower for trigger in foods):
                        if allergen not in allergens:
                            allergens.append(allergen)
        
        return allergens
    
    def is_allergen_free(self, food_name: str, allergen: str) -> bool:
        """Check if food is free from specific allergen"""
        allergens = self.get_allergens(food_name)
        return allergen.lower() not in [a.lower() for a in allergens]