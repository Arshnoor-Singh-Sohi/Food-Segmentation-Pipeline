
"""
Measurement Unit System - Assigns appropriate units to food items
Implements the 25 predefined units 
"""

from typing import Dict, List, Optional, Tuple
import re

class MeasurementUnitSystem:
    """
    Manages the 25 predefined measurement units for food items
    """
    
    def __init__(self):
        # Define the 25 measurement units organized by category
        self.units = {
            # Volume units (for liquids)
            'volume': {
                'ml': 'milliliters',
                'l': 'liters',
                'fl oz': 'fluid ounces',
                'cup': 'cups',
                'tbsp': 'tablespoons',
                'tsp': 'teaspoons'
            },
            
            # Weight units (for solids)
            'weight': {
                'g': 'grams',
                'kg': 'kilograms',
                'mg': 'milligrams',
                'oz': 'ounces',
                'lb': 'pounds'
            },
            
            # Count units (for discrete items)
            'count': {
                'piece': 'pieces',
                'unit': 'units',
                'item': 'items',
                'slice': 'slices',
                'serving': 'servings'
            },
            
            # Special food units
            'special': {
                'portion': 'portions',  # For complete dishes
                'bowl': 'bowls',
                'plate': 'plates',
                'scoop': 'scoops',
                'handful': 'handfuls',
                'bunch': 'bunches',
                'package': 'packages',
                'container': 'containers'
            }
        }
        
        # Create flat list of all 25 units
        self.all_units = []
        for category, units in self.units.items():
            self.all_units.extend(units.keys())
        
        # Define unit mappings for different food types
        self.food_unit_mappings = {
            # Liquids
            'milk': 'ml',
            'juice': 'ml',
            'water': 'ml',
            'coffee': 'ml',
            'tea': 'ml',
            'soda': 'ml',
            'wine': 'ml',
            'beer': 'ml',
            'soup': 'ml',
            'sauce': 'ml',
            'oil': 'ml',
            
            # Fruits (countable)
            'apple': 'piece',
            'banana': 'piece',
            'orange': 'piece',
            'grape': 'g',  # Grapes by weight
            'strawberry': 'piece',
            'blueberry': 'g',  # Small berries by weight
            
            # Vegetables
            'carrot': 'piece',
            'broccoli': 'g',
            'lettuce': 'g',
            'tomato': 'piece',
            'potato': 'piece',
            'onion': 'piece',
            
            # Grains and cereals
            'rice': 'g',
            'pasta': 'g',
            'cereal': 'g',
            'oatmeal': 'g',
            
            # Proteins
            'chicken': 'g',
            'beef': 'g',
            'fish': 'g',
            'egg': 'piece',
            'tofu': 'g',
            
            # Bread and bakery
            'bread': 'slice',
            'bagel': 'piece',
            'muffin': 'piece',
            'cookie': 'piece',
            'croissant': 'piece',
            
            # Complete dishes (always portion)
            'pizza': 'portion',
            'burger': 'portion',
            'salad': 'portion',
            'sandwich': 'portion',
            'curry': 'portion'
        }
    
    def get_unit_for_food(self, food_name: str, 
                         food_type: str,
                         physical_properties: Optional[Dict] = None) -> Tuple[str, str]:
        """
        Get the appropriate measurement unit for a food item
        
        Args:
            food_name: Name of the food item
            food_type: 'complete_dish' or 'individual_items'
            physical_properties: Optional dict with properties like is_liquid, is_countable
            
        Returns:
            Tuple of (unit_abbreviation, unit_full_name)
        """
        # Complete dishes always use portion
        if food_type == 'complete_dish':
            return 'portion', 'portions'
        
        food_lower = food_name.lower()
        
        # Check direct mapping
        for food_key, unit in self.food_unit_mappings.items():
            if food_key in food_lower:
                return unit, self._get_full_unit_name(unit)
        
        # Use physical properties if available
        if physical_properties:
            if physical_properties.get('is_liquid', False):
                return 'ml', 'milliliters'
            elif physical_properties.get('is_countable', False):
                return 'piece', 'pieces'
            elif physical_properties.get('is_powder', False):
                return 'g', 'grams'
        
        # Default rules based on keywords
        if any(liquid in food_lower for liquid in ['juice', 'milk', 'water', 'drink', 'beverage']):
            return 'ml', 'milliliters'
        elif any(countable in food_lower for countable in ['piece', 'whole', 'individual']):
            return 'piece', 'pieces'
        elif 'slice' in food_lower:
            return 'slice', 'slices'
        else:
            # Default to grams for most solid foods
            return 'g', 'grams'
    
    def _get_full_unit_name(self, unit_abbr: str) -> str:
        """Get full name for unit abbreviation"""
        for category, units in self.units.items():
            if unit_abbr in units:
                return units[unit_abbr]
        return unit_abbr  # Return as-is if not found
    
    def convert_between_units(self, value: float, from_unit: str, to_unit: str) -> Optional[float]:
        """
        Convert between compatible units
        
        Args:
            value: The numerical value to convert
            from_unit: Source unit
            to_unit: Target unit
            
        Returns:
            Converted value or None if conversion not possible
        """
        # Define conversion factors
        conversions = {
            # Volume conversions (to ml as base)
            'ml': 1,
            'l': 1000,
            'tsp': 4.92892,
            'tbsp': 14.7868,
            'fl oz': 29.5735,
            'cup': 236.588,
            
            # Weight conversions (to g as base)
            'g': 1,
            'kg': 1000,
            'mg': 0.001,
            'oz': 28.3495,
            'lb': 453.592
        }
        
        # Check if units are in the same category
        from_category = None
        to_category = None
        
        for category, units in self.units.items():
            if from_unit in units:
                from_category = category
            if to_unit in units:
                to_category = category
        
        if from_category != to_category or from_category is None:
            return None  # Can't convert between different categories
        
        # Convert through base unit
        if from_unit in conversions and to_unit in conversions:
            base_value = value * conversions[from_unit]
            return base_value / conversions[to_unit]
        
        return None
    
    def format_measurement(self, value: float, unit: str, precision: int = 1) -> str:
        """
        Format a measurement value with its unit
        
        Args:
            value: The numerical value
            unit: The unit abbreviation
            precision: Decimal places for rounding
            
        Returns:
            Formatted string like "250 ml" or "3 pieces"
        """
        # Round to appropriate precision
        if unit in ['piece', 'unit', 'item', 'slice']:
            # Count units should be integers
            value = int(round(value))
            
        # Handle pluralization for count units
        if value == 1 and unit in ['piece', 'slice', 'portion']:
            unit = unit.rstrip('s')  # Remove plural
        elif value != 1 and unit in ['piece', 'slice', 'portion']:
            if not unit.endswith('s'):
                unit += 's'  # Add plural
        
        # Format based on value
        if isinstance(value, int) or value.is_integer():
            return f"{int(value)} {unit}"
        else:
            return f"{value:.{precision}f} {unit}"