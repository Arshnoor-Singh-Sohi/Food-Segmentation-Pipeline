# Save as: src/metadata/food_type_classifier_fixed.py
"""
Fixed Food Type Classifier - Properly handles refrigerators and individual items
This fixes the critical issues with refrigerator classification
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)

class FoodTypeClassifier:
    """
    Classifies food images into two categories:
    1. Complete Dishes - Require single portion-based segmentation
    2. Individual Items - Require item-by-item segmentation with specific units
    
    FIXED: Now properly handles refrigerators and storage contexts
    """
    
    def __init__(self):
        # Define complete dishes that should be treated as single portions
        self.complete_dishes = {
            # Italian dishes
            'pizza', 'pasta', 'lasagna', 'risotto', 'carbonara', 'bolognese',
            'margherita pizza', 'pepperoni pizza', 'spaghetti carbonara',
            
            # Salads (when served as complete meals on a single plate)
            'caesar salad', 'greek salad', 'cobb salad', 'waldorf salad',
            
            # Plated meals
            'burger', 'hamburger', 'cheeseburger', 'sandwich', 'club sandwich',
            'steak dinner', 'fish and chips', 'chicken parmesan',
            
            # Asian dishes (when plated)
            'pad thai', 'fried rice', 'kung pao chicken', 'sushi roll',
            
            # Other complete meals
            'soup', 'stew', 'curry', 'burrito', 'taco plate',
            
            # Breakfast plates
            'eggs benedict', 'full breakfast', 'pancakes', 'waffles',
            
            # Single-serving desserts
            'cake slice', 'pie slice', 'tiramisu', 'cheesecake slice'
        }
        
        # Storage and container indicators - ALWAYS individual items
        self.storage_indicators = {
            'refrigerator', 'fridge', 'freezer', 'pantry', 'shelf', 
            'cabinet', 'drawer', 'storage', 'container', 'jar',
            'bottle', 'carton', 'package', 'bag', 'box'
        }
        
        # Individual items that need separate segmentation
        self.individual_items = {
            # Fruits
            'apple', 'banana', 'orange', 'grape', 'strawberry', 'blueberry',
            'mango', 'pineapple', 'watermelon', 'peach', 'pear', 'plum',
            'cherry', 'kiwi', 'lemon', 'lime', 'grapefruit', 'avocado',
            
            # Vegetables
            'carrot', 'broccoli', 'cauliflower', 'tomato', 'cucumber',
            'bell pepper', 'pepper', 'lettuce', 'spinach', 'onion', 'potato',
            'corn', 'peas', 'green beans', 'asparagus', 'celery', 'cabbage',
            
            # Dairy products
            'milk', 'cheese', 'yogurt', 'butter', 'cream', 'eggs', 'egg carton',
            
            # Beverages
            'water bottle', 'juice', 'soda', 'beer', 'wine',
            'coffee', 'tea', 'drink', 'beverage',
            
            # Packaged items
            'bread', 'cereal', 'rice', 'pasta package', 'flour',
            'sugar', 'salt', 'oil', 'vinegar', 'sauce', 'condiment',
            
            # Proteins (when not prepared)
            'raw chicken', 'raw meat', 'fish', 'tofu', 'beans',
            
            # Snacks and small items
            'cookie', 'cracker', 'chip', 'nut', 'candy', 'chocolate',
            'granola bar', 'pretzel', 'popcorn',
            
            # Containers and packages
            'jar', 'bottle', 'can', 'box', 'container', 'package',
            'carton', 'bag', 'wrapper'
        }
        
        # Context keywords that strongly indicate storage/individual items
        self.storage_context_keywords = [
            'shelf', 'stored', 'multiple', 'various', 'assorted',
            'collection', 'supplies', 'groceries', 'ingredients'
        ]
        
        # Keywords that indicate complete dishes
        self.dish_keywords = [
            'plate', 'platter', 'meal', 'dinner', 'lunch',
            'breakfast', 'combo', 'served', 'prepared'
        ]
    
    def classify_food_type(self, 
                          food_names: List[str], 
                          all_detected_items: List[str],
                          detection_count: int,
                          image_context: Optional[Dict] = None) -> Tuple[str, float]:
        """
        Classify whether the image contains a complete dish or individual items
        
        FIXED: Now considers all detected items, not just food items
        
        Args:
            food_names: List of detected food names
            all_detected_items: ALL items detected (including non-food)
            detection_count: Total number of items detected
            image_context: Optional context about the image
            
        Returns:
            Tuple of (classification, confidence)
            classification: 'complete_dish' or 'individual_items'
            confidence: 0.0 to 1.0
        """
        # CRITICAL FIX: Check for storage/refrigerator context FIRST
        all_items_lower = [item.lower() for item in all_detected_items]
        
        # Check if this is a storage context (refrigerator, pantry, etc.)
        for item in all_items_lower:
            for storage_word in self.storage_indicators:
                if storage_word in item:
                    logger.info(f"Storage context detected: {storage_word} in {item}")
                    return 'individual_items', 1.0
        
        # If we have many detections (>5), it's likely individual items
        if detection_count > 5:
            logger.info(f"Many items detected ({detection_count}), classifying as individual items")
            return 'individual_items', 0.9
        
        # If we have diverse item types, it's individual items
        if len(set(all_items_lower)) > 3:
            logger.info(f"Diverse items detected, classifying as individual items")
            return 'individual_items', 0.85
        
        # Now check food-specific classification
        dish_score = 0
        individual_score = 0
        
        # If only 1-2 items and they're complete dishes, it might be a meal
        if len(food_names) <= 2:
            for food_name in food_names:
                food_lower = food_name.lower()
                if food_lower in self.complete_dishes:
                    dish_score += 2.0
        
        # Check all food items for individual item indicators
        for food_name in food_names:
            food_lower = food_name.lower()
            
            # Check for individual items
            for item in self.individual_items:
                if item in food_lower or food_lower in item:
                    individual_score += 2.0
                    break
        
        # If we have eggs, milk, fruits, or vegetables, it's definitely individual items
        individual_indicators = ['egg', 'milk', 'banana', 'apple', 'carrot', 'bottle', 'jar']
        for food_name in food_names:
            if any(indicator in food_name.lower() for indicator in individual_indicators):
                individual_score += 3.0
        
        # Default to individual items if scores are close or zero
        if individual_score >= dish_score:
            return 'individual_items', min(1.0, 0.7 + (individual_score / 10))
        else:
            return 'complete_dish', min(1.0, 0.7 + (dish_score / 10))
    
    def should_merge_segments(self, food_type: str) -> bool:
        """
        Determine if segments should be merged based on food type
        
        Args:
            food_type: 'complete_dish' or 'individual_items'
            
        Returns:
            bool: True if segments should be merged into one
        """
        return food_type == 'complete_dish'
    
    def get_classification_explanation(self, food_type: str, confidence: float) -> str:
        """
        Get human-readable explanation of the classification
        """
        if food_type == 'complete_dish':
            return f"Complete dish detected (confidence: {confidence:.1%}). Will create single portion segment."
        else:
            return f"Individual items detected (confidence: {confidence:.1%}). Will segment each item separately."