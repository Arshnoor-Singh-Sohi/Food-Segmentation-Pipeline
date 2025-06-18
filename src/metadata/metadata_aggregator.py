"""
Core metadata extraction and aggregation module
This is the heart of the metadata labeling system
"""

import cv2
import numpy as np
import torch
from typing import Dict, List, Any, Tuple
from pathlib import Path
import json
import sqlite3
from transformers import AutoImageProcessor, AutoModelForImageClassification
import logging
<<<<<<< HEAD
=======
from src.metadata.food_type_classifier import FoodTypeClassifier
from src.metadata.measurement_units import MeasurementUnitSystem
from src.models.portion_aware_segmentation import PortionAwareSegmentation
>>>>>>> 82a126b (Complete Meal or Portion integration)

logger = logging.getLogger(__name__)

class MetadataAggregator:
    """Aggregate all metadata extraction functions"""
    
    def __init__(self, config_path="config/metadata_config.yaml"):
        import yaml
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔧 Metadata system using device: {self.device}")
        
        # Load models
        self._load_models()
        
        # Load databases
        self._load_databases()
<<<<<<< HEAD
=======

        self.food_type_classifier = FoodTypeClassifier()
        self.measurement_system = MeasurementUnitSystem()
        self.portion_segmentation = PortionAwareSegmentation(
            self.food_type_classifier,
            self.measurement_system
        )
>>>>>>> 82a126b (Complete Meal or Portion integration)
        
    def _load_models(self):
        """Load all metadata extraction models"""
        print("📦 Loading metadata models...")
        
        # Food classifier
        model_path = self.config['models']['food_classifier']['model_path']
        self.food_processor = AutoImageProcessor.from_pretrained(model_path)
        self.food_model = AutoModelForImageClassification.from_pretrained(model_path).to(self.device)
        self.food_model.eval()
        
        print("✅ Models loaded successfully")
    
    def _load_databases(self):
        """Load all databases"""
        # Nutrition database
        self.nutrition_db_path = self.config['databases']['nutrition']
        
        # Load cuisine mapping
        cuisine_path = Path(self.config['databases']['cuisine_mapping'])
        if cuisine_path.exists():
            with open(cuisine_path, 'r') as f:
                self.cuisine_mapping = json.load(f)
        else:
            # Default mapping
            self.cuisine_mapping = {
                "pizza": "Italian",
                "hamburger": "American",
                "sushi": "Japanese",
                "pasta": "Italian",
                "taco": "Mexican",
                "curry": "Indian",
                "croissant": "French",
                "pad_thai": "Thai",
                "dim_sum": "Chinese"
            }
            # Save default mapping
            cuisine_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cuisine_path, 'w') as f:
                json.dump(self.cuisine_mapping, f, indent=2)
    
    def extract_metadata(self, image_path: str, detection_results: Dict) -> Dict[str, Any]:
        """
        Extract complete metadata for all detected food items
        
        Args:
            image_path: Path to the original image
            detection_results: Results from YOLO detection/segmentation
            
        Returns:
            Enhanced results with metadata
        """
        # Load image
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process each detected food item
<<<<<<< HEAD
        enriched_items = []
        
        for item in detection_results.get('food_items', []):
            if item.get('is_food', False):
                # Extract crop for this item
                bbox = item['bbox']
                crop = self._extract_crop(image_rgb, bbox)
                
                # Extract all metadata
                metadata = self._extract_item_metadata(crop, item, image_rgb.shape)
                
                # Merge with original detection
                enriched_item = {**item, **metadata}
                enriched_items.append(enriched_item)
        
        # Create enhanced output
        enhanced_results = {
            'original_results': detection_results,
            'enriched_items': enriched_items,
            'meal_summary': self._generate_meal_summary(enriched_items),
            'total_nutrition': self._calculate_total_nutrition(enriched_items),
            'metadata_version': '1.0'
        }
        
        return enhanced_results
    
=======
        segmentation_results = self.portion_segmentation.process_segmentation(
        detection_results, image_rgb
        )
        
        # Then process metadata for each segment
        enriched_segments = []
        
        for segment in segmentation_results['segments']:
            # Extract metadata based on segment type
            if segment['type'] == 'complete_dish':
                metadata = self._extract_dish_metadata(segment, image_rgb)
            else:
                metadata = self._extract_item_metadata(segment, image_rgb)
            
            # Merge segment info with metadata
            enriched_segment = {**segment, **metadata}
            enriched_segments.append(enriched_segment)
        
        # Create final output following CEO's requirements
        enhanced_results = {
            'original_results': detection_results,
            'segmentation_type': segmentation_results['food_type_classification']['type'],
            'segmentation_confidence': segmentation_results['food_type_classification']['confidence'],
            'segments': enriched_segments,
            'measurement_summary': segmentation_results['measurement_summary'],
            'meal_summary': self._generate_meal_summary(enriched_segments),
            'total_nutrition': self._calculate_total_nutrition(enriched_segments)
        }
        
        return enhanced_results
        
>>>>>>> 82a126b (Complete Meal or Portion integration)
    def _extract_crop(self, image: np.ndarray, bbox: Dict) -> np.ndarray:
        """Extract crop from image using bbox"""
        x1, y1 = int(bbox['x1']), int(bbox['y1'])
        x2, y2 = int(bbox['x2']), int(bbox['y2'])
        
        # Ensure bounds are valid
        h, w = image.shape[:2]
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)
        
        return image[y1:y2, x1:x2]
    
    def _extract_item_metadata(self, crop: np.ndarray, item: Dict, image_shape: Tuple) -> Dict:
        """Extract all metadata for a single food item"""
        metadata = {}
        
        # 1. Detailed food classification
        food_class, food_conf = self._classify_food(crop)
        metadata['detailed_food_type'] = food_class
        metadata['classification_confidence'] = food_conf
        
        # 2. Cuisine identification
        metadata['cuisine'] = self._identify_cuisine(food_class)
        
        # 3. Nutritional information
        metadata['nutrition'] = self._get_nutrition_info(food_class, item)
        
        # 4. Portion estimation
        metadata['portion'] = self._estimate_portion(item, food_class, image_shape)
        
        # 5. Ingredients
        metadata['ingredients'] = self._identify_ingredients(food_class)
        
        # 6. Allergens
        metadata['allergens'] = self._identify_allergens(food_class)
        
        # 7. Dietary tags
        metadata['dietary_tags'] = self._get_dietary_tags(food_class, metadata['ingredients'])
        
        # 8. Preparation method
        metadata['preparation_method'] = self._estimate_preparation(food_class, crop)
        
        return metadata
    
    def _classify_food(self, crop: np.ndarray) -> Tuple[str, float]:
        """Classify food using Food-101 model"""
        # Preprocess
        inputs = self.food_processor(images=crop, return_tensors="pt").to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.food_model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            
        # Get top prediction
        top_prob, top_idx = torch.max(probs, dim=-1)
        food_class = self.food_model.config.id2label[top_idx.item()]
        confidence = top_prob.item()
        
        # Clean up class name (Food-101 uses underscores)
        food_class = food_class.replace('_', ' ')
        
        return food_class, confidence
    
    def _identify_cuisine(self, food_type: str) -> str:
        """Identify cuisine based on food type"""
        # Check direct mapping
        for food, cuisine in self.cuisine_mapping.items():
            if food.lower() in food_type.lower():
                return cuisine
        
        # Pattern matching for cuisine keywords
        cuisine_patterns = {
            'Italian': ['pasta', 'pizza', 'risotto', 'lasagna', 'tiramisu'],
            'Japanese': ['sushi', 'ramen', 'tempura', 'miso', 'sashimi'],
            'Chinese': ['dim sum', 'fried rice', 'dumpling', 'noodle', 'wonton'],
            'Mexican': ['taco', 'burrito', 'nachos', 'enchilada', 'guacamole'],
            'Indian': ['curry', 'naan', 'samosa', 'biryani', 'tikka'],
            'French': ['croissant', 'crepe', 'baguette', 'quiche', 'macaron'],
            'Thai': ['pad thai', 'tom yum', 'green curry', 'satay'],
            'American': ['hamburger', 'hot dog', 'barbecue', 'pancake', 'donut']
        }
        
        for cuisine, patterns in cuisine_patterns.items():
            for pattern in patterns:
                if pattern in food_type.lower():
                    return cuisine
        
        return "International"
    
    def _get_nutrition_info(self, food_type: str, item: Dict) -> Dict:
        """Get nutritional information from database"""
        # Connect to database
        conn = sqlite3.connect(self.nutrition_db_path)
        cursor = conn.cursor()
        
        # Try exact match first
        cursor.execute("""
            SELECT n.* FROM nutrition n
            JOIN foods f ON n.food_id = f.id
            WHERE LOWER(f.name) = LOWER(?)
        """, (food_type,))
        
        result = cursor.fetchone()
        
        if not result:
            # Try partial match
            cursor.execute("""
                SELECT n.* FROM nutrition n
                JOIN foods f ON n.food_id = f.id
                WHERE LOWER(f.name) LIKE LOWER(?)
                LIMIT 1
            """, (f"%{food_type.split()[0]}%",))
            result = cursor.fetchone()
        
        conn.close()
        
        if result:
            # Scale by portion size
            portion_scale = item.get('portion_estimate', {}).get('estimated_grams', 100) / 100
            
            return {
                'calories': round(result[1] * portion_scale, 1),
                'protein_g': round(result[2] * portion_scale, 1),
                'carbs_g': round(result[3] * portion_scale, 1),
                'fat_g': round(result[4] * portion_scale, 1),
                'fiber_g': round(result[5] * portion_scale, 1),
                'sugar_g': round(result[6] * portion_scale, 1),
                'sodium_mg': round(result[7] * portion_scale, 1)
            }
        else:
            # Default values
            return {
                'calories': 200,
                'protein_g': 10,
                'carbs_g': 25,
                'fat_g': 8,
                'fiber_g': 2,
                'sugar_g': 5,
                'sodium_mg': 300,
                'note': 'Estimated values'
            }
    
    def _estimate_portion(self, item: Dict, food_type: str, image_shape: Tuple) -> Dict:
        """Estimate portion size"""
        # Get mask area
        mask_info = item.get('mask_info', {})
        area_percentage = mask_info.get('area_percentage', 0.1)
        
        # Reference sizes (assuming average plate is 25cm diameter)
        image_area_cm2 = 490  # Approximate for typical food photo
        food_area_cm2 = image_area_cm2 * (area_percentage / 100)
        
        # Density factors
        density_factors = self.config['models']['portion_estimator']['density_factors']
        
        # Find appropriate density
        density = density_factors.get('default', 1.0)
        for food_key, factor in density_factors.items():
            if food_key in food_type.lower():
                density = factor
                break
        
        # Estimate weight
        estimated_weight = food_area_cm2 * density * 0.5  # Thickness factor
        
        # Determine serving size
        if estimated_weight < 50:
            serving = "small (snack size)"
        elif estimated_weight < 150:
            serving = "medium (single serving)"
        elif estimated_weight < 300:
            serving = "large (full meal)"
        else:
            serving = "extra large (sharing size)"
        
        return {
            'estimated_weight_g': round(estimated_weight, 1),
            'serving_description': serving,
            'confidence': 'medium',
            'area_cm2': round(food_area_cm2, 1)
        }
    
    def _identify_ingredients(self, food_type: str) -> List[str]:
        """Identify likely ingredients"""
        # Common ingredients database
        ingredient_map = {
            'pizza': ['dough', 'tomato sauce', 'cheese', 'herbs'],
            'hamburger': ['bun', 'beef patty', 'lettuce', 'tomato', 'cheese'],
            'sushi': ['rice', 'fish', 'seaweed', 'vegetables'],
            'pasta': ['pasta', 'sauce', 'cheese', 'herbs'],
            'salad': ['lettuce', 'vegetables', 'dressing'],
            'sandwich': ['bread', 'meat', 'cheese', 'vegetables'],
            'curry': ['sauce', 'vegetables', 'meat', 'rice', 'spices'],
            'steak': ['beef', 'seasoning'],
            'chicken': ['chicken', 'seasoning'],
            'fish': ['fish', 'lemon', 'herbs']
        }
        
        # Find matching ingredients
        ingredients = []
        for key, value in ingredient_map.items():
            if key in food_type.lower():
                ingredients.extend(value)
                break
        
        # If no match, use generic
        if not ingredients:
            ingredients = ['main ingredient', 'seasoning', 'garnish']
        
        return ingredients
    
    def _identify_allergens(self, food_type: str) -> List[str]:
        """Identify potential allergens"""
        allergen_map = {
            'gluten': ['pizza', 'pasta', 'bread', 'sandwich', 'burger', 'cake', 'donut'],
            'dairy': ['pizza', 'cheese', 'ice cream', 'cake', 'cream'],
            'nuts': ['pad thai', 'baklava', 'brownie', 'pesto'],
            'seafood': ['sushi', 'shrimp', 'fish', 'lobster'],
            'eggs': ['cake', 'mayonnaise', 'pasta', 'tempura'],
            'soy': ['sushi', 'tofu', 'edamame', 'soy sauce']
        }
        
        allergens = []
        for allergen, foods in allergen_map.items():
            for food in foods:
                if food in food_type.lower():
                    allergens.append(allergen)
                    break
        
        return list(set(allergens))  # Remove duplicates
    
    def _get_dietary_tags(self, food_type: str, ingredients: List[str]) -> List[str]:
        """Get dietary tags"""
        tags = []
        
        # Check for vegetarian/vegan
        meat_keywords = ['beef', 'chicken', 'pork', 'fish', 'meat', 'bacon', 'ham']
        dairy_keywords = ['cheese', 'milk', 'cream', 'butter']
        
        has_meat = any(meat in ' '.join(ingredients).lower() for meat in meat_keywords)
        has_dairy = any(dairy in ' '.join(ingredients).lower() for dairy in dairy_keywords)
        
        if not has_meat and not has_dairy:
            tags.append('vegan')
            tags.append('vegetarian')
        elif not has_meat:
            tags.append('vegetarian')
        
        # Other tags
        if 'salad' in food_type.lower():
            tags.append('low-calorie')
        
        if any(grain in food_type.lower() for grain in ['rice', 'quinoa', 'corn']):
            tags.append('gluten-free')
        
        return tags
    
    def _estimate_preparation(self, food_type: str, image: np.ndarray) -> str:
        """Estimate preparation method"""
        # Simple heuristic based on food type and appearance
        preparation_map = {
            'grilled': ['steak', 'chicken', 'fish', 'vegetables'],
            'fried': ['chicken', 'tempura', 'french fries', 'donut'],
            'baked': ['pizza', 'bread', 'cake', 'lasagna'],
            'steamed': ['dumplings', 'vegetables', 'fish'],
            'raw': ['sushi', 'salad', 'fruit'],
            'boiled': ['pasta', 'eggs', 'vegetables'],
            'roasted': ['chicken', 'vegetables', 'meat']
        }
        
        for method, foods in preparation_map.items():
            for food in foods:
                if food in food_type.lower():
                    return method
        
        return 'cooked'
    
    def _generate_meal_summary(self, items: List[Dict]) -> Dict:
        """Generate meal summary from all items"""
        if not items:
            return {'meal_type': 'empty', 'total_items': 0}
        
        # Analyze meal composition
        cuisines = [item.get('cuisine', 'unknown') for item in items]
        main_cuisine = max(set(cuisines), key=cuisines.count)
        
        total_calories = sum(item.get('nutrition', {}).get('calories', 0) for item in items)
        
        # Determine meal type
        if total_calories < 300:
            meal_type = 'snack'
        elif total_calories < 600:
            meal_type = 'light meal'
        elif total_calories < 900:
            meal_type = 'regular meal'
        else:
            meal_type = 'large meal'
        
        # Dietary analysis
        all_tags = []
        for item in items:
            all_tags.extend(item.get('dietary_tags', []))
        
        return {
            'meal_type': meal_type,
            'main_cuisine': main_cuisine,
            'total_items': len(items),
            'total_calories': round(total_calories, 1),
            'dietary_friendly': list(set(all_tags)),
            'cuisines_present': list(set(cuisines))
        }
    
    def _calculate_total_nutrition(self, items: List[Dict]) -> Dict:
        """Calculate total nutrition for all items"""
        totals = {
            'calories': 0,
            'protein_g': 0,
            'carbs_g': 0,
            'fat_g': 0,
            'fiber_g': 0,
            'sugar_g': 0,
            'sodium_mg': 0
        }
        
        for item in items:
            nutrition = item.get('nutrition', {})
            for key in totals:
                totals[key] += nutrition.get(key, 0)
        
        # Round all values
        return {k: round(v, 1) for k, v in totals.items()}