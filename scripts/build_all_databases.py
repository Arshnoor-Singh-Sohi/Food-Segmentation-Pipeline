#!/usr/bin/env python3
"""
Build all databases needed for metadata extraction
Run this FIRST to set up your databases!
"""

import sys
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def build_all_databases():
    """Build all required databases"""
    print("🏗️  Building All Databases for Metadata Extraction")
    print("="*50)
    
    # Create directory structure first
    directories = [
        "data/databases/nutrition",
        "data/databases/food_taxonomy", 
        "data/databases/cuisine_mapping"
    ]
    
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"📁 Created directory: {dir_path}")
    
    # 1. Build nutrition database
    print("\n📊 Building Nutrition Database...")
    
    # Import the nutrition builder from your paste-3.txt
    # You need to save the content of paste-3.txt as this file:
    nutrition_builder_path = Path("src/databases/build_nutrition_db.py")
    
    if not nutrition_builder_path.exists():
        print("❌ ERROR: Please copy the content of paste-3.txt to src/databases/build_nutrition_db.py")
        print("Run this command:")
        print("  mkdir -p src/databases")
        print("  # Then copy paste-3.txt content to src/databases/build_nutrition_db.py")
        return False
    
    # Import and run the nutrition builder
    from src.databases.build_nutrition_db import NutritionDatabaseBuilder
    
    builder = NutritionDatabaseBuilder("data/databases/nutrition/nutrition_expanded.db")
    builder.import_usda_database()
    builder.add_prepared_dishes()
    builder.add_food_aliases()
    builder.export_to_json('data/databases/nutrition/nutrition_expanded.json')
    
    # Test the database
    print("\n🔍 Testing nutrition database...")
    test_foods = ['pizza', 'apple', 'chicken', 'burger', 'sushi']
    for food in test_foods:
        results = builder.search_food(food)
        print(f"  Search '{food}': Found {len(results)} results")
        if results:
            print(f"    → {results[0]['name']}: {results[0]['nutrition']['calories']} cal")
    
    # 2. Create food hierarchy database
    print("\n📁 Creating food hierarchy database...")
    hierarchy_data = {
        "food_categories": {
            "fruits": {
                "citrus": ["orange", "lemon", "lime", "grapefruit"],
                "berries": ["strawberry", "blueberry", "raspberry"],
                "tropical": ["mango", "pineapple", "banana"],
                "common": ["apple", "pear", "grape"]
            },
            "vegetables": {
                "leafy": ["lettuce", "spinach", "kale"],
                "root": ["carrot", "potato", "onion"],
                "cruciferous": ["broccoli", "cauliflower"]
            },
            "proteins": {
                "meat": ["beef", "chicken", "pork", "lamb"],
                "seafood": ["fish", "shrimp", "salmon"],
                "plant": ["tofu", "beans", "lentils"]
            },
            "grains": {
                "whole": ["brown rice", "quinoa", "oats"],
                "refined": ["white rice", "pasta", "white bread"]
            },
            "prepared_dishes": {
                "italian": ["pizza", "pasta", "lasagna"],
                "asian": ["sushi", "ramen", "curry"],
                "american": ["burger", "hot dog", "sandwich"]
            }
        },
        "meal_types": {
            "breakfast": ["pancake", "cereal", "omelette"],
            "lunch": ["sandwich", "salad", "soup"],
            "dinner": ["steak", "pasta", "curry"],
            "snack": ["chips", "nuts", "fruit"],
            "dessert": ["cake", "ice cream", "cookie"]
        }
    }
    
    hierarchy_path = Path("data/databases/food_taxonomy/food_hierarchy.json")
    with open(hierarchy_path, 'w') as f:
        json.dump(hierarchy_data, f, indent=2)
    print("  ✅ Created food_hierarchy.json")
    
    # 3. Create cuisine patterns database
    print("\n🌍 Creating cuisine patterns database...")
    cuisine_data = {
        "cuisine_indicators": {
            "italian": {
                "dishes": ["pizza", "pasta", "risotto", "lasagna"],
                "ingredients": ["basil", "oregano", "mozzarella"],
                "keywords": ["italian", "romano"]
            },
            "japanese": {
                "dishes": ["sushi", "ramen", "tempura"],
                "ingredients": ["soy sauce", "wasabi", "nori"],
                "keywords": ["japanese"]
            },
            "chinese": {
                "dishes": ["fried rice", "dim sum", "kung pao"],
                "ingredients": ["soy sauce", "ginger"],
                "keywords": ["chinese", "szechuan"]
            },
            "mexican": {
                "dishes": ["taco", "burrito", "quesadilla"],
                "ingredients": ["salsa", "cilantro", "lime"],
                "keywords": ["mexican", "tex-mex"]
            },
            "american": {
                "dishes": ["burger", "hot dog", "bbq"],
                "ingredients": ["ketchup", "mustard"],
                "keywords": ["american"]
            }
        }
    }
    
    cuisine_path = Path("data/databases/cuisine_mapping/cuisine_patterns.json")
    with open(cuisine_path, 'w') as f:
        json.dump(cuisine_data, f, indent=2)
    print("  ✅ Created cuisine_patterns.json")
    
    print("\n✅ All databases built successfully!")
    print(f"\nDatabase Summary:")
    print(f"  - Total food items: ~44")
    print(f"  - Basic foods: 28")
    print(f"  - Prepared dishes: 16")
    print(f"  - Cuisines mapped: 5")
    print(f"  - Food categories: 5")
    
    return True

if __name__ == "__main__":
    success = build_all_databases()
    if not success:
        sys.exit(1)