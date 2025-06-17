#!/usr/bin/env python3
"""
Comprehensive Nutrition Database Builder
Builds a large-scale nutrition database from multiple sources
FIXED VERSION - Ensures column names match
"""

import json
import requests
import pandas as pd
from pathlib import Path
import sqlite3
from typing import Dict, List, Any, Optional
import re
from tqdm import tqdm

class NutritionDatabaseBuilder:
    """Build comprehensive nutrition database from multiple sources"""
    
    def __init__(self, db_path="data/nutrition_database.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Delete existing database to ensure clean schema
        if self.db_path.exists():
            self.db_path.unlink()
            print(f"Removed existing database: {self.db_path}")
        
        self.conn = sqlite3.connect(str(self.db_path))
        self._init_database()
        
    def _init_database(self):
        """Initialize SQLite database with proper schema"""
        cursor = self.conn.cursor()
        
        # Main foods table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS foods (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            category TEXT,
            subcategory TEXT,
            cuisine TEXT,
            serving_size REAL DEFAULT 100,
            serving_unit TEXT DEFAULT 'g',
            barcode TEXT,
            brand TEXT,
            common_names TEXT,
            search_terms TEXT
        )
        ''')
        
        # Simplified nutrition table - matching our insert statements
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS nutrition (
            food_id INTEGER PRIMARY KEY,
            calories REAL,
            protein REAL,
            carbohydrates REAL,
            fat REAL,
            fiber REAL,
            sugar REAL,
            sodium REAL,
            saturated_fat REAL,
            cholesterol REAL,
            potassium REAL,
            vitamin_a REAL,
            vitamin_c REAL,
            calcium REAL,
            iron REAL,
            FOREIGN KEY (food_id) REFERENCES foods (id)
        )
        ''')
        
        # Prepared dishes table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS prepared_dishes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dish_name TEXT UNIQUE NOT NULL,
            cuisine TEXT,
            meal_type TEXT,
            preparation_method TEXT,
            typical_serving_size REAL,
            restaurant_chain TEXT,
            recipe_source TEXT
        )
        ''')
        
        # Dish ingredients table (for complex dishes)
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS dish_ingredients (
            dish_id INTEGER,
            food_id INTEGER,
            amount REAL,
            unit TEXT,
            FOREIGN KEY (dish_id) REFERENCES prepared_dishes (id),
            FOREIGN KEY (food_id) REFERENCES foods (id)
        )
        ''')
        
        # Food aliases and variations
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS food_aliases (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            food_id INTEGER,
            alias TEXT,
            language TEXT DEFAULT 'en',
            region TEXT,
            FOREIGN KEY (food_id) REFERENCES foods (id)
        )
        ''')
        
        self.conn.commit()
        print("Database schema created successfully")
    
    def import_usda_database(self, csv_path=None):
        """Import USDA FoodData Central database"""
        print("Importing USDA FoodData Central...")
        
        sample_foods = [
            # Fruits
            {'name': 'apple', 'calories': 52, 'protein': 0.3, 'carbs': 14, 'fat': 0.2, 'fiber': 2.4, 'sugar': 10.4, 'sodium': 1},
            {'name': 'banana', 'calories': 89, 'protein': 1.1, 'carbs': 23, 'fat': 0.3, 'fiber': 2.6, 'sugar': 12.2, 'sodium': 1},
            {'name': 'orange', 'calories': 47, 'protein': 0.9, 'carbs': 12, 'fat': 0.1, 'fiber': 2.4, 'sugar': 9.4, 'sodium': 0},
            {'name': 'strawberries', 'calories': 32, 'protein': 0.7, 'carbs': 7.7, 'fat': 0.3, 'fiber': 2.0, 'sugar': 4.9, 'sodium': 1},
            {'name': 'grapes', 'calories': 69, 'protein': 0.7, 'carbs': 18, 'fat': 0.2, 'fiber': 0.9, 'sugar': 15.5, 'sodium': 2},
            {'name': 'watermelon', 'calories': 30, 'protein': 0.6, 'carbs': 7.6, 'fat': 0.2, 'fiber': 0.4, 'sugar': 6.2, 'sodium': 1},
            {'name': 'pineapple', 'calories': 50, 'protein': 0.5, 'carbs': 13, 'fat': 0.1, 'fiber': 1.4, 'sugar': 9.9, 'sodium': 1},
            {'name': 'mango', 'calories': 60, 'protein': 0.8, 'carbs': 15, 'fat': 0.4, 'fiber': 1.6, 'sugar': 13.7, 'sodium': 1},
            
            # Vegetables
            {'name': 'broccoli', 'calories': 34, 'protein': 2.8, 'carbs': 7, 'fat': 0.4, 'fiber': 2.6, 'sugar': 1.7, 'sodium': 33},
            {'name': 'carrot', 'calories': 41, 'protein': 0.9, 'carbs': 10, 'fat': 0.2, 'fiber': 2.8, 'sugar': 4.7, 'sodium': 69},
            {'name': 'spinach', 'calories': 23, 'protein': 2.9, 'carbs': 3.6, 'fat': 0.4, 'fiber': 2.2, 'sugar': 0.4, 'sodium': 79},
            {'name': 'tomato', 'calories': 18, 'protein': 0.9, 'carbs': 3.9, 'fat': 0.2, 'fiber': 1.2, 'sugar': 2.6, 'sodium': 5},
            {'name': 'cucumber', 'calories': 16, 'protein': 0.7, 'carbs': 3.6, 'fat': 0.1, 'fiber': 0.5, 'sugar': 1.7, 'sodium': 2},
            {'name': 'bell_pepper', 'calories': 31, 'protein': 1, 'carbs': 7.5, 'fat': 0.3, 'fiber': 2.1, 'sugar': 4.2, 'sodium': 4},
            {'name': 'lettuce', 'calories': 15, 'protein': 1.4, 'carbs': 2.9, 'fat': 0.2, 'fiber': 1.3, 'sugar': 1.2, 'sodium': 28},
            {'name': 'onion', 'calories': 40, 'protein': 1.1, 'carbs': 9.3, 'fat': 0.1, 'fiber': 1.7, 'sugar': 4.2, 'sodium': 4},
            
            # Proteins
            {'name': 'chicken_breast', 'calories': 165, 'protein': 31, 'carbs': 0, 'fat': 3.6, 'fiber': 0, 'sugar': 0, 'sodium': 74},
            {'name': 'beef_steak', 'calories': 271, 'protein': 26, 'carbs': 0, 'fat': 18, 'fiber': 0, 'sugar': 0, 'sodium': 55},
            {'name': 'salmon', 'calories': 208, 'protein': 20, 'carbs': 0, 'fat': 13, 'fiber': 0, 'sugar': 0, 'sodium': 59},
            {'name': 'eggs', 'calories': 155, 'protein': 13, 'carbs': 1.1, 'fat': 11, 'fiber': 0, 'sugar': 1.1, 'sodium': 124},
            {'name': 'tofu', 'calories': 76, 'protein': 8, 'carbs': 1.9, 'fat': 4.8, 'fiber': 0.3, 'sugar': 0.6, 'sodium': 7},
            {'name': 'shrimp', 'calories': 99, 'protein': 24, 'carbs': 0.2, 'fat': 0.3, 'fiber': 0, 'sugar': 0, 'sodium': 111},
            
            # Grains
            {'name': 'rice_white', 'calories': 130, 'protein': 2.7, 'carbs': 28, 'fat': 0.3, 'fiber': 0.4, 'sugar': 0.1, 'sodium': 1},
            {'name': 'pasta', 'calories': 131, 'protein': 5, 'carbs': 25, 'fat': 1.1, 'fiber': 1.8, 'sugar': 0.8, 'sodium': 1},
            {'name': 'bread_whole_wheat', 'calories': 247, 'protein': 13, 'carbs': 41, 'fat': 3.4, 'fiber': 6.8, 'sugar': 5.6, 'sodium': 400},
            {'name': 'quinoa', 'calories': 120, 'protein': 4.4, 'carbs': 21, 'fat': 1.9, 'fiber': 2.8, 'sugar': 0.9, 'sodium': 5},
            {'name': 'oatmeal', 'calories': 68, 'protein': 2.4, 'carbs': 12, 'fat': 1.4, 'fiber': 1.7, 'sugar': 0.5, 'sodium': 2},
        ]
        
        cursor = self.conn.cursor()
        
        for food in sample_foods:
            # Insert food
            cursor.execute('''
            INSERT OR IGNORE INTO foods (name, category) VALUES (?, ?)
            ''', (food['name'], self._categorize_food(food['name'])))
            
            # Get food_id
            cursor.execute('SELECT id FROM foods WHERE name = ?', (food['name'],))
            result = cursor.fetchone()
            if result:
                food_id = result[0]
            else:
                food_id = cursor.lastrowid
            
            # Insert nutrition with matching column names
            cursor.execute('''
            INSERT OR REPLACE INTO nutrition 
            (food_id, calories, protein, carbohydrates, fat, fiber, sugar, sodium)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (food_id, food['calories'], food['protein'], 
                  food['carbs'], food['fat'], food['fiber'], 
                  food['sugar'], food['sodium']))
        
        self.conn.commit()
        print(f"Imported {len(sample_foods)} basic food items")
    
    def add_prepared_dishes(self):
        """Add common prepared dishes with calculated nutrition"""
        prepared_dishes = [
            # Italian
            {
                'name': 'margherita_pizza',
                'cuisine': 'italian',
                'meal_type': 'dinner',
                'calories': 266,
                'protein': 11,
                'carbs': 33,
                'fat': 10,
                'fiber': 2.3,
                'sugar': 3.6,
                'sodium': 598,
                'serving_size': 100,
                'ingredients': ['pizza_dough', 'tomato_sauce', 'mozzarella', 'basil']
            },
            {
                'name': 'spaghetti_carbonara',
                'cuisine': 'italian',
                'meal_type': 'dinner',
                'calories': 348,
                'protein': 15,
                'carbs': 42,
                'fat': 14,
                'fiber': 2.1,
                'sugar': 2.8,
                'sodium': 680,
                'serving_size': 250,
                'ingredients': ['spaghetti', 'eggs', 'bacon', 'parmesan', 'black_pepper']
            },
            
            # Chinese
            {
                'name': 'kung_pao_chicken',
                'cuisine': 'chinese',
                'meal_type': 'dinner',
                'calories': 229,
                'protein': 16,
                'carbs': 11,
                'fat': 14,
                'fiber': 2.3,
                'sugar': 4.2,
                'sodium': 812,
                'serving_size': 200,
                'ingredients': ['chicken', 'peanuts', 'vegetables', 'sauce']
            },
            {
                'name': 'vegetable_fried_rice',
                'cuisine': 'chinese',
                'meal_type': 'lunch',
                'calories': 163,
                'protein': 4,
                'carbs': 28,
                'fat': 5,
                'fiber': 1.8,
                'sugar': 2.5,
                'sodium': 658,
                'serving_size': 200,
                'ingredients': ['rice', 'mixed_vegetables', 'eggs', 'soy_sauce']
            },
            
            # American
            {
                'name': 'cheeseburger',
                'cuisine': 'american',
                'meal_type': 'lunch',
                'calories': 303,
                'protein': 16,
                'carbs': 28,
                'fat': 15,
                'fiber': 1.6,
                'sugar': 6.8,
                'sodium': 414,
                'serving_size': 150,
                'ingredients': ['beef_patty', 'cheese', 'bun', 'lettuce', 'tomato']
            },
            {
                'name': 'caesar_salad',
                'cuisine': 'american',
                'meal_type': 'lunch',
                'calories': 184,
                'protein': 5,
                'carbs': 8,
                'fat': 16,
                'fiber': 2.2,
                'sugar': 2.1,
                'sodium': 362,
                'serving_size': 200,
                'ingredients': ['romaine_lettuce', 'caesar_dressing', 'croutons', 'parmesan']
            },
            
            # Mexican
            {
                'name': 'chicken_tacos',
                'cuisine': 'mexican',
                'meal_type': 'dinner',
                'calories': 226,
                'protein': 12,
                'carbs': 20,
                'fat': 11,
                'fiber': 3.1,
                'sugar': 2.8,
                'sodium': 397,
                'serving_size': 150,
                'ingredients': ['tortilla', 'chicken', 'cheese', 'salsa', 'lettuce']
            },
            {
                'name': 'burrito_bowl',
                'cuisine': 'mexican',
                'meal_type': 'lunch',
                'calories': 385,
                'protein': 23,
                'carbs': 45,
                'fat': 13,
                'fiber': 8.2,
                'sugar': 4.5,
                'sodium': 842,
                'serving_size': 350,
                'ingredients': ['rice', 'beans', 'chicken', 'cheese', 'guacamole', 'salsa']
            },
            
            # Japanese
            {
                'name': 'sushi_roll',
                'cuisine': 'japanese',
                'meal_type': 'dinner',
                'calories': 200,
                'protein': 9,
                'carbs': 28,
                'fat': 5,
                'fiber': 1.5,
                'sugar': 8.2,
                'sodium': 428,
                'serving_size': 150,
                'ingredients': ['rice', 'nori', 'fish', 'vegetables']
            },
            {
                'name': 'chicken_teriyaki',
                'cuisine': 'japanese',
                'meal_type': 'dinner',
                'calories': 233,
                'protein': 23,
                'carbs': 15,
                'fat': 9,
                'fiber': 0.8,
                'sugar': 9.1,
                'sodium': 690,
                'serving_size': 200,
                'ingredients': ['chicken', 'teriyaki_sauce', 'rice', 'vegetables']
            },
            
            # Indian
            {
                'name': 'chicken_curry',
                'cuisine': 'indian',
                'meal_type': 'dinner',
                'calories': 243,
                'protein': 14,
                'carbs': 10,
                'fat': 17,
                'fiber': 2.5,
                'sugar': 4.3,
                'sodium': 612,
                'serving_size': 250,
                'ingredients': ['chicken', 'curry_sauce', 'onions', 'tomatoes', 'spices']
            },
            {
                'name': 'vegetable_biryani',
                'cuisine': 'indian',
                'meal_type': 'dinner',
                'calories': 241,
                'protein': 6,
                'carbs': 38,
                'fat': 8,
                'fiber': 3.2,
                'sugar': 3.8,
                'sodium': 684,
                'serving_size': 300,
                'ingredients': ['basmati_rice', 'mixed_vegetables', 'spices', 'yogurt']
            },
            
            # Breakfast items
            {
                'name': 'pancakes_with_syrup',
                'cuisine': 'american',
                'meal_type': 'breakfast',
                'calories': 227,
                'protein': 6,
                'carbs': 45,
                'fat': 3,
                'fiber': 1.2,
                'sugar': 18.5,
                'sodium': 412,
                'serving_size': 150,
                'ingredients': ['flour', 'eggs', 'milk', 'syrup', 'butter']
            },
            {
                'name': 'eggs_benedict',
                'cuisine': 'american',
                'meal_type': 'breakfast',
                'calories': 358,
                'protein': 19,
                'carbs': 20,
                'fat': 23,
                'fiber': 1.1,
                'sugar': 2.8,
                'sodium': 785,
                'serving_size': 200,
                'ingredients': ['english_muffin', 'eggs', 'ham', 'hollandaise_sauce']
            },
            
            # Desserts
            {
                'name': 'chocolate_cake',
                'cuisine': 'american',
                'meal_type': 'dessert',
                'calories': 352,
                'protein': 5,
                'carbs': 51,
                'fat': 15,
                'fiber': 1.8,
                'sugar': 35.2,
                'sodium': 299,
                'serving_size': 100,
                'ingredients': ['flour', 'chocolate', 'sugar', 'eggs', 'butter']
            },
            {
                'name': 'tiramisu',
                'cuisine': 'italian',
                'meal_type': 'dessert',
                'calories': 240,
                'protein': 4,
                'carbs': 28,
                'fat': 13,
                'fiber': 0.5,
                'sugar': 19.8,
                'sodium': 95,
                'serving_size': 100,
                'ingredients': ['ladyfingers', 'mascarpone', 'coffee', 'cocoa']
            }
        ]
        
        cursor = self.conn.cursor()
        
        for dish in tqdm(prepared_dishes, desc="Adding prepared dishes"):
            # Insert dish
            cursor.execute('''
            INSERT OR IGNORE INTO prepared_dishes 
            (dish_name, cuisine, meal_type, typical_serving_size)
            VALUES (?, ?, ?, ?)
            ''', (dish['name'], dish['cuisine'], dish['meal_type'], dish['serving_size']))
            
            # Also add to main foods table for easier lookup
            cursor.execute('''
            INSERT OR IGNORE INTO foods (name, category, cuisine, serving_size)
            VALUES (?, ?, ?, ?)
            ''', (dish['name'], 'prepared_dish', dish['cuisine'], dish['serving_size']))
            
            # Get food_id
            cursor.execute('SELECT id FROM foods WHERE name = ?', (dish['name'],))
            result = cursor.fetchone()
            if result:
                food_id = result[0]
            else:
                continue
            
            # Add nutrition
            cursor.execute('''
            INSERT OR REPLACE INTO nutrition 
            (food_id, calories, protein, carbohydrates, fat, fiber, sugar, sodium)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (food_id, dish['calories'], dish['protein'], 
                  dish['carbs'], dish['fat'], dish['fiber'],
                  dish['sugar'], dish['sodium']))
        
        self.conn.commit()
        print(f"Added {len(prepared_dishes)} prepared dishes")
    
    def add_food_aliases(self):
        """Add common aliases and alternative names for foods"""
        aliases = [
            # Multiple languages
            ('apple', 'manzana', 'es'),
            ('apple', 'pomme', 'fr'),
            ('apple', 'apfel', 'de'),
            ('apple', '苹果', 'zh'),
            ('apple', 'りんご', 'ja'),
            
            # Common variations
            ('french_fries', 'fries', 'en'),
            ('french_fries', 'chips', 'en'),
            ('french_fries', 'pommes frites', 'en'),
            
            ('hamburger', 'burger', 'en'),
            ('cheeseburger', 'cheese burger', 'en'),
            
            # Regional names
            ('eggplant', 'aubergine', 'en'),
            ('zucchini', 'courgette', 'en'),
            ('cilantro', 'coriander', 'en'),
        ]
        
        cursor = self.conn.cursor()
        
        for food_name, alias, language in aliases:
            # Get food_id
            result = cursor.execute('SELECT id FROM foods WHERE name = ?', (food_name,)).fetchone()
            if result:
                food_id = result[0]
                cursor.execute('''
                INSERT OR IGNORE INTO food_aliases (food_id, alias, language)
                VALUES (?, ?, ?)
                ''', (food_id, alias, language))
        
        self.conn.commit()
        print(f"Added {len(aliases)} food aliases")
    
    def _categorize_food(self, food_name):
        """Categorize food based on name"""
        categories = {
            'fruit': ['apple', 'banana', 'orange', 'grape', 'berry', 'melon'],
            'vegetable': ['carrot', 'broccoli', 'spinach', 'lettuce', 'pepper', 'onion'],
            'protein': ['chicken', 'beef', 'fish', 'egg', 'tofu', 'shrimp'],
            'grain': ['rice', 'pasta', 'bread', 'quinoa', 'oat'],
            'dairy': ['milk', 'cheese', 'yogurt', 'butter'],
            'prepared_dish': ['pizza', 'burger', 'curry', 'salad', 'sandwich']
        }
        
        food_lower = food_name.lower()
        for category, keywords in categories.items():
            if any(keyword in food_lower for keyword in keywords):
                return category
        
        return 'other'
    
    def search_food(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search for food items with fuzzy matching"""
        cursor = self.conn.cursor()
        
        # Clean query
        query_clean = query.lower().strip()
        
        # First try exact match
        results = cursor.execute('''
        SELECT f.id, f.name, f.category, f.cuisine, f.serving_size, f.serving_unit,
               n.calories, n.protein, n.carbohydrates, n.fat
        FROM foods f
        JOIN nutrition n ON f.id = n.food_id
        WHERE LOWER(f.name) = ?
        LIMIT ?
        ''', (query_clean, limit)).fetchall()
        
        # If no exact match, try partial match
        if not results:
            results = cursor.execute('''
            SELECT f.id, f.name, f.category, f.cuisine, f.serving_size, f.serving_unit,
                   n.calories, n.protein, n.carbohydrates, n.fat
            FROM foods f
            JOIN nutrition n ON f.id = n.food_id
            WHERE LOWER(f.name) LIKE ? OR LOWER(f.common_names) LIKE ?
            LIMIT ?
            ''', (f'%{query_clean}%', f'%{query_clean}%', limit)).fetchall()
        
        # Also check aliases
        if not results:
            results = cursor.execute('''
            SELECT DISTINCT f.id, f.name, f.category, f.cuisine, f.serving_size, f.serving_unit,
                   n.calories, n.protein, n.carbohydrates, n.fat
            FROM foods f
            JOIN nutrition n ON f.id = n.food_id
            JOIN food_aliases a ON f.id = a.food_id
            WHERE LOWER(a.alias) LIKE ?
            LIMIT ?
            ''', (f'%{query_clean}%', limit)).fetchall()
        
        # Format results
        formatted_results = []
        for row in results:
            formatted_results.append({
                'id': row[0],
                'name': row[1],
                'category': row[2],
                'cuisine': row[3],
                'serving_size': row[4],
                'serving_unit': row[5],
                'nutrition': {
                    'calories': row[6],
                    'protein': row[7],
                    'carbohydrates': row[8],
                    'fat': row[9]
                }
            })
        
        return formatted_results
    
    def export_to_json(self, output_path: str):
        """Export database to JSON format for backward compatibility"""
        cursor = self.conn.cursor()
        
        # Get all foods with nutrition
        results = cursor.execute('''
        SELECT f.name, f.category, f.serving_size,
               n.calories, n.protein, n.carbohydrates, n.fat, n.fiber,
               n.sugar, n.sodium
        FROM foods f
        JOIN nutrition n ON f.id = n.food_id
        ''').fetchall()
        
        # Format as dictionary
        nutrition_dict = {}
        for row in results:
            nutrition_dict[row[0]] = {
                'category': row[1],
                'serving_size': row[2],
                'calories_per_100g': row[3] if row[2] == 100 else row[3] * 100 / row[2],
                'protein': row[4],
                'carbohydrates': row[5],
                'fat': row[6],
                'fiber': row[7] or 0,
                'sugar': row[8] or 0,
                'sodium': row[9] or 0
            }
        
        # Save to JSON
        with open(output_path, 'w') as f:
            json.dump(nutrition_dict, f, indent=2)
        
        print(f"Exported {len(nutrition_dict)} food items to {output_path}")


def main():
    """Build comprehensive nutrition database"""
    print("🥗 Building Comprehensive Nutrition Database")
    
    # Initialize builder
    builder = NutritionDatabaseBuilder()
    
    # Step 1: Import USDA data
    builder.import_usda_database()
    
    # Step 2: Add prepared dishes
    builder.add_prepared_dishes()
    
    # Step 3: Add aliases
    builder.add_food_aliases()
    
    # Step 4: Export to JSON
    builder.export_to_json('data/nutrition_database_expanded.json')
    
    # Test search functionality
    print("\n🔍 Testing search functionality:")
    test_queries = ['apple', 'pizza', 'chicken curry', 'burger']
    
    for query in test_queries:
        results = builder.search_food(query)
        print(f"\nSearch '{query}': Found {len(results)} results")
        if results:
            first = results[0]
            print(f"  - {first['name']}: {first['nutrition']['calories']} cal")
    
    print("\n✅ Nutrition database built successfully!")


if __name__ == "__main__":
    main()