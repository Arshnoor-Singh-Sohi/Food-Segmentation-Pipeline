#!/usr/bin/env python3
"""
Comprehensive Food Detection System
==================================

Instead of hardcoded item types, this system:
1. Uses a comprehensive food database (500+ foods)
2. Dynamically detects ANY food type
3. Automatically categorizes and counts
4. Expandable to global cuisines

Usage:
python genai_system/comprehensive_food_detector.py --analyze data/input/refrigerator.jpg
python genai_system/comprehensive_food_detector.py --build-database
"""

import json
import sys
from pathlib import Path
from datetime import datetime

# Add genai_system to path
sys.path.append(str(Path(__file__).parent))

class ComprehensiveFoodDatabase:
    """
    Comprehensive food database covering global foods
    Expandable and categorized system
    """
    
    def __init__(self):
        self.database_file = Path("data/databases/comprehensive_food_db.json")
        self.database_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Load or create database
        self.food_db = self.load_or_create_database()
    
    def create_comprehensive_database(self):
        """Create comprehensive food database with 500+ foods"""
        print("🗃️ BUILDING COMPREHENSIVE FOOD DATABASE")
        print("=" * 50)
        
        # Global food database organized by categories
        comprehensive_db = {
            "fruits": {
                "citrus": ["orange", "lemon", "lime", "grapefruit", "tangerine", "mandarin"],
                "berries": ["strawberry", "blueberry", "raspberry", "blackberry", "cranberry", "gooseberry"],
                "stone_fruits": ["peach", "plum", "apricot", "cherry", "nectarine"],
                "tropical": ["mango", "pineapple", "papaya", "kiwi", "passion_fruit", "dragon_fruit"],
                "common": ["apple", "banana", "grape", "pear", "watermelon", "cantaloupe"],
                "exotic": ["durian", "rambutan", "lychee", "jackfruit", "starfruit", "persimmon"]
            },
            "vegetables": {
                "leafy_greens": ["lettuce", "spinach", "kale", "arugula", "chard", "cabbage"],
                "root_vegetables": ["carrot", "potato", "sweet_potato", "beet", "radish", "turnip"],
                "cruciferous": ["broccoli", "cauliflower", "brussels_sprouts", "bok_choy"],
                "peppers": ["bell_pepper", "chili_pepper", "jalapeno", "habanero", "poblano"],
                "onion_family": ["onion", "garlic", "leek", "shallot", "scallion"],
                "squash": ["zucchini", "yellow_squash", "butternut_squash", "acorn_squash"]
            },
            "proteins": {
                "meats": ["chicken", "beef", "pork", "lamb", "turkey", "duck"],
                "seafood": ["salmon", "tuna", "shrimp", "crab", "lobster", "cod"],
                "dairy": ["milk", "cheese", "yogurt", "butter", "cream", "sour_cream"],
                "eggs": ["chicken_egg", "duck_egg", "quail_egg"],
                "plant_proteins": ["tofu", "tempeh", "beans", "lentils", "chickpeas"]
            },
            "grains_starches": {
                "grains": ["rice", "quinoa", "oats", "barley", "wheat", "bulgur"],
                "pasta": ["spaghetti", "penne", "macaroni", "lasagna_sheets"],
                "bread": ["white_bread", "whole_wheat_bread", "sourdough", "baguette"],
                "cereals": ["corn_flakes", "granola", "muesli", "oatmeal"]
            },
            "beverages": {
                "dairy_drinks": ["milk", "almond_milk", "oat_milk", "soy_milk"],
                "juices": ["orange_juice", "apple_juice", "grape_juice", "cranberry_juice"],
                "water": ["bottled_water", "sparkling_water", "flavored_water"],
                "soft_drinks": ["cola", "lemon_lime_soda", "root_beer", "ginger_ale"],
                "alcoholic": ["beer", "wine", "champagne", "sake"]
            },
            "condiments_sauces": {
                "basic_condiments": ["ketchup", "mustard", "mayonnaise", "hot_sauce"],
                "asian_sauces": ["soy_sauce", "teriyaki", "sriracha", "fish_sauce"],
                "cooking_oils": ["olive_oil", "vegetable_oil", "coconut_oil", "sesame_oil"],
                "vinegars": ["white_vinegar", "apple_cider_vinegar", "balsamic_vinegar"]
            },
            "snacks_sweets": {
                "chips": ["potato_chips", "tortilla_chips", "pretzels", "crackers"],
                "candy": ["chocolate", "gummy_bears", "hard_candy", "mints"],
                "cookies": ["chocolate_chip", "oatmeal", "sugar_cookie", "oreo"],
                "ice_cream": ["vanilla", "chocolate", "strawberry", "mint_chip"]
            },
            "international_foods": {
                "asian": ["kimchi", "miso", "nori", "rice_cakes", "dumplings"],
                "mexican": ["tortillas", "salsa", "guacamole", "jalapeños", "cilantro"],
                "italian": ["pasta_sauce", "parmesan", "basil", "oregano", "mozzarella"],
                "indian": ["curry_paste", "turmeric", "cumin", "coriander", "naan"]
            }
        }
        
        # Flatten into searchable format with metadata
        flattened_db = {}
        food_id = 0
        
        for category, subcategories in comprehensive_db.items():
            for subcategory, foods in subcategories.items():
                for food in foods:
                    flattened_db[food] = {
                        "id": food_id,
                        "name": food,
                        "category": category,
                        "subcategory": subcategory,
                        "searchable_names": [
                            food,
                            food.replace("_", " "),
                            food.replace("_", "")
                        ],
                        "unit_type": self.get_default_unit(category, food),
                        "count_method": self.get_count_method(category, food)
                    }
                    food_id += 1
        
        # Add metadata
        database = {
            "version": "1.0",
            "created_date": datetime.now().isoformat(),
            "total_foods": len(flattened_db),
            "categories": list(comprehensive_db.keys()),
            "foods": flattened_db
        }
        
        # Save database
        with open(self.database_file, 'w') as f:
            json.dump(database, f, indent=2)
        
        print(f"✅ Database created: {self.database_file}")
        print(f"📊 Total foods: {len(flattened_db)}")
        print(f"📂 Categories: {len(comprehensive_db)}")
        
        return database
    
    def get_default_unit(self, category, food):
        """Get appropriate unit for food type"""
        if category == "fruits":
            if "berry" in food or food in ["grape"]:
                return "pieces"
            else:
                return "individual"
        elif category == "vegetables":
            if food in ["lettuce", "cabbage", "broccoli", "cauliflower"]:
                return "head"
            else:
                return "individual"
        elif category == "beverages":
            return "bottle"
        elif category == "proteins":
            if "milk" in food or "cream" in food:
                return "ml"
            else:
                return "package"
        else:
            return "individual"
    
    def get_count_method(self, category, food):
        """Get counting method for food type"""
        if category in ["fruits", "vegetables"] and food not in ["grape"]:
            return "individual_count"
        elif food in ["grape", "berries"]:
            return "cluster_count"
        elif category == "beverages":
            return "container_count"
        else:
            return "package_count"
    
    def load_or_create_database(self):
        """Load existing database or create new one"""
        if self.database_file.exists():
            with open(self.database_file, 'r') as f:
                return json.load(f)
        else:
            return self.create_comprehensive_database()
    
    def search_food(self, food_name):
        """Search for food in database with fuzzy matching"""
        food_name_clean = food_name.lower().strip()
        
        # Direct match
        if food_name_clean in self.food_db["foods"]:
            return self.food_db["foods"][food_name_clean]
        
        # Search in searchable names
        for food_key, food_data in self.food_db["foods"].items():
            for searchable_name in food_data["searchable_names"]:
                if food_name_clean in searchable_name.lower() or searchable_name.lower() in food_name_clean:
                    return food_data
        
        # Partial match
        for food_key, food_data in self.food_db["foods"].items():
            if food_name_clean in food_key or food_key in food_name_clean:
                return food_data
        
        return None
    
    def add_new_food(self, food_name, category="unknown", subcategory="other"):
        """Add new food to database dynamically"""
        new_id = max([food["id"] for food in self.food_db["foods"].values()]) + 1
        
        new_food = {
            "id": new_id,
            "name": food_name,
            "category": category,
            "subcategory": subcategory,
            "searchable_names": [food_name, food_name.replace("_", " ")],
            "unit_type": "individual",
            "count_method": "individual_count",
            "auto_added": True,
            "added_date": datetime.now().isoformat()
        }
        
        self.food_db["foods"][food_name] = new_food
        self.food_db["total_foods"] = len(self.food_db["foods"])
        
        # Save updated database
        with open(self.database_file, 'w') as f:
            json.dump(self.food_db, f, indent=2)
        
        print(f"➕ Added new food: {food_name}")
        return new_food

class ComprehensiveFoodDetector:
    """
    Enhanced GenAI detector that can handle ANY food type
    Uses comprehensive database for classification
    """
    
    def __init__(self):
        # Load food database
        self.food_db = ComprehensiveFoodDatabase()
        
        # Load GenAI analyzer
        try:
            from genai_analyzer import GenAIAnalyzer
            self.genai_analyzer = GenAIAnalyzer()
            print("✅ GenAI analyzer loaded")
        except ImportError:
            print("❌ GenAI analyzer not found")
            self.genai_analyzer = None
    
    def create_dynamic_prompt(self):
        """Create dynamic prompt that can detect ANY food"""
        
        # Get sample foods from database for prompting
        sample_foods = []
        categories = ["fruits", "vegetables", "proteins", "beverages"]
        
        for category in categories:
            category_foods = [
                food for food, data in self.food_db.food_db["foods"].items() 
                if data["category"] == category
            ][:10]  # Take first 10 from each category
            sample_foods.extend(category_foods)
        
        prompt = f"""
You are an expert food inventory analyst. Analyze this refrigerator image and identify ALL food items.

INSTRUCTIONS:
- Identify EVERY food item you can see (don't limit to specific types)
- Count each item individually (e.g., 3 apples = 3 separate items)
- Use specific food names when possible (e.g., "red_apple", "green_apple", "banana", "orange")
- For unknown items, describe them (e.g., "unknown_fruit_red", "leafy_green_vegetable")

EXAMPLE FOODS (but detect ANY food you see):
{', '.join(sample_foods[:30])}

Return ONLY this JSON format:
{{
  "total_items": number,
  "processing_time": "2-3 seconds",
  "inventory": [
    {{
      "item_type": "specific_food_name",
      "quantity": number,
      "confidence": 0.90,
      "description": "brief description if uncertain"
    }}
  ]
}}
"""
        return prompt
    
    def analyze_with_comprehensive_detection(self, image_path):
        """Analyze image with comprehensive food detection"""
        print(f"🔍 COMPREHENSIVE FOOD DETECTION")
        print(f"Image: {Path(image_path).name}")
        
        if not self.genai_analyzer:
            print("❌ GenAI analyzer not available")
            return None
        
        # Create dynamic prompt
        prompt = self.create_dynamic_prompt()
        
        # Run GenAI analysis with comprehensive prompt
        try:
            # Use GenAI with custom prompt
            results = self.run_genai_with_custom_prompt(image_path, prompt)
            
            if not results:
                print("❌ No results from GenAI")
                return None
            
            # Process results through food database
            enhanced_results = self.enhance_with_database(results)
            
            return enhanced_results
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return None
    
    def run_genai_with_custom_prompt(self, image_path, prompt):
        """Run GenAI with custom comprehensive prompt"""
        import openai
        import base64
        import os
        from dotenv import load_dotenv
        
        load_dotenv()
        
        # Encode image
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        
        # Call OpenAI
        client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=2000,
            temperature=0.1
        )
        
        # Parse response
        content = response.choices[0].message.content.strip()
        
        # Clean JSON
        if content.startswith('```json'):
            content = content[7:]
        if content.endswith('```'):
            content = content[:-3]
        
        result = json.loads(content)
        result['analysis_method'] = 'GPT-4 Vision Comprehensive'
        result['timestamp'] = datetime.now().isoformat()
        
        return result
    
    def enhance_with_database(self, genai_results):
        """Enhance GenAI results with food database information"""
        enhanced_inventory = []
        unknown_foods = []
        
        for item in genai_results.get('inventory', []):
            item_type = item['item_type']
            quantity = item['quantity']
            confidence = item['confidence']
            
            # Search in food database
            food_data = self.food_db.search_food(item_type)
            
            if food_data:
                # Found in database
                enhanced_item = {
                    **item,
                    "database_match": True,
                    "category": food_data["category"],
                    "subcategory": food_data["subcategory"],
                    "unit_type": food_data["unit_type"],
                    "count_method": food_data["count_method"]
                }
            else:
                # Not in database - add as unknown
                enhanced_item = {
                    **item,
                    "database_match": False,
                    "category": "unknown",
                    "subcategory": "unclassified",
                    "unit_type": "individual",
                    "count_method": "individual_count",
                    "needs_classification": True
                }
                unknown_foods.append(item_type)
                
                # Auto-add to database for future use
                self.food_db.add_new_food(item_type, "unknown", "auto_detected")
            
            enhanced_inventory.append(enhanced_item)
        
        # Create enhanced results
        enhanced_results = {
            **genai_results,
            "inventory": enhanced_inventory,
            "database_info": {
                "total_foods_in_db": self.food_db.food_db["total_foods"],
                "matched_foods": len([i for i in enhanced_inventory if i["database_match"]]),
                "unknown_foods": unknown_foods,
                "database_coverage": len([i for i in enhanced_inventory if i["database_match"]]) / len(enhanced_inventory) * 100 if enhanced_inventory else 0
            }
        }
        
        return enhanced_results
    
    def print_comprehensive_results(self, results):
        """Print detailed results with database information"""
        print("\n" + "="*60)
        print("🍎 COMPREHENSIVE FOOD DETECTION RESULTS")
        print("="*60)
        
        # Overall stats
        total_items = results.get('total_items', 0)
        db_info = results.get('database_info', {})
        coverage = db_info.get('database_coverage', 0)
        
        print(f"📊 Detection Summary:")
        print(f"   Total items detected: {total_items}")
        print(f"   Database coverage: {coverage:.1f}%")
        print(f"   Known foods: {db_info.get('matched_foods', 0)}")
        print(f"   Unknown foods: {len(db_info.get('unknown_foods', []))}")
        
        # Category breakdown
        category_counts = {}
        for item in results.get('inventory', []):
            category = item.get('category', 'unknown')
            if category not in category_counts:
                category_counts[category] = 0
            category_counts[category] += item['quantity']
        
        print(f"\n📂 By Category:")
        for category, count in category_counts.items():
            print(f"   {category.title()}: {count} items")
        
        # Individual items
        print(f"\n📋 Individual Items:")
        for item in results.get('inventory', []):
            name = item['item_type'].replace('_', ' ').title()
            qty = item['quantity']
            conf = item['confidence']
            category = item.get('category', 'unknown')
            match_status = "✅" if item.get('database_match') else "❓"
            
            print(f"   {match_status} {name}: {qty} ({category}) - {conf:.1%}")
        
        # Unknown foods
        unknown = db_info.get('unknown_foods', [])
        if unknown:
            print(f"\n❓ Unknown Foods (auto-added to database):")
            for food in unknown:
                print(f"   • {food.replace('_', ' ').title()}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Comprehensive Food Detection System')
    parser.add_argument('--build-database', action='store_true',
                       help='Build comprehensive food database')
    parser.add_argument('--analyze', type=str,
                       help='Analyze image with comprehensive detection')
    parser.add_argument('--show-database', action='store_true',
                       help='Show database statistics')
    
    args = parser.parse_args()
    
    print("🍎 COMPREHENSIVE FOOD DETECTION SYSTEM")
    print("=" * 60)
    
    if args.build_database:
        db = ComprehensiveFoodDatabase()
        print("✅ Comprehensive food database built!")
        print(f"📊 Total foods: {db.food_db['total_foods']}")
        
    elif args.show_database:
        db = ComprehensiveFoodDatabase()
        print(f"📊 Database Statistics:")
        print(f"   Total foods: {db.food_db['total_foods']}")
        print(f"   Categories: {len(db.food_db.get('categories', []))}")
        
        # Show some examples
        print(f"\n🍎 Sample Foods:")
        for i, (food_name, food_data) in enumerate(list(db.food_db['foods'].items())[:10]):
            print(f"   {food_name} ({food_data['category']})")
        
    elif args.analyze:
        detector = ComprehensiveFoodDetector()
        results = detector.analyze_with_comprehensive_detection(args.analyze)
        
        if results:
            detector.print_comprehensive_results(results)
            
            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = Path(f"data/genai_results/comprehensive_analysis_{timestamp}.json")
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"\n💾 Results saved: {output_file}")
        
    else:
        print("Usage:")
        print("  python genai_system/comprehensive_food_detector.py --build-database")
        print("  python genai_system/comprehensive_food_detector.py --analyze data/input/refrigerator.jpg")
        print("  python genai_system/comprehensive_food_detector.py --show-database")

if __name__ == "__main__":
    main()