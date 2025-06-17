"""
Output formatter for metadata results
Handles formatting and saving of enriched food detection results
"""

import json
import csv
from pathlib import Path
from typing import Dict, List, Any
import cv2
import numpy as np
from datetime import datetime

class OutputFormatter:
    """Format and save metadata extraction results"""
    
    def __init__(self, output_dir: str = "data/output/metadata_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def save_results(self, results: Dict[str, Any], base_name: str) -> Dict[str, str]:
        """Save results in multiple formats"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save paths
        saved_files = {}
        
        # 1. Save complete JSON
        json_path = self.output_dir / f"{base_name}_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        saved_files['json'] = str(json_path)
        
        # 2. Save summary CSV
        if 'enriched_items' in results:
            csv_path = self.output_dir / f"{base_name}_{timestamp}_summary.csv"
            self._save_csv_summary(results['enriched_items'], csv_path)
            saved_files['csv'] = str(csv_path)
        
        # 3. Save nutrition report
        report_path = self.output_dir / f"{base_name}_{timestamp}_nutrition.txt"
        self._save_nutrition_report(results, report_path)
        saved_files['report'] = str(report_path)
        
        return saved_files
    
    def _save_csv_summary(self, items: List[Dict], csv_path: Path):
        """Save summary as CSV"""
        if not items:
            return
            
        # Extract fields for CSV
        fieldnames = [
            'food_type', 'confidence', 'cuisine', 'calories', 
            'protein_g', 'carbs_g', 'fat_g', 'portion_size'
        ]
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for item in items:
                row = {
                    'food_type': item.get('detailed_food_type', 'unknown'),
                    'confidence': item.get('classification_confidence', 0),
                    'cuisine': item.get('cuisine', 'unknown'),
                    'calories': item.get('nutrition', {}).get('calories', 0),
                    'protein_g': item.get('nutrition', {}).get('protein_g', 0),
                    'carbs_g': item.get('nutrition', {}).get('carbs_g', 0),
                    'fat_g': item.get('nutrition', {}).get('fat_g', 0),
                    'portion_size': item.get('portion', {}).get('serving_description', 'unknown')
                }
                writer.writerow(row)
    
    def _save_nutrition_report(self, results: Dict, report_path: Path):
        """Save detailed nutrition report"""
        with open(report_path, 'w') as f:
            f.write("FOOD ANALYSIS NUTRITION REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            # Meal summary
            if 'meal_summary' in results:
                summary = results['meal_summary']
                f.write(f"Meal Type: {summary.get('meal_type', 'unknown')}\n")
                f.write(f"Main Cuisine: {summary.get('main_cuisine', 'unknown')}\n")
                f.write(f"Total Items: {summary.get('total_items', 0)}\n")
                f.write(f"Total Calories: {summary.get('total_calories', 0):.0f}\n\n")
            
            # Individual items
            f.write("FOOD ITEMS DETECTED:\n")
            f.write("-" * 30 + "\n")
            
            for i, item in enumerate(results.get('enriched_items', []), 1):
                f.write(f"\n{i}. {item.get('detailed_food_type', 'unknown').upper()}\n")
                f.write(f"   Confidence: {item.get('classification_confidence', 0):.2%}\n")
                f.write(f"   Cuisine: {item.get('cuisine', 'unknown')}\n")
                f.write(f"   Portion: {item.get('portion', {}).get('serving_description', 'unknown')}\n")
                
                # Nutrition
                nutrition = item.get('nutrition', {})
                f.write(f"   Nutrition per serving:\n")
                f.write(f"     - Calories: {nutrition.get('calories', 0):.0f}\n")
                f.write(f"     - Protein: {nutrition.get('protein_g', 0):.1f}g\n")
                f.write(f"     - Carbs: {nutrition.get('carbs_g', 0):.1f}g\n")
                f.write(f"     - Fat: {nutrition.get('fat_g', 0):.1f}g\n")
                
                # Ingredients
                if 'ingredients' in item:
                    f.write(f"   Ingredients: {', '.join(item['ingredients'])}\n")
                
                # Allergens
                if 'allergens' in item and item['allergens']:
                    f.write(f"   ⚠️  Allergens: {', '.join(item['allergens'])}\n")
            
            # Total nutrition
            if 'total_nutrition' in results:
                f.write("\n" + "=" * 50 + "\n")
                f.write("TOTAL MEAL NUTRITION:\n")
                total = results['total_nutrition']
                f.write(f"  Calories: {total.get('calories', 0):.0f}\n")
                f.write(f"  Protein: {total.get('protein_g', 0):.1f}g\n")
                f.write(f"  Carbohydrates: {total.get('carbs_g', 0):.1f}g\n")
                f.write(f"  Fat: {total.get('fat_g', 0):.1f}g\n")
                f.write(f"  Fiber: {total.get('fiber_g', 0):.1f}g\n")
                f.write(f"  Sugar: {total.get('sugar_g', 0):.1f}g\n")
                f.write(f"  Sodium: {total.get('sodium_mg', 0):.0f}mg\n")