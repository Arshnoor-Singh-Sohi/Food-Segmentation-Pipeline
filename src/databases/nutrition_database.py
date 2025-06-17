"""
Nutrition database handler
"""

import sqlite3
import json
from pathlib import Path
from typing import Dict, List, Optional

class NutritionDatabase:
    """Interface to nutrition database"""
    
    def __init__(self, db_path: str = "data/databases/nutrition/nutrition_expanded.db"):
        self.db_path = Path(db_path)
        self.conn = None
        
        if self.db_path.exists():
            self.conn = sqlite3.connect(str(self.db_path))
        else:
            print(f"Warning: Nutrition database not found at {db_path}")
            
    def get_nutrition(self, food_name: str) -> Optional[Dict]:
        """Get nutrition info for food"""
        if not self.conn:
            return None
            
        cursor = self.conn.cursor()
        
        # Try exact match
        result = cursor.execute("""
            SELECT n.* FROM nutrition n
            JOIN foods f ON n.food_id = f.id
            WHERE LOWER(f.name) = LOWER(?)
        """, (food_name,)).fetchone()
        
        if not result:
            # Try partial match
            result = cursor.execute("""
                SELECT n.* FROM nutrition n
                JOIN foods f ON n.food_id = f.id
                WHERE LOWER(f.name) LIKE LOWER(?)
                LIMIT 1
            """, (f"%{food_name.split()[0]}%",)).fetchone()
        
        if result:
            return {
                'calories': result[1],
                'protein_g': result[2],
                'carbs_g': result[3],
                'fat_g': result[4],
                'fiber_g': result[5] or 0,
                'sugar_g': result[6] or 0,
                'sodium_mg': result[7] or 0
            }
        
        return None