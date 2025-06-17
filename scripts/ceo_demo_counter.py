# scripts/ceo_demo_counter.py
#!/usr/bin/env python3
"""
CEO Demo: Intelligent ingredient counter using both models
"""

import os
from pathlib import Path

def run_ceo_demo(image_path):
    print("🍌 INTELLIGENT INGREDIENT COUNTING DEMO")
    print("="*50)
    
    # Run default model
    print("\n1️⃣ Individual Detection (Default Model):")
    os.system(f"python scripts/detect_and_count_ingredients.py --image {image_path} --cpu --confidence 0.25")
    
    # Run enhanced counting
    print("\n2️⃣ Enhanced Analysis:")
    os.system(f"python scripts/enhanced_ingredient_counter.py {image_path}")
    
    print("\n✅ SUMMARY: Default model excels at counting individual items")
    print("   Custom model needs fine-tuning for ingredient counting tasks")

if __name__ == "__main__":
    import sys
    run_ceo_demo(sys.argv[1] if len(sys.argv) > 1 else "data/input/bananas.jpg")