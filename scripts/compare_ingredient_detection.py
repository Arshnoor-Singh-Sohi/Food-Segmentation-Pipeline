#!/usr/bin/env python3
"""
Compare ingredient detection between custom and default models
"""

import os
import sys
from pathlib import Path

def compare_models(image_path):
    """Run both models and compare"""
    print("🔄 COMPARING MODELS ON INGREDIENT DETECTION")
    print("="*50)
    
    # Test with default model
    print("\n1️⃣ DEFAULT MODEL:")
    os.system(f"python scripts/detect_and_count_ingredients.py --image {image_path} --cpu")
    
    # Test with custom model
    print("\n2️⃣ CUSTOM MODEL:")
    custom_model = "data/models/custom_food_detection.pt"
    if Path(custom_model).exists():
        os.system(f"python scripts/detect_and_count_ingredients.py --image {image_path} --model {custom_model} --cpu")
    else:
        print("❌ Custom model not found at:", custom_model)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        compare_models(sys.argv[1])
    else:
        print("Usage: python scripts/compare_ingredient_detection.py <image_path>")