#!/usr/bin/env python3
"""
Quick demo script for CEO - Banana cluster detection
"""

import os
from pathlib import Path

# Test images to demonstrate
test_images = [
    "data/input/bananas.jpg",
    "data/input/banana_cluster.jpg",
    "data/input/fruit_bowl.jpg",
    "data/input/ingredients.jpg"
]

print("🍌 INGREDIENT DETECTION & COUNTING DEMO")
print("="*50)

# Run detection on each test image
for image_path in test_images:
    if Path(image_path).exists():
        print(f"\nProcessing: {image_path}")
        os.system(f"python scripts/detect_and_count_ingredients.py --image {image_path}")
    else:
        print(f"⚠️  Skipping {image_path} (not found)")

print("\n✅ Demo complete! Check data/output/ingredient_counts/ for results")