# Save as: scripts/test_portion_segmentation_enhanced.py
"""
Enhanced test script for portion-aware segmentation
Includes visual output and fixes JSON overwriting
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.metadata.food_type_classifier import FoodTypeClassifier
from src.metadata.measurement_units import MeasurementUnitSystem
from src.models.portion_aware_segmentation_enhanced import PortionAwareSegmentation
from src.models.fast_yolo_segmentation import FastFoodSegmentation
import json
import cv2
from datetime import datetime
import numpy as np
from typing import Dict, List, Any

def test_portion_segmentation(image_path: str, output_name: str = None, output_subfolder: str = "general"):
    """
    Test the portion-aware segmentation on an image
    
    Args:
        image_path: Path to input image
        output_name: Optional name for output files (uses timestamp if not provided)
        output_subfolder: Subfolder within output directory for organization
    """
    
    print("🍕 TESTING PORTION-AWARE SEGMENTATION")
    print("="*60)
    print(f"Image: {image_path}")
    
    # Generate unique output name if not provided
    if output_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_name = Path(image_path).stem
        output_name = f"{image_name}_{timestamp}"
    
    # Create organized output directory
    output_dir = Path("data/output") / output_subfolder
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Step 1: Run detection
    print("\n1️⃣ Running YOLO detection...")
    detector = FastFoodSegmentation()
    detection_results = detector.process_single_image(image_path, save_visualization=False)
    
    # Debug: Print raw detection results
    print(f"\n🔍 Debug - Raw YOLO Results:")
    if 'food_items' in detection_results:
        print(f"   Total items in 'food_items': {len(detection_results['food_items'])}")
        for idx, item in enumerate(detection_results['food_items']):
            print(f"   Item {idx}: {item.get('name', 'unknown')} - Confidence: {item.get('confidence', 0):.2f}")
    
    # Step 2: Initialize our systems
    print("\n2️⃣ Initializing portion-aware system...")
    food_classifier = FoodTypeClassifier()
    measurement_system = MeasurementUnitSystem()
    portion_segmentation = PortionAwareSegmentation(food_classifier, measurement_system)
    
    # Step 3: Process segmentation with visualization
    print("\n3️⃣ Processing segmentation based on food type...")
    image = cv2.imread(image_path)
    
    segmentation_results = portion_segmentation.process_segmentation(
        detection_results, 
        image,
        save_visualization=True,
        output_dir=str(output_dir)
    )
    
    # Step 4: Display results
    print("\n📊 RESULTS:")
    print(f"Food Type: {segmentation_results['food_type_classification']['type']}")
    print(f"Confidence: {segmentation_results['food_type_classification']['confidence']:.1%}")
    print(f"Explanation: {segmentation_results['food_type_classification']['explanation']}")
    
    # Show processing stats
    stats = segmentation_results.get('processing_stats', {})
    print(f"\n🔢 Processing Statistics:")
    print(f"   Total Detections: {stats.get('total_detections', 'N/A')}")
    print(f"   Food Items Found: {stats.get('food_items_found', 'N/A')}")
    print(f"   Segments Created: {stats.get('segments_created', 'N/A')}")
    
    print(f"\n📦 Segments Created: {len(segmentation_results['segments'])}")
    
    for segment in segmentation_results['segments']:
        print(f"\n   Segment: {segment['id']}")
        print(f"   Name: {segment['name']}")
        print(f"   Type: {segment['type']}")
        print(f"   Measurement: {segment['measurement']['formatted']}")
        print(f"   BBox: [{segment['bbox']['x1']}, {segment['bbox']['y1']}, {segment['bbox']['x2']}, {segment['bbox']['y2']}]")
    
    print("\n📏 Measurement Summary:")
    for food, measurement in segmentation_results['measurement_summary'].items():
        if isinstance(measurement, dict) and 'formatted' in measurement:
            print(f"   {food}: {measurement['formatted']}")
    
    # Save JSON results with unique filename
    json_path = output_dir / f"segmentation_results_{output_name}.json"
    
    # Prepare JSON data with proper formatting
    json_data = {
        'image_path': str(image_path),
        'timestamp': datetime.now().isoformat(),
        'food_type_classification': segmentation_results['food_type_classification'],
        'processing_stats': segmentation_results.get('processing_stats', {}),
        'segments': segmentation_results['segments'],
        'measurement_summary': segmentation_results['measurement_summary'],
        'visualization_path': segmentation_results.get('visualization_path', '')
    }
    
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2, default=str)
    
    print(f"\n✅ JSON results saved to: {json_path}")
    print(f"✅ Visual results saved to: {segmentation_results.get('visualization_path', 'N/A')}")
    
    return segmentation_results

def test_refrigerator_image():
    """
    Special test for the refrigerator image with many items
    This is what your CEO specifically requested!
    """
    print("\n" + "="*60)
    print("🏪 TESTING REFRIGERATOR IMAGE (CEO's Request)")
    print("="*60)
    
    fridge_path = "data/input/refrigerator.jpg"  # Update with your actual path
    
    if not Path(fridge_path).exists():
        print(f"❌ Refrigerator image not found at: {fridge_path}")
        print("Please update the path to your refrigerator image")
        return
    
    # Use specific folder for fridge-related outputs
    results = test_portion_segmentation(fridge_path, "refrigerator_analysis", "refrigerator_analysis")
    
    # Special analysis for refrigerator
    print("\n📊 REFRIGERATOR CONTENT ANALYSIS:")
    print("="*40)
    
    # Group by categories
    categories = {
        'fruits': [],
        'vegetables': [],
        'dairy': [],
        'condiments': [],
        'dry_goods': [],
        'beverages': [],
        'other': []
    }
    
    # Categorize items
    for segment in results['segments']:
        name = segment['name'].lower()
        
        if any(fruit in name for fruit in ['banana', 'apple', 'orange', 'grape', 'berry']):
            categories['fruits'].append(segment)
        elif any(veg in name for veg in ['lettuce', 'pepper', 'cabbage', 'carrot', 'broccoli']):
            categories['vegetables'].append(segment)
        elif any(dairy in name for dairy in ['milk', 'cheese', 'yogurt']):
            categories['dairy'].append(segment)
        elif any(cond in name for cond in ['ketchup', 'sauce', 'mayo', 'mustard']):
            categories['condiments'].append(segment)
        elif any(dry in name for dry in ['cereal', 'oats', 'nuts', 'grain']):
            categories['dry_goods'].append(segment)
        elif any(bev in name for bev in ['juice', 'water', 'soda', 'drink']):
            categories['beverages'].append(segment)
        else:
            categories['other'].append(segment)
    
    # Print categorized results
    for category, items in categories.items():
        if items:
            print(f"\n{category.upper()} ({len(items)} items):")
            for item in items:
                print(f"  - {item['name']}: {item['measurement']['formatted']}")
    
    # Create a special visualization for the fridge
    output_dir = Path("data/output/refrigerator_analysis")
    create_fridge_inventory_report(results, str(output_dir / "fridge_inventory.html"))

def create_fridge_inventory_report(results: Dict[str, Any], output_path: str):
    """
    Create an HTML report of the fridge inventory
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Refrigerator Inventory Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1 {{ color: #333; }}
            .stats {{ background: #f0f0f0; padding: 10px; border-radius: 5px; }}
            .item {{ margin: 10px 0; padding: 10px; background: #fff; border: 1px solid #ddd; }}
            .measurement {{ color: #0066cc; font-weight: bold; }}
            .bbox {{ color: #666; font-size: 0.9em; }}
        </style>
    </head>
    <body>
        <h1>Refrigerator Inventory Report</h1>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <div class="stats">
            <h2>Summary</h2>
            <p>Total Items Detected: {len(results['segments'])}</p>
            <p>Classification Type: {results['food_type_classification']['type']}</p>
        </div>
        
        <h2>Detected Items</h2>
    """
    
    for segment in results['segments']:
        html_content += f"""
        <div class="item">
            <strong>{segment['name']}</strong>
            <span class="measurement">{segment['measurement']['formatted']}</span>
            <div class="bbox">
                Location: [{segment['bbox']['x1']}, {segment['bbox']['y1']}, 
                           {segment['bbox']['x2']}, {segment['bbox']['y2']}]
            </div>
        </div>
        """
    
    html_content += """
    </body>
    </html>
    """
    
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    print(f"\n📄 HTML inventory report saved to: {output_path}")

def test_multiple_scenarios():
    """Test different food scenarios"""
    test_images = {
        'pizza': 'data/input/pizza.jpg',
        'fruit_bowl': 'data/input/fruits.jpg',
        'burger_meal': 'data/input/burger.jpg',
        'bananas': 'data/input/bananas.jpg',
        'salad': 'data/input/salad.jpg'
    }
    
    for scenario, image_path in test_images.items():
        if Path(image_path).exists():
            print(f"\n\n{'='*60}")
            print(f"TESTING SCENARIO: {scenario}")
            print('='*60)
            # Save each scenario in its own folder
            test_portion_segmentation(image_path, scenario, f"test_scenarios/{scenario}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test portion-aware segmentation')
    parser.add_argument('--image', type=str, help='Path to specific image')
    parser.add_argument('--fridge', action='store_true', help='Test refrigerator image')
    parser.add_argument('--all', action='store_true', help='Test all scenarios')
    
    args = parser.parse_args()
    
    if args.image:
        test_portion_segmentation(args.image)
    elif args.fridge:
        test_refrigerator_image()
    elif args.all:
        test_multiple_scenarios()
        test_refrigerator_image()
    else:
        # Default: test refrigerator as CEO requested
        test_refrigerator_image()