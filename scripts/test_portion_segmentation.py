"""
Test the portion-aware segmentation system
This demonstrates the CEO's requirements in action
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.metadata.food_type_classifier import FoodTypeClassifier
from src.metadata.measurement_units import MeasurementUnitSystem
from src.models.portion_aware_segmentation import PortionAwareSegmentation
from src.models.fast_yolo_segmentation import FastFoodSegmentation
import json

def test_portion_segmentation(image_path: str):
    """Test the portion-aware segmentation on an image"""
    
    print("🍕 TESTING PORTION-AWARE SEGMENTATION")
    print("="*60)
    print(f"Image: {image_path}")
    
    # Step 1: Run detection
    print("\n1️⃣ Running YOLO detection...")
    detector = FastFoodSegmentation()
    detection_results = detector.process_single_image(image_path, save_visualization=False)
    
    print(f"   Found {len(detection_results['food_items'])} items")
    
    # Step 2: Initialize our systems
    print("\n2️⃣ Initializing portion-aware system...")
    food_classifier = FoodTypeClassifier()
    measurement_system = MeasurementUnitSystem()
    portion_segmentation = PortionAwareSegmentation(food_classifier, measurement_system)
    
    # Step 3: Process segmentation
    print("\n3️⃣ Processing segmentation based on food type...")
    import cv2
    image = cv2.imread(image_path)
    
    segmentation_results = portion_segmentation.process_segmentation(
        detection_results, image
    )
    
    # Step 4: Display results
    print("\n📊 RESULTS:")
    print(f"Food Type: {segmentation_results['food_type_classification']['type']}")
    print(f"Confidence: {segmentation_results['food_type_classification']['confidence']:.1%}")
    print(f"Explanation: {segmentation_results['food_type_classification']['explanation']}")
    
    print(f"\n📦 Segments Created: {len(segmentation_results['segments'])}")
    
    for segment in segmentation_results['segments']:
        print(f"\n   Segment: {segment['id']}")
        print(f"   Name: {segment['name']}")
        print(f"   Type: {segment['type']}")
        print(f"   Measurement: {segment['measurement']['formatted']}")
    
    print("\n📏 Measurement Summary:")
    for food, measurement in segmentation_results['measurement_summary'].items():
        if isinstance(measurement, dict) and 'formatted' in measurement:
            print(f"   {food}: {measurement['formatted']}")
    
    # Save results
    output_path = Path("data/output/portion_segmentation_test.json")
    output_path.parent.mkdir(exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(segmentation_results, f, indent=2, default=str)
    
    print(f"\n✅ Results saved to: {output_path}")

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
            test_portion_segmentation(image_path)

if __name__ == "__main__":
    # Test specific image
    if len(sys.argv) > 1:
        test_portion_segmentation(sys.argv[1])
    else:
        # Test multiple scenarios
        test_multiple_scenarios()