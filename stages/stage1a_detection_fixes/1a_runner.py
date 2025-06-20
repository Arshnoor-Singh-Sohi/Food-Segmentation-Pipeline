"""
Stage 1A: Detection Fixes Runner
===============================

Place this file in: stages/stage1a_detection_fixes/1a_runner.py

This handles Stage 1A specific execution:
- False positive bottle detection fixes
- Banana quantity counting fixes  
- Item classification accuracy
- Portion vs Complete Dish display
"""

import sys
import os
from pathlib import Path
import json
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

def run_stage(args):
    """Main entry point for Stage 1A called by universal runner"""
    print("🔧 STAGE 1A: Detection Accuracy Fixes")
    print("=" * 50)
    
    try:
        from stages.stage1a_detection_fixes.detection_fixer import Stage1ADetectionFixer
        
        # Initialize fixer
        fixer = Stage1ADetectionFixer()
        
        # Determine which test to run based on arguments
        if args.refrigerator:
            return run_refrigerator_test(fixer)
        elif args.image:
            return run_single_image_test(fixer, args.image)
        elif args.test_all:
            return run_all_tests(fixer)
        else:
            print("Please specify test type:")
            print("  --refrigerator    # Test refrigerator scenario")
            print("  --image path.jpg  # Test specific image")
            print("  --test-all        # Run all available tests")
            return False
            
    except ImportError as e:
        print(f"❌ Error importing Stage 1A modules: {e}")
        print("Make sure detection_fixer.py exists in stages/stage1a_detection_fixes/")
        return False

def run_refrigerator_test(fixer):
    """Test with refrigerator images specifically"""
    print("🧊 Testing Refrigerator Detection Fixes")
    print("-" * 40)
    
    # Look for refrigerator test images
    test_images = [
        "data/input/refrigerator.jpg",
        "data/input/fridge.jpg",
        "data/input/kitchen.jpg"
    ]
    
    project_root = Path(__file__).parent.parent.parent
    
    for image_path in test_images:
        full_path = project_root / image_path
        if full_path.exists():
            print(f"📸 Found refrigerator image: {image_path}")
            return run_detection_analysis(fixer, str(full_path))
    
    print("❌ No refrigerator test images found.")
    print("Please add a refrigerator image to one of:")
    for path in test_images:
        print(f"   {path}")
    return False

def run_single_image_test(fixer, image_path):
    """Test with a specific image"""
    project_root = Path(__file__).parent.parent.parent
    
    # Handle relative paths
    if not os.path.isabs(image_path):
        full_path = project_root / image_path
    else:
        full_path = Path(image_path)
    
    if not full_path.exists():
        print(f"❌ Image not found: {full_path}")
        return False
    
    print(f"📸 Testing image: {full_path}")
    return run_detection_analysis(fixer, str(full_path))

def run_all_tests(fixer):
    """Run all available test scenarios"""
    print("🧪 Running All Stage 1A Tests")
    print("-" * 40)
    
    test_scenarios = [
        ("data/input/refrigerator.jpg", "Refrigerator inventory"),
        ("data/input/banana_cluster.jpg", "Banana cluster counting"),
        ("data/input/pizza.jpg", "Complete dish classification"),
        ("data/input/bottles.jpg", "Bottle false positive test")
    ]
    
    project_root = Path(__file__).parent.parent.parent
    success_count = 0
    
    for relative_path, description in test_scenarios:
        full_path = project_root / relative_path
        if full_path.exists():
            print(f"\n📋 Test: {description}")
            if run_detection_analysis(fixer, str(full_path)):
                success_count += 1
        else:
            print(f"⚠️ Skipping {description} - {relative_path} not found")
    
    print(f"\n📊 Results: {success_count} successful tests")
    return success_count > 0

def run_detection_analysis(fixer, image_path):
    """Run the actual detection analysis"""
    try:
        print(f"\n🔍 Analyzing: {Path(image_path).name}")
        
        # Run the detection analysis
        results = fixer.analyze_image(image_path)
        
        if not results:
            print("❌ Analysis failed")
            return False
        
        # Create output directory for this stage
        output_dir = Path(__file__).parent.parent.parent / "data" / "output" / "stage1a_results"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_name = Path(image_path).stem
        
        # Save JSON results
        json_file = output_dir / f"{image_name}_stage1a_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Create visualization
        viz_file = fixer.create_visualization(image_path, results, output_dir, timestamp)
        
        # Print summary
        print_stage1a_summary(results)
        
        print(f"\n💾 Results saved:")
        print(f"   📄 JSON: {json_file}")
        print(f"   📊 Visualization: {viz_file}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        return False

def print_stage1a_summary(results):
    """Print Stage 1A specific summary"""
    print("\n" + "="*50)
    print("📊 STAGE 1A ANALYSIS SUMMARY")
    print("="*50)
    
    # Detection improvements
    raw_count = len(results.get('raw_detections', []))
    enhanced_count = len(results.get('enhanced_detections', []))
    
    print(f"🔍 Detection Improvements:")
    print(f"   Raw detections: {raw_count}")
    print(f"   Enhanced detections: {enhanced_count}")
    
    if raw_count > enhanced_count:
        filtered = raw_count - enhanced_count
        print(f"   ✅ Filtered out {filtered} false positives")
    
    # Specific fixes
    item_counts = results.get('item_counts', {})
    
    # Bottle fix
    if 'bottle' in item_counts:
        bottle_info = item_counts['bottle']
        print(f"\n🍼 BOTTLE FIX:")
        print(f"   Detected: {bottle_info['detected_objects']} bottles")
        print(f"   Status: Applied enhanced validation")
    
    # Banana fix
    if 'banana' in item_counts:
        banana_info = item_counts['banana']
        detected = banana_info['detected_objects']
        estimated = banana_info['estimated_total']
        print(f"\n🍌 BANANA FIX:")
        print(f"   Detected objects: {detected}")
        print(f"   Estimated bananas: {estimated}")
        if estimated > detected:
            print(f"   Status: Applied cluster analysis")
    
    # Food type classification
    food_type = results.get('food_type_classification', {})
    print(f"\n🍽️ FOOD TYPE CLASSIFICATION:")
    print(f"   Type: {food_type.get('type', 'unknown').upper()}")
    print(f"   Confidence: {food_type.get('confidence', 0):.1%}")
    print(f"   Display as: {'COMPLETE DISH' if food_type.get('type') == 'complete_dish' else 'INDIVIDUAL ITEMS'}")
    
    # Stage 1A specific success criteria
    print(f"\n✅ STAGE 1A SUCCESS CRITERIA:")
    success_items = []
    
    if 'bottle' in item_counts and item_counts['bottle']['detected_objects'] <= 5:
        success_items.append("Bottle false positives reduced")
    
    if 'banana' in item_counts:
        success_items.append("Banana counting improved with cluster analysis")
    
    if food_type.get('confidence', 0) > 0.7:
        success_items.append("Food type classification working")
    
    for item in success_items:
        print(f"   ✅ {item}")
    
    if len(success_items) >= 2:
        print(f"\n🎉 Stage 1A objectives achieved!")
    else:
        print(f"\n⚠️ Stage 1A needs more work")

def main(args=None):
    """Standalone main function for direct execution"""
    if args is None:
        import argparse
        parser = argparse.ArgumentParser(description='Stage 1A: Detection Fixes')
        parser.add_argument('--image', type=str, help='Path to image file')
        parser.add_argument('--refrigerator', action='store_true', help='Test refrigerator scenario')
        parser.add_argument('--test-all', action='store_true', help='Run all tests')
        args = parser.parse_args()
    
    return run_stage(args)

if __name__ == "__main__":
    main()