#!/usr/bin/env python3
"""Process a single image with YOLO - Fast and reliable!"""

import sys
import argparse
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.fast_yolo_segmentation import FastFoodSegmentation

def main():
    parser = argparse.ArgumentParser(description="Fast YOLO food analysis")
    parser.add_argument('image_path', help="Path to the food image")
    parser.add_argument('--model-size', choices=['n', 's', 'm', 'l', 'x'], 
                       default='n', help="YOLO model size (n=fastest, x=most accurate)")
    parser.add_argument('--no-viz', action='store_true', 
                       help="Skip visualization creation")
    parser.add_argument('--output-dir', default='data/output/yolo_results',
                       help="Output directory")
    
    args = parser.parse_args()
    
    # Check if image exists
    if not Path(args.image_path).exists():
        print(f"ERROR: Image not found: {args.image_path}")
        return 1
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Initialize processor
        print(f"Initializing YOLO{args.model_size} processor...")
        processor = FastFoodSegmentation(model_size=args.model_size)
        
        # Process image
        print(f"Processing: {Path(args.image_path).name}")
        results = processor.process_single_image(
            args.image_path, 
            save_visualization=not args.no_viz
        )
        
        # Save JSON results
        json_file = output_dir / f"{Path(args.image_path).stem}_results.json"
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Print summary
        print_results_summary(results)
        print(f"\nResults saved to: {json_file}")
        
        if 'visualization_path' in results:
            print(f"Visualization saved to: {results['visualization_path']}")
        
        return 0
        
    except Exception as e:
        print(f"Processing failed: {e}")
        return 1

def print_results_summary(results):
    """Print a nice summary of results."""
    if 'error' in results:
        print(f"ERROR: {results['error']}")
        return
    
    info = results['image_info']
    summary = results['analysis_summary']
    nutrition = results['nutrition_totals']
    
    print("\n" + "="*60)
    print("FOOD ANALYSIS RESULTS")
    print("="*60)
    print(f"Image: {info['filename']}")
    print(f"Processing time: {info['processing_time_seconds']}s")
    print(f"Items detected: {summary['total_items_detected']}")
    print(f"Food items: {summary['food_items_count']}")
    print(f"Non-food items: {summary['non_food_items_count']}")
    print(f"Average confidence: {summary['avg_confidence']}")
    
    print(f"\nNUTRITION TOTALS:")
    print(f"   Calories: {nutrition['calories']:.1f}")
    print(f"   Protein: {nutrition['protein_g']:.1f}g")
    print(f"   Carbs: {nutrition['carbs_g']:.1f}g")
    print(f"   Fat: {nutrition['fat_g']:.1f}g")
    
    print(f"\nDETECTED ITEMS:")
    for i, item in enumerate(results['food_items'], 1):
        food_marker = "[FOOD]" if item['is_food'] else "[ITEM]"
        print(f"   {i}. {food_marker} {item['name']} ({item['confidence']:.2f} conf)")
        if item['is_food']:
            nutrition = item['nutrition']
            print(f"      -> ~{nutrition['calories']:.0f} cal, {nutrition['portion_grams']}g")
    
    print("="*60)

if __name__ == "__main__":
    sys.exit(main())