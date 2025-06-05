#!/usr/bin/env python3
"""Process a single food image through the complete pipeline."""

import sys
import argparse
import logging
from pathlib import Path
import yaml
import matplotlib.pyplot as plt
import numpy as np

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models.combined_pipeline import FoodAnalysisPipeline
from utils.visualization import FoodVisualization

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Process a single food image")
    parser.add_argument('image_path', help="Path to the food image")
    parser.add_argument(
        '--config', 
        default='config/config.yaml',
        help="Path to config file"
    )
    parser.add_argument(
        '--interactive', 
        action='store_true',
        help="Enable interactive point selection"
    )
    parser.add_argument(
        '--save-results', 
        action='store_true',
        default=True,
        help="Save analysis results"
    )
    parser.add_argument(
        '--show-visualization', 
        action='store_true',
        default=True,
        help="Show visualization"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"Config file not found: {args.config}")
        return 1
    
    # Check if image exists
    image_path = Path(args.image_path)
    if not image_path.exists():
        logger.error(f"Image not found: {image_path}")
        return 1
    
    try:
        # Initialize pipeline
        logger.info("Initializing food analysis pipeline...")
        pipeline = FoodAnalysisPipeline(config)
        
        # Process interactive points if requested
        interactive_points = None
        if args.interactive:
            logger.info("Click on food items in the image (close window when done)")
            interactive_points = collect_interactive_points(str(image_path))
        
        # Analyze the image
        logger.info(f"Analyzing image: {image_path}")
        results = pipeline.analyze_meal_image(
            str(image_path),
            interactive_points=interactive_points,
            save_results=args.save_results
        )
        
        # Print results summary
        print_results_summary(results)
        
        # Show visualization if requested
        if args.show_visualization:
            visualizer = FoodVisualization()
            visualizer.visualize_meal_analysis(results)
        
        logger.info("Analysis completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return 1

def collect_interactive_points(image_path: str):
    """Collect interactive points from user clicks."""
    import cv2
    
    # Global variable to store points
    points = []
    
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            # Draw a circle at the clicked point
            cv2.circle(param, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow('Click on food items', param)
    
    # Load and display image
    image = cv2.imread(image_path)
    if image is None:
        logger.error(f"Could not load image: {image_path}")
        return None
    
    # Convert BGR to RGB for display
    display_image = image.copy()
    
    cv2.namedWindow('Click on food items', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('Click on food items', mouse_callback, display_image)
    cv2.imshow('Click on food items', display_image)
    
    print("Click on food items you want to segment. Press any key when done.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    logger.info(f"Collected {len(points)} interactive points")
    return points

def print_results_summary(results: dict):
    """Print a summary of the analysis results."""
    print("\n" + "="*60)
    print("FOOD ANALYSIS RESULTS")
    print("="*60)
    
    meal_summary = results['meal_summary']
    total_nutrition = results['total_nutrition']
    
    print(f"Image: {Path(results['image_path']).name}")
    print(f"Total food items detected: {meal_summary['total_items']}")
    print(f"  - YOLO detections: {meal_summary['detected_items']}")
    print(f"  - Interactive items: {meal_summary['interactive_items']}")
    print(f"  - Automatic items: {meal_summary['automatic_items']}")
    print(f"Average quality score: {meal_summary['avg_quality_score']:.2f}")
    
    print("\nFood types identified:")
    for food_type in meal_summary['food_types']:
        print(f"  - {food_type}")
    
    print("\nTotal Nutrition:")
    print(f"  Calories: {total_nutrition['calories']:.1f}")
    print(f"  Protein: {total_nutrition['protein']:.1f}g")
    print(f"  Carbohydrates: {total_nutrition['carbohydrates']:.1f}g")
    print(f"  Fat: {total_nutrition['fat']:.1f}g")
    print(f"  Fiber: {total_nutrition['fiber']:.1f}g")
    
    print("\nDetailed breakdown:")
    for i, item in enumerate(results['food_items']):
        print(f"\n{i+1}. {item['name']}")
        print(f"   Detection confidence: {item['detection_confidence']:.2f}")
        print(f"   Segmentation confidence: {item['segmentation_confidence']:.2f}")
        print(f"   Portion: {item['portion_info']['portion_size']}")
        nutrition = item.get('nutrition', {})
        if 'calories' in nutrition:
            print(f"   Calories: {nutrition['calories']:.1f}")
    
    print("="*60)

if __name__ == "__main__":
    sys.exit(main())