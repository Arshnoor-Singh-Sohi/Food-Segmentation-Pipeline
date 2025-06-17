#!/usr/bin/env python3
"""
Main script to process images with complete metadata extraction
This is what you run to analyze food images
"""

import argparse
import json
import time
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
from datetime import datetime

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.fast_yolo_segmentation import FastFoodSegmentation
from src.metadata.metadata_aggregator import MetadataAggregator
from src.pipeline.output_formatter import OutputFormatter

def process_image_with_metadata(image_path: str, output_dir: str = None):
    """
    Process a single image through the complete pipeline with metadata
    
    Args:
        image_path: Path to image
        output_dir: Where to save results
    """
    print(f"\n🍔 PROCESSING IMAGE WITH METADATA EXTRACTION")
    print("="*50)
    print(f"📸 Image: {image_path}")
    
    start_time = time.time()
    
    # Step 1: Detection and Segmentation
    print("\n📍 Step 1: Detection & Segmentation")
    segmenter = FastFoodSegmentation(model_size='m')
    detection_results = segmenter.process_single_image(image_path, save_visualization=False)
    
    if 'error' in detection_results:
        print(f"❌ Detection failed: {detection_results['error']}")
        return None
    
    print(f"✅ Found {len(detection_results['food_items'])} items")
    
    # Step 2: Metadata Extraction
    print("\n🏷️  Step 2: Metadata Labeling")
    metadata_extractor = MetadataAggregator()
    enhanced_results = metadata_extractor.extract_metadata(image_path, detection_results)
    
    print(f"✅ Extracted metadata for {len(enhanced_results['enriched_items'])} food items")
    
    # Step 3: Format and Save Results
    print("\n💾 Step 3: Saving Results")
    
    if output_dir is None:
        output_dir = Path("data/output/metadata_results")
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = Path(image_path).stem
    
    # Save JSON
    json_path = output_dir / f"{base_name}_metadata_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump(enhanced_results, f, indent=2)
    print(f"✅ Saved JSON: {json_path}")
    
    # Create visualization
    create_metadata_visualization(image_path, enhanced_results, output_dir, base_name, timestamp)
    
    # Print summary
    total_time = time.time() - start_time
    print_results_summary(enhanced_results, total_time)
    
    return enhanced_results

def create_metadata_visualization(image_path, results, output_dir, base_name, timestamp):
    """Create comprehensive visualization with metadata"""
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Original image with bounding boxes
    ax1 = plt.subplot(2, 3, 1)
    ax1.imshow(image_rgb)
    ax1.set_title("Detected Food Items", fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # Draw bounding boxes and labels
    for i, item in enumerate(results['enriched_items']):
        bbox = item['bbox']
        x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
        
        # Draw box
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                           linewidth=3, edgecolor='red', facecolor='none')
        ax1.add_patch(rect)
        
        # Add label
        label = f"{item['detailed_food_type']} ({item['classification_confidence']:.2f})"
        ax1.text(x1, y1-10, label, color='red', fontsize=10, 
                weight='bold', bbox=dict(boxstyle="round,pad=0.3", 
                facecolor='yellow', alpha=0.7))
    
    # 2. Metadata table
    ax2 = plt.subplot(2, 3, 2)
    ax2.axis('tight')
    ax2.axis('off')
    
    # Create metadata table
    table_data = []
    for item in results['enriched_items']:
        table_data.append([
            item['detailed_food_type'],
            item['cuisine'],
            f"{item['nutrition']['calories']:.0f} cal",
            item['portion']['serving_description']
        ])
    
    if table_data:
        table = ax2.table(cellText=table_data,
                         colLabels=['Food', 'Cuisine', 'Calories', 'Portion'],
                         cellLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
    
    ax2.set_title("Detected Food Metadata", fontsize=14, fontweight='bold')
    
    # 3. Nutrition breakdown
    ax3 = plt.subplot(2, 3, 3)
    nutrition = results['total_nutrition']
    
    # Pie chart of macros
    macros = [nutrition['protein_g'] * 4, 
              nutrition['carbs_g'] * 4, 
              nutrition['fat_g'] * 9]
    macro_labels = ['Protein', 'Carbs', 'Fat']
    colors = ['#ff9999', '#66b3ff', '#99ff99']
    
    if sum(macros) > 0:
        ax3.pie(macros, labels=macro_labels, colors=colors, autopct='%1.1f%%')
        ax3.set_title(f"Macronutrients\nTotal: {nutrition['calories']:.0f} calories", 
                     fontsize=14, fontweight='bold')
    
    # 4. Ingredients list
    ax4 = plt.subplot(2, 3, 4)
    ax4.axis('off')
    
    ingredients_text = "INGREDIENTS DETECTED:\n\n"
    for item in results['enriched_items']:
        ingredients_text += f"{item['detailed_food_type'].upper()}:\n"
        ingredients_text += f"  • " + "\n  • ".join(item['ingredients']) + "\n\n"
    
    ax4.text(0.1, 0.9, ingredients_text, transform=ax4.transAxes, 
            fontsize=11, verticalalignment='top')
    ax4.set_title("Ingredients Analysis", fontsize=14, fontweight='bold')
    
    # 5. Dietary information
    ax5 = plt.subplot(2, 3, 5)
    ax5.axis('off')
    
    # Collect all dietary tags and allergens
    all_tags = set()
    all_allergens = set()
    
    for item in results['enriched_items']:
        all_tags.update(item.get('dietary_tags', []))
        all_allergens.update(item.get('allergens', []))
    
    dietary_text = "DIETARY INFORMATION:\n\n"
    dietary_text += "Dietary Tags:\n"
    dietary_text += "  • " + "\n  • ".join(all_tags) if all_tags else "  • None identified"
    dietary_text += "\n\nAllergens:\n"
    dietary_text += "  • " + "\n  • ".join(all_allergens) if all_allergens else "  • None identified"
    
    ax5.text(0.1, 0.9, dietary_text, transform=ax5.transAxes, 
            fontsize=11, verticalalignment='top')
    ax5.set_title("Dietary & Allergen Info", fontsize=14, fontweight='bold')
    
    # 6. Meal summary
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    summary = results['meal_summary']
    summary_text = f"MEAL ANALYSIS:\n\n"
    summary_text += f"Meal Type: {summary['meal_type'].upper()}\n"
    summary_text += f"Main Cuisine: {summary['main_cuisine']}\n"
    summary_text += f"Total Items: {summary['total_items']}\n"
    summary_text += f"Total Calories: {summary['total_calories']:.0f}\n"
    summary_text += f"\nCuisines Present:\n"
    summary_text += "  • " + "\n  • ".join(summary['cuisines_present'])
    
    ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes, 
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.5))
    ax6.set_title("Meal Summary", fontsize=14, fontweight='bold')
    
    # Overall title
    fig.suptitle(f"Complete Food Analysis with Metadata - {base_name}", 
                fontsize=16, fontweight='bold')
    
    # Save visualization
    plt.tight_layout()
    viz_path = output_dir / f"{base_name}_metadata_viz_{timestamp}.png"
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved visualization: {viz_path}")

def print_results_summary(results, total_time):
    """Print summary of results"""
    print("\n" + "="*50)
    print("📊 ANALYSIS COMPLETE")
    print("="*50)
    
    print(f"\n⏱️  Processing time: {total_time:.2f} seconds")
    print(f"\n🍽️  Meal Summary:")
    summary = results['meal_summary']
    print(f"  - Type: {summary['meal_type']}")
    print(f"  - Main cuisine: {summary['main_cuisine']}")
    print(f"  - Total items: {summary['total_items']}")
    print(f"  - Total calories: {summary['total_calories']:.0f}")
    
    print(f"\n🥘 Detected Foods:")
    for i, item in enumerate(results['enriched_items'], 1):
        print(f"  {i}. {item['detailed_food_type']} ({item['cuisine']})")
        print(f"     - Calories: {item['nutrition']['calories']:.0f}")
        print(f"     - Portion: {item['portion']['serving_description']}")
        print(f"     - Confidence: {item['classification_confidence']:.2%}")
    
    print(f"\n🥗 Total Nutrition:")
    nutrition = results['total_nutrition']
    print(f"  - Calories: {nutrition['calories']:.0f}")
    print(f"  - Protein: {nutrition['protein_g']:.1f}g")
    print(f"  - Carbs: {nutrition['carbs_g']:.1f}g")
    print(f"  - Fat: {nutrition['fat_g']:.1f}g")
    
    # Dietary tags
    all_tags = set()
    for item in results['enriched_items']:
        all_tags.update(item.get('dietary_tags', []))
    
    if all_tags:
        print(f"\n🏷️  Dietary Tags: {', '.join(all_tags)}")
    
    # Allergens
    all_allergens = set()
    for item in results['enriched_items']:
        all_allergens.update(item.get('allergens', []))
    
    if all_allergens:
        print(f"\n⚠️  Allergens: {', '.join(all_allergens)}")

def main():
    parser = argparse.ArgumentParser(description="Process food images with metadata extraction")
    parser.add_argument('--image', type=str, required=True, help='Path to image file')
    parser.add_argument('--output-dir', type=str, help='Output directory for results')
    parser.add_argument('--batch', action='store_true', help='Process all images in directory')
    
    args = parser.parse_args()
    
    if args.batch:
        # Process all images in directory
        image_dir = Path(args.image)
        if not image_dir.is_dir():
            print(f"❌ {args.image} is not a directory")
            return
        
        image_files = list(image_dir.glob('*.jpg')) + list(image_dir.glob('*.png'))
        print(f"📁 Found {len(image_files)} images to process")
        
        for img_path in image_files:
            print(f"\n{'='*50}")
            process_image_with_metadata(str(img_path), args.output_dir)
    else:
        # Process single image
        if not Path(args.image).exists():
            print(f"❌ Image not found: {args.image}")
            return
        
        process_image_with_metadata(args.image, args.output_dir)

if __name__ == "__main__":
    main()