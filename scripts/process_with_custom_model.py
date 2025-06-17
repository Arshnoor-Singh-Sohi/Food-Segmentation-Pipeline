#!/usr/bin/env python3
"""
Process images using CUSTOM trained model with metadata extraction
This script specifically uses your custom_food_detection.pt model
"""

import argparse
import json
import time
from pathlib import Path
import cv2
import sys
import torch
from ultralytics import YOLO

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.metadata.metadata_aggregator import MetadataAggregator
from src.pipeline.output_formatter import OutputFormatter

import matplotlib.pyplot as plt

def create_custom_model_visualization(image_path, results, output_dir, base_name, timestamp, model_name):
    """Create visualization for custom model results"""
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Create figure
    fig = plt.figure(figsize=(16, 10))
    
    # 1. Main image with detections
    ax1 = plt.subplot(2, 2, 1)
    ax1.imshow(image_rgb)
    ax1.set_title(f"Custom Model Detections ({model_name})", fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # Draw bounding boxes
    for i, item in enumerate(results['enriched_items']):
        bbox = item['bbox']
        x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
        
        # Draw box
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                           linewidth=3, edgecolor='red', facecolor='none')
        ax1.add_patch(rect)
        
        # Add label
        label = f"{item['detailed_food_type']} ({item['confidence']:.2f})"
        ax1.text(x1, y1-10, label, color='red', fontsize=10, 
                weight='bold', bbox=dict(boxstyle="round,pad=0.3", 
                facecolor='yellow', alpha=0.7))
    
    # 2. Detection summary table
    ax2 = plt.subplot(2, 2, 2)
    ax2.axis('tight')
    ax2.axis('off')
    
    # Create summary data
    table_data = []
    for item in results['enriched_items']:
        table_data.append([
            item.get('name', 'unknown'),
            item['detailed_food_type'],
            f"{item['confidence']:.2%}",
            f"{item['nutrition']['calories']:.0f} cal"
        ])
    
    if table_data:
        table = ax2.table(cellText=table_data,
                         colLabels=['Detected', 'Classified As', 'Confidence', 'Calories'],
                         cellLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
    
    ax2.set_title("Custom Model Results", fontsize=14, fontweight='bold')
    
    # 3. Nutrition info
    ax3 = plt.subplot(2, 2, 3)
    ax3.axis('off')
    
    nutrition_text = "NUTRITIONAL INFORMATION:\n\n"
    total_nutrition = results.get('total_nutrition', {})
    nutrition_text += f"Total Calories: {total_nutrition.get('calories', 0):.0f}\n"
    nutrition_text += f"Protein: {total_nutrition.get('protein_g', 0):.1f}g\n"
    nutrition_text += f"Carbs: {total_nutrition.get('carbs_g', 0):.1f}g\n"
    nutrition_text += f"Fat: {total_nutrition.get('fat_g', 0):.1f}g\n"
    nutrition_text += f"Fiber: {total_nutrition.get('fiber_g', 0):.1f}g\n"
    
    ax3.text(0.1, 0.9, nutrition_text, transform=ax3.transAxes, 
            fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.5))
    ax3.set_title("Nutrition Summary", fontsize=14, fontweight='bold')
    
    # 4. Model info
    ax4 = plt.subplot(2, 2, 4)
    ax4.axis('off')
    
    model_info_text = f"MODEL INFORMATION:\n\n"
    model_info_text += f"Model: {model_name}\n"
    model_info_text += f"Type: Custom Food Detection\n"
    model_info_text += f"Items Detected: {len(results['enriched_items'])}\n"
    model_info_text += f"Processing Device: {'GPU' if torch.cuda.is_available() else 'CPU'}\n"
    model_info_text += f"\n📌 This is YOUR custom trained model!\nCompare with default YOLOv8 results\nto see improvement."
    
    ax4.text(0.1, 0.9, model_info_text, transform=ax4.transAxes, 
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle="round,pad=0.5", facecolor='lightyellow', alpha=0.5))
    ax4.set_title("Custom Model Info", fontsize=14, fontweight='bold')
    
    # Overall title
    fig.suptitle(f"Custom Model Food Analysis - {base_name}", 
                fontsize=16, fontweight='bold')
    
    # Save visualization
    plt.tight_layout()
    viz_path = output_dir / f"{base_name}_custom_model_viz_{timestamp}.png"
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved visualization: {viz_path}")
    return viz_path

def create_comparison_note(output_dir, base_name, timestamp):
    """Create a note about comparing models"""
    comparison_file = output_dir / f"{base_name}_comparison_guide_{timestamp}.txt"
    
    with open(comparison_file, 'w') as f:
        f.write("MODEL COMPARISON GUIDE\n")
        f.write("="*50 + "\n\n")
        f.write("You've run the CUSTOM model. To compare:\n\n")
        f.write("1. Run default model:\n")
        f.write(f"   python scripts/process_with_metadata.py --image data/input/{base_name}.jpg\n\n")
        f.write("2. Compare outputs:\n")
        f.write(f"   - Custom: data/output/custom_model_results/\n")
        f.write(f"   - Default: data/output/metadata_results/\n\n")
        f.write("3. Look for differences in:\n")
        f.write("   - Number of items detected\n")
        f.write("   - Detection confidence\n")
        f.write("   - Food classification accuracy\n")
    
    print(f"📝 Created comparison guide: {comparison_file}")

class CustomModelProcessor:
    """Process images using custom trained YOLO model"""
    
    def __init__(self, custom_model_path: str):
        """Initialize with custom model"""
        self.model_path = Path(custom_model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Custom model not found at: {custom_model_path}")
        
        print(f"🎯 Loading CUSTOM model: {self.model_path}")
        self.model = YOLO(str(self.model_path))
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"🔧 Using device: {self.device}")
        
        # Get model info
        self.print_model_info()
    
    def print_model_info(self):
        """Print information about the custom model"""
        print("\n📊 Custom Model Information:")
        print(f"  - Model path: {self.model_path}")
        print(f"  - Model type: Custom Food Detection")
        print(f"  - Task: Segmentation")
        
        # Check if it's really a custom model
        if 'custom' in str(self.model_path).lower() or 'food' in str(self.model_path).lower():
            print(f"  ✅ This is your CUSTOM trained model!")
        else:
            print(f"  ⚠️  This might not be your custom model")
    
    def process_image(self, image_path: str):
        """Process single image with custom model"""
        image = cv2.imread(image_path)
        
        # Run inference
        results = self.model(image_path, device=self.device)
        
        if not results or len(results) == 0:
            return {'error': 'No detections', 'food_items': []}
        
        result = results[0]
        
        # Process detections
        food_items = []
        
        if result.boxes is not None and len(result.boxes) > 0:
            for i, box in enumerate(result.boxes):
                # Get segmentation mask if available
                mask = None
                if result.masks is not None and i < len(result.masks):
                    mask = result.masks[i].data.cpu().numpy()
                
                # Extract detection info
                item = {
                    'id': i,
                    'bbox': {
                        'x1': float(box.xyxy[0][0]),
                        'y1': float(box.xyxy[0][1]),
                        'x2': float(box.xyxy[0][2]),
                        'y2': float(box.xyxy[0][3])
                    },
                    'confidence': float(box.conf),
                    'class_id': int(box.cls),
                    'class_name': result.names[int(box.cls)],
                    'is_food': True,  # Assuming custom model only detects food
                    'has_mask': mask is not None
                }
                
                if mask is not None:
                    # Calculate mask area
                    mask_binary = (mask[0] > 0.5).astype(int)
                    area_pixels = mask_binary.sum()
                    total_pixels = mask_binary.shape[0] * mask_binary.shape[1]
                    
                    item['mask_info'] = {
                        'area_pixels': int(area_pixels),
                        'area_percentage': float(area_pixels / total_pixels * 100)
                    }
                
                food_items.append(item)
        
        return {
            'food_items': food_items,
            'image_info': {
                'width': image.shape[1],
                'height': image.shape[0],
                'model_used': 'custom_food_detection'
            }
        }

def process_with_custom_model(image_path: str, model_path: str, output_dir: str = None, compare: bool = False):
    """Main processing function using custom model"""
    print(f"\n🍔 PROCESSING WITH CUSTOM MODEL")
    print("="*50)
    print(f"📸 Image: {image_path}")
    print(f"🎯 Custom Model: {model_path}")
    
    start_time = time.time()
    
    # Step 1: Detection with Custom Model
    print("\n📍 Step 1: Detection & Segmentation (CUSTOM MODEL)")
    processor = CustomModelProcessor(model_path)
    detection_results = processor.process_image(image_path)
    
    if 'error' in detection_results:
        print(f"❌ Detection failed: {detection_results['error']}")
        return None
    
    print(f"✅ Found {len(detection_results['food_items'])} items with CUSTOM model")
    
    # Step 2: Metadata Extraction
    print("\n🏷️  Step 2: Metadata Labeling")
    metadata_extractor = MetadataAggregator()
    enhanced_results = metadata_extractor.extract_metadata(image_path, detection_results)
    
    print(f"✅ Extracted metadata for {len(enhanced_results['enriched_items'])} food items")
    
    # Step 3: Save Results
    print("\n💾 Step 3: Saving Results")
    output_dir = Path(output_dir) if output_dir else Path("data/output/custom_model_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    base_name = Path(image_path).stem
    
    # Save JSON
    json_path = output_dir / f"{base_name}_custom_model_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump(enhanced_results, f, indent=2)
    print(f"✅ Saved JSON: {json_path}")
    
    # Create visualization
    viz_path = create_custom_model_visualization(
        image_path, 
        enhanced_results, 
        output_dir, 
        base_name, 
        timestamp,
        Path(model_path).name
    )
    
    # Create comparison note if requested
    if compare: 
        create_comparison_note(output_dir, base_name, timestamp)
    
    # Print summary
    total_time = time.time() - start_time
    print_custom_model_summary(enhanced_results, total_time, model_path)
    
    return enhanced_results
def print_custom_model_summary(results, total_time, model_path):
    """Print summary for custom model results"""
    print("\n" + "="*50)
    print("📊 CUSTOM MODEL ANALYSIS COMPLETE")
    print("="*50)
    
    print(f"\n🎯 Model Used: {Path(model_path).name}")
    print(f"⏱️  Processing time: {total_time:.2f} seconds")
    
    if 'meal_summary' in results:
        print(f"\n🍽️  Meal Summary:")
        summary = results['meal_summary']
        print(f"  - Type: {summary['meal_type']}")
        print(f"  - Main cuisine: {summary['main_cuisine']}")
        print(f"  - Total items: {summary['total_items']}")
        print(f"  - Total calories: {summary['total_calories']:.0f}")
    
    print(f"\n🥘 Detected Foods (by CUSTOM MODEL):")
    for i, item in enumerate(results.get('enriched_items', []), 1):
        print(f"  {i}. {item['detailed_food_type']} ({item['cuisine']})")
        print(f"     - Original detection: {item.get('name', 'unknown')}")
        print(f"     - Confidence: {item['classification_confidence']:.2%}")
        print(f"     - Calories: {item['nutrition']['calories']:.0f}")

def main():
    parser = argparse.ArgumentParser(description="Process food images with CUSTOM model")
    parser.add_argument('--image', type=str, required=True, help='Path to image file')
    parser.add_argument('--model', type=str, default='data/models/custom_food_detection.pt',
                        help='Path to custom model (default: data/models/custom_food_detection.pt)')
    parser.add_argument('--output-dir', type=str, help='Output directory for results')
    parser.add_argument('--compare', action='store_true', 
                        help='Compare results with default YOLOv8 model')
    
    args = parser.parse_args()
    
    # Check if custom model exists
    if not Path(args.model).exists():
        print(f"❌ Custom model not found at: {args.model}")
        print("\nPlease ensure your custom model is at one of these locations:")
        print("  - data/models/custom_food_detection.pt")
        print("  - data/models/best.pt")
        print("  - data/models/yolov8m_custom_food.pt")
        return
    
    # Process with custom model
    results = process_with_custom_model(
        args.image, 
        args.model, 
        args.output_dir,
        args.compare  # Add this line
    )
    
    if args.compare and results:
        print("\n" + "="*50)
        print("🔄 COMPARING WITH DEFAULT MODEL...")
        print("="*50)
        # You could run the default model here for comparison
        print("Run this to compare: python scripts/process_with_metadata.py --image " + args.image)

if __name__ == "__main__":
    main()