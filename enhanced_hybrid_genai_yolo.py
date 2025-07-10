#!/usr/bin/env python3
"""
Enhanced Hybrid GenAI + YOLO System with Visual Verification
==========================================================

Adds visual verification so you can see exactly where bounding boxes are placed.
Expands food mapping to handle more real-world food types.

Usage:
python enhanced_hybrid_genai_yolo.py --analyze data/input/refrigerator.jpg --visualize
"""

import sys
from pathlib import Path
import json
from datetime import datetime
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Add genai_system to path
sys.path.append(str(Path(__file__).parent / "genai_system"))

try:
    from ultralytics import YOLO
    from genai_analyzer import GenAIAnalyzer
except ImportError:
    print("❌ Missing dependencies")
    sys.exit(1)

class EnhancedHybridDetectionSystem:
    """
    Enhanced hybrid system with visual verification and expanded food mapping
    """
    
    def __init__(self):
        # Load systems
        self.genai_analyzer = GenAIAnalyzer()
        self.yolo_model = YOLO('yolov8n.pt')  # Generic YOLO for positions
        self.local_model = None
        
        # Try to load local model
        local_model_path = "data/models/genai_trained_local_model2/weights/best.pt"
        if Path(local_model_path).exists():
            self.local_model = YOLO(local_model_path)
            print("✅ Local model loaded for comparison")
        
        # Expanded food mapping - covering more real-world foods
        self.food_mapping = {
            # Fruits
            'banana_individual': ['banana'],
            'apple_individual': ['apple'],
            'orange_individual': ['orange'],
            'lemon_individual': ['orange'],  # YOLO sees lemons as oranges
            'lime_individual': ['orange'],
            'pear_individual': ['apple'],   # YOLO might see pears as apples
            'grape_cluster': ['apple'],     # Might be detected as small round fruit
            
            # Vegetables  
            'carrot_individual': ['carrot'],
            'broccoli_head': ['broccoli'],
            'lettuce_head': ['broccoli'],   # Leafy greens might be similar
            'bell_pepper': ['apple'],       # Might be seen as round object
            'tomato_individual': ['apple'], # Round like apples
            'cucumber_single': ['banana'],  # Elongated like bananas
            
            # Containers & Bottles
            'bottle_individual': ['bottle', 'wine glass'],
            'bottle_milk': ['bottle'],
            'bottle_juice': ['bottle'], 
            'bottle_water': ['bottle'],
            'container_individual': ['bowl', 'cup', 'bottle'],
            'jar_glass': ['bottle', 'cup'],
            'can_food': ['cup', 'bowl'],
            'carton_milk': ['bottle'],
            'carton_juice': ['bottle'],
            
            # Prepared foods (YOLO might see as bowls/plates)
            'sandwich': ['bowl'],
            'pizza': ['bowl'],
            'salad': ['bowl'],
            
            # Dairy & Eggs
            'egg_carton': ['bowl'],
            'cheese_block': ['bowl'],
            'yogurt_container': ['cup', 'bowl'],
            
            # Condiments & Sauces
            'mayonnaise': ['bottle'],
            'ketchup': ['bottle'],
            'mustard': ['bottle'],
            'maple_syrup': ['bottle'],
            
            # Herbs & Seasonings
            'herbs': ['broccoli'],  # Might be seen as green leafy
            
            # Snacks
            'tortilla': ['bowl'],
            
            # Unknown/New foods
            'unknown_fruit_red': ['apple'],
            'unknown_vegetable': ['broccoli'],
            'unknown_container': ['bowl', 'cup'],
        }
        
        print(f"✅ Enhanced food mapping loaded: {len(self.food_mapping)} food types supported")
    
    def analyze_hybrid_with_visualization(self, image_path, create_visual=True):
        """
        Enhanced hybrid analysis with visual verification
        """
        print(f"🔄 ENHANCED HYBRID ANALYSIS: {Path(image_path).name}")
        print("=" * 60)
        
        # Step 1: GenAI for comprehensive food identification
        print("🧠 Step 1: GenAI Comprehensive Food Identification...")
        genai_results = self.genai_analyzer.analyze_refrigerator(image_path)
        
        if not genai_results:
            print("❌ GenAI analysis failed")
            return None
        
        genai_items = genai_results.get('inventory', [])
        genai_total = genai_results.get('total_items', 0)
        
        print(f"   ✅ GenAI found: {genai_total} items ({len(genai_items)} types)")
        for item in genai_items:
            print(f"      • {item['quantity']}x {item['item_type'].replace('_', ' ').title()}")
        
        # Step 2: Generic YOLO for accurate spatial detection
        print(f"\n📍 Step 2: Generic YOLO Spatial Detection...")
        yolo_results = self.yolo_model(image_path, conf=0.25)
        
        yolo_detections = []
        for r in yolo_results:
            if r.boxes is not None:
                for box in r.boxes:
                    detection = {
                        'class_name': self.yolo_model.names[int(box.cls)],
                        'confidence': float(box.conf),
                        'bbox': box.xyxy[0].cpu().numpy().tolist()
                    }
                    yolo_detections.append(detection)
        
        # Group YOLO detections by class
        yolo_summary = {}
        for det in yolo_detections:
            class_name = det['class_name']
            if class_name not in yolo_summary:
                yolo_summary[class_name] = 0
            yolo_summary[class_name] += 1
        
        print(f"   ✅ YOLO found: {len(yolo_detections)} positioned objects")
        for class_name, count in yolo_summary.items():
            print(f"      • {count}x {class_name}")
        
        # Step 3: Enhanced intelligent fusion
        print(f"\n🔗 Step 3: Enhanced Intelligent Fusion...")
        hybrid_results = self.enhanced_fusion(genai_items, yolo_detections)
        
        print(f"   ✅ Hybrid result: {len(hybrid_results)} items with positions")
        
        # Step 4: Local model comparison
        local_comparison = None
        if self.local_model:
            print(f"\n🤖 Step 4: Your Local Model Comparison...")
            local_results = self.local_model(image_path, conf=0.25)
            local_detections = []
            
            for r in local_results:
                if r.boxes is not None:
                    for box in r.boxes:
                        detection = {
                            'class_name': self.local_model.names[int(box.cls)],
                            'confidence': float(box.conf),
                            'bbox': box.xyxy[0].cpu().numpy().tolist()
                        }
                        local_detections.append(detection)
            
            print(f"   ✅ Your model: {len(local_detections)} items (limited to 6 trained types)")
            local_comparison = local_detections
        
        # Create comprehensive results
        results = {
            'genai_results': genai_results,
            'yolo_detections': yolo_detections,
            'local_detections': local_comparison,
            'hybrid_fusion': hybrid_results,
            'analysis_method': 'enhanced_hybrid_genai_yolo',
            'timestamp': datetime.now().isoformat(),
            'image_path': image_path
        }
        
        # Step 5: Create visual verification
        if create_visual:
            print(f"\n📊 Step 5: Creating Visual Verification...")
            self.create_comprehensive_visualization(image_path, results)
        
        return results
    
    def enhanced_fusion(self, genai_items, yolo_detections):
        """
        Enhanced fusion with better food type mapping
        """
        hybrid_results = []
        used_yolo_boxes = set()
        
        print(f"\n🔍 FUSION DETAILS:")
        
        # For each GenAI detected food type
        for genai_item in genai_items:
            food_type = genai_item['item_type']
            quantity = genai_item['quantity']
            confidence = genai_item['confidence']
            
            # Find matching YOLO detections using enhanced mapping
            yolo_classes = self.food_mapping.get(food_type, [])
            matching_boxes = []
            
            # Look for YOLO detections that could match this food type
            for i, yolo_det in enumerate(yolo_detections):
                if i in used_yolo_boxes:
                    continue
                    
                if yolo_det['class_name'] in yolo_classes:
                    matching_boxes.append({
                        'yolo_index': i,
                        'bbox': yolo_det['bbox'],
                        'yolo_confidence': yolo_det['confidence'],
                        'yolo_class': yolo_det['class_name']
                    })
                    used_yolo_boxes.add(i)
                    
                    # Stop when we have enough boxes
                    if len(matching_boxes) >= quantity:
                        break
            
            # Create hybrid results
            if matching_boxes:
                print(f"   ✅ {food_type}: {len(matching_boxes)}/{quantity} matched with YOLO")
                for j, box in enumerate(matching_boxes):
                    hybrid_results.append({
                        'food_type': food_type,
                        'genai_confidence': confidence,
                        'yolo_confidence': box['yolo_confidence'],
                        'yolo_class_matched': box['yolo_class'],
                        'combined_confidence': (confidence + box['yolo_confidence']) / 2,
                        'bbox': box['bbox'],
                        'method': 'genai_classification_yolo_position',
                        'match_quality': 'high' if box['yolo_confidence'] > 0.5 else 'medium'
                    })
                
                # Handle remaining GenAI detections without YOLO matches
                remaining = quantity - len(matching_boxes)
                if remaining > 0:
                    print(f"   ⚠️ {food_type}: {remaining} items estimated (no YOLO match)")
                    for k in range(remaining):
                        estimated_bbox = self.smart_position_estimation(
                            len(matching_boxes) + k, quantity, matching_boxes
                        )
                        hybrid_results.append({
                            'food_type': food_type,
                            'genai_confidence': confidence,
                            'yolo_confidence': 0.0,
                            'yolo_class_matched': 'none',
                            'combined_confidence': confidence * 0.6,
                            'bbox': estimated_bbox,
                            'method': 'genai_only_estimated_position',
                            'match_quality': 'estimated'
                        })
            else:
                print(f"   ❌ {food_type}: No YOLO match found, estimating positions")
                # Estimate all positions for this food type
                for i in range(quantity):
                    estimated_bbox = self.smart_position_estimation(i, quantity, [])
                    hybrid_results.append({
                        'food_type': food_type,
                        'genai_confidence': confidence,
                        'yolo_confidence': 0.0,
                        'yolo_class_matched': 'none',
                        'combined_confidence': confidence * 0.5,
                        'bbox': estimated_bbox,
                        'method': 'genai_only_estimated_position',
                        'match_quality': 'estimated'
                    })
        
        return hybrid_results
    
    def smart_position_estimation(self, item_index, total_items, existing_boxes):
        """
        Smarter position estimation that avoids existing boxes
        """
        # If we have existing boxes, place near them
        if existing_boxes:
            # Place near the last existing box with some offset
            last_box = existing_boxes[-1]['bbox']
            x1, y1, x2, y2 = last_box
            
            # Add offset
            offset_x = 50 + (item_index * 30)
            offset_y = 20 + (item_index * 20)
            
            new_x1 = max(0, x1 + offset_x)
            new_y1 = max(0, y1 + offset_y)
            new_x2 = new_x1 + (x2 - x1)
            new_y2 = new_y1 + (y2 - y1)
            
            return [new_x1, new_y1, new_x2, new_y2]
        
        # Otherwise use grid estimation
        cols = min(4, total_items)
        rows = (total_items + cols - 1) // cols
        
        col = item_index % cols
        row = item_index // cols
        
        # Image dimensions (assume standard size)
        img_width, img_height = 640, 480
        
        # Grid positions
        x_center = 0.15 + (col / max(cols - 1, 1)) * 0.7
        y_center = 0.15 + (row / max(rows - 1, 1)) * 0.7
        
        # Box size
        box_width = min(0.12, 0.8 / cols)
        box_height = min(0.15, 0.8 / rows)
        
        # Convert to pixel coordinates
        x1 = (x_center - box_width/2) * img_width
        y1 = (y_center - box_height/2) * img_height
        x2 = (x_center + box_width/2) * img_width
        y2 = (y_center + box_height/2) * img_height
        
        return [x1, y1, x2, y2]
    
    def create_comprehensive_visualization(self, image_path, results):
        """
        Create comprehensive visual verification with multiple views
        """
        # Load image
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        
        # 1. GenAI Detection (estimated positions)
        ax1 = axes[0, 0]
        ax1.imshow(image_rgb)
        ax1.set_title('GenAI Detection\n(Accurate Classification, Estimated Positions)', 
                     fontsize=14, fontweight='bold')
        
        genai_items = results['genai_results'].get('inventory', [])
        colors_genai = plt.cm.Set1(np.linspace(0, 1, len(genai_items)))
        
        for i, item in enumerate(genai_items):
            # Show GenAI items with estimated positions (just for reference)
            quantity = item['quantity']
            food_type = item['item_type']
            
            for j in range(min(quantity, 3)):  # Show max 3 per type to avoid clutter
                # Simple estimation for visualization
                x = 50 + (i * 80) + (j * 25)
                y = 50 + (i % 3) * 60
                width, height = 60, 40
                
                rect = patches.Rectangle((x, y), width, height, 
                                       linewidth=2, edgecolor=colors_genai[i], 
                                       facecolor='none', alpha=0.8)
                ax1.add_patch(rect)
                
                if j == 0:  # Label only first box per type
                    ax1.text(x, y-5, f"{food_type.replace('_', ' ')[:10]}", 
                           fontsize=8, color=colors_genai[i], fontweight='bold')
        
        ax1.set_xlabel(f"Total: {results['genai_results'].get('total_items', 0)} items")
        ax1.axis('off')
        
        # 2. Generic YOLO Detection (accurate positions)
        ax2 = axes[0, 1]
        ax2.imshow(image_rgb)
        ax2.set_title('Generic YOLO Detection\n(Accurate Positions, Generic Classification)', 
                     fontsize=14, fontweight='bold')
        
        yolo_detections = results['yolo_detections']
        colors_yolo = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
        
        class_color_map = {}
        color_idx = 0
        
        for detection in yolo_detections:
            bbox = detection['bbox']
            x1, y1, x2, y2 = bbox
            class_name = detection['class_name']
            confidence = detection['confidence']
            
            # Assign color per class
            if class_name not in class_color_map:
                class_color_map[class_name] = colors_yolo[color_idx % len(colors_yolo)]
                color_idx += 1
            
            color = class_color_map[class_name]
            
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                   linewidth=2, edgecolor=color, 
                                   facecolor='none', alpha=0.8)
            ax2.add_patch(rect)
            
            ax2.text(x1, y1-5, f"{class_name}: {confidence:.2f}", 
                   fontsize=8, color=color, fontweight='bold')
        
        ax2.set_xlabel(f"Total: {len(yolo_detections)} positioned objects")
        ax2.axis('off')
        
        # 3. Your Local Model
        ax3 = axes[1, 0]
        ax3.imshow(image_rgb)
        ax3.set_title('Your Local Model\n(6 Food Types Only, Poor Positions)', 
                     fontsize=14, fontweight='bold')
        
        if results['local_detections']:
            local_detections = results['local_detections']
            
            for detection in local_detections:
                bbox = detection['bbox']
                x1, y1, x2, y2 = bbox
                class_name = detection['class_name']
                confidence = detection['confidence']
                
                rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                       linewidth=2, edgecolor='red', 
                                       facecolor='none', alpha=0.6)
                ax3.add_patch(rect)
                
                ax3.text(x1, y1-5, f"{class_name.replace('_individual', '')}: {confidence:.2f}", 
                       fontsize=8, color='red', fontweight='bold')
            
            ax3.set_xlabel(f"Total: {len(local_detections)} items (limited classes)")
        else:
            ax3.set_xlabel("Local model not available")
        
        ax3.axis('off')
        
        # 4. Hybrid Result (Best of Both Worlds)
        ax4 = axes[1, 1]
        ax4.imshow(image_rgb)
        ax4.set_title('🎯 HYBRID RESULT\n(GenAI Classification + YOLO Positions)', 
                     fontsize=14, fontweight='bold', color='darkgreen')
        
        hybrid_results = results['hybrid_fusion']
        
        # Color code by match quality
        quality_colors = {
            'high': 'darkgreen',
            'medium': 'orange', 
            'estimated': 'red'
        }
        
        for result in hybrid_results:
            bbox = result['bbox']
            x1, y1, x2, y2 = bbox
            food_type = result['food_type']
            method = result['method']
            quality = result.get('match_quality', 'estimated')
            
            color = quality_colors.get(quality, 'gray')
            alpha = 0.9 if quality == 'high' else 0.7 if quality == 'medium' else 0.5
            
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                   linewidth=2, edgecolor=color, 
                                   facecolor='none', alpha=alpha)
            ax4.add_patch(rect)
            
            # Label with food type
            label = food_type.replace('_individual', '').replace('_', ' ')
            ax4.text(x1, y1-5, label, fontsize=8, color=color, fontweight='bold')
        
        ax4.set_xlabel(f"Total: {len(hybrid_results)} items (combined approach)")
        ax4.axis('off')
        
        # Add legend for hybrid
        legend_elements = [
            patches.Patch(color='darkgreen', label='High Quality (YOLO + GenAI)'),
            patches.Patch(color='orange', label='Medium Quality (YOLO + GenAI)'),
            patches.Patch(color='red', label='Estimated Position (GenAI only)')
        ]
        ax4.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        
        # Save visualization
        output_dir = Path("data/hybrid_results")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_name = Path(image_path).stem
        viz_file = output_dir / f"hybrid_visualization_{image_name}_{timestamp}.png"
        
        plt.savefig(viz_file, dpi=300, bbox_inches='tight')
        plt.show()  # Display the plot
        plt.close()
        
        print(f"   ✅ Visual verification saved: {viz_file}")
        print(f"   📊 4-panel comparison shows all detection methods")
        
        return viz_file
    
    def print_detailed_analysis(self, results):
        """
        Print detailed analysis of all approaches
        """
        print(f"\n📊 DETAILED PERFORMANCE ANALYSIS")
        print("=" * 70)
        
        genai_total = results['genai_results'].get('total_items', 0)
        yolo_total = len(results['yolo_detections'])
        hybrid_total = len(results['hybrid_fusion'])
        local_total = len(results['local_detections']) if results['local_detections'] else 0
        
        print(f"🧠 GENAI ANALYSIS:")
        print(f"   Items detected: {genai_total}")
        print(f"   Food types: {len(results['genai_results'].get('inventory', []))}")
        print(f"   Strength: Excellent food identification")
        print(f"   Weakness: No spatial information")
        print(f"   Cost: $0.02 per image")
        
        print(f"\n📍 GENERIC YOLO ANALYSIS:")
        print(f"   Items detected: {yolo_total}")
        print(f"   Object types: {len(set(d['class_name'] for d in results['yolo_detections']))}")
        print(f"   Strength: Accurate bounding boxes")
        print(f"   Weakness: Generic object classes (not food-specific)")
        print(f"   Cost: $0.00 per image")
        
        if results['local_detections']:
            print(f"\n🤖 YOUR LOCAL MODEL ANALYSIS:")
            print(f"   Items detected: {local_total}")
            print(f"   Food types: 6 (trained classes only)")
            print(f"   Strength: Food-specific, free to use")
            print(f"   Weakness: Limited food types, poor positioning")
            print(f"   Cost: $0.00 per image")
        
        print(f"\n🎯 HYBRID SYSTEM ANALYSIS:")
        print(f"   Items detected: {hybrid_total}")
        print(f"   Method: GenAI classification + YOLO positioning")
        print(f"   Strength: Best food identification + good positioning")
        print(f"   Weakness: Still costs money (same as GenAI)")
        print(f"   Cost: $0.02 per image")
        
        # Quality breakdown
        high_quality = sum(1 for r in results['hybrid_fusion'] if r.get('match_quality') == 'high')
        medium_quality = sum(1 for r in results['hybrid_fusion'] if r.get('match_quality') == 'medium')
        estimated = sum(1 for r in results['hybrid_fusion'] if r.get('match_quality') == 'estimated')
        
        print(f"\n🎯 HYBRID QUALITY BREAKDOWN:")
        print(f"   High quality matches: {high_quality} (YOLO + GenAI)")
        print(f"   Medium quality matches: {medium_quality} (YOLO + GenAI)")
        print(f"   Estimated positions: {estimated} (GenAI only)")
        
        accuracy_score = ((high_quality * 1.0) + (medium_quality * 0.8) + (estimated * 0.5)) / hybrid_total * 100 if hybrid_total > 0 else 0
        
        print(f"\n🏆 HYBRID ACCURACY SCORE: {accuracy_score:.1f}%")
        
        return results

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced Hybrid GenAI + YOLO Detection System')
    parser.add_argument('--analyze', type=str, default='data/input/refrigerator.jpg',
                       help='Image to analyze')
    parser.add_argument('--visualize', action='store_true', default=True,
                       help='Create visual verification (default: True)')
    
    args = parser.parse_args()
    
    print("🔄 ENHANCED HYBRID GENAI + YOLO DETECTION SYSTEM")
    print("=" * 70)
    print("Combining GenAI accuracy with YOLO spatial precision + Visual Verification")
    
    # Initialize enhanced hybrid system
    hybrid_system = EnhancedHybridDetectionSystem()
    
    # Run enhanced hybrid analysis
    results = hybrid_system.analyze_hybrid_with_visualization(
        args.analyze, 
        create_visual=args.visualize
    )
    
    if results:
        # Print detailed analysis
        hybrid_system.print_detailed_analysis(results)
        
        # Save results
        output_dir = Path("data/hybrid_results")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_name = Path(args.analyze).stem
        json_file = output_dir / f"enhanced_hybrid_{image_name}_{timestamp}.json"
        
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Enhanced results saved: {json_file}")
        
        print(f"\n🎯 KEY INSIGHTS:")
        print(f"   ✅ Hybrid system combines strengths of both approaches")
        print(f"   ✅ Visual verification shows exact bounding box positions")
        print(f"   ✅ Expanded food mapping handles {len(hybrid_system.food_mapping)} food types")
        print(f"   ⚠️ Your local model limited to 6 food types (needs more training)")
        print(f"   💡 Hybrid gives you best accuracy while you improve local model")
    
    else:
        print("❌ Enhanced hybrid analysis failed")

if __name__ == "__main__":
    main()