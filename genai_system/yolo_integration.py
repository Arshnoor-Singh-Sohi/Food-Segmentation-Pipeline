"""
YOLO + JSON Integration - Dr. Niaki's Suggestion #5
==================================================

Extract class names from GenAI JSON and use YOLO to generate bounding boxes
Add this to genai_system/yolo_integration.py

Usage:
python genai_system/yolo_integration.py --image data/input/refrigerator.jpg
"""

import json
from pathlib import Path
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

class YOLOGenAIIntegration:
    """
    Combines GenAI accuracy with YOLO bounding boxes
    Dr. Niaki's suggestion: Extract class names from JSON, input to YOLO
    """
    
    def __init__(self):
        self.yolo_model = YOLO('yolov8n.pt')  # Generic YOLO
        
        # Map GenAI specific names to YOLO classes
        self.class_mapping = {
            'banana_individual': 'banana',
            'apple_individual': 'apple', 
            'bottle_individual': 'bottle',
            'container_individual': 'bowl',
            'orange_individual': 'orange',
            'carrot_individual': 'carrot',
            'broccoli_individual': 'broccoli'
        }
    
    def combine_genai_with_yolo(self, image_path, genai_json_file):
        """
        Step 1: Get accurate item list from GenAI JSON
        Step 2: Extract class names  
        Step 3: Use YOLO to generate bounding boxes for those classes
        Step 4: Combine GenAI accuracy with YOLO precision
        """
        print(f"🔄 Combining GenAI + YOLO for: {Path(image_path).name}")
        
        # Load GenAI results
        with open(genai_json_file, 'r') as f:
            genai_results = json.load(f)
        
        # Extract class names from GenAI JSON
        genai_classes = self._extract_class_names(genai_results)
        print(f"📋 GenAI detected classes: {genai_classes}")
        
        # Map to YOLO classes
        yolo_classes = self._map_to_yolo_classes(genai_classes)
        print(f"🎯 YOLO target classes: {yolo_classes}")
        
        # Run YOLO with specific classes
        yolo_results = self._run_yolo_with_classes(image_path, yolo_classes)
        
        # Combine results
        combined_results = self._combine_results(genai_results, yolo_results)
        
        return combined_results
    
    def _extract_class_names(self, genai_results):
        """
        Extract class names from GenAI JSON response
        Dr. Niaki's suggestion: Extract class names from JSON
        """
        class_names = []
        
        # Handle different JSON formats
        if 'inventory' in genai_results:
            for item in genai_results['inventory']:
                if 'item_type' in item:
                    class_names.append(item['item_type'])
                elif 'item_name' in item:
                    class_names.append(item['item_name'])
        
        return list(set(class_names))  # Remove duplicates
    
    def _map_to_yolo_classes(self, genai_classes):
        """
        Map GenAI specific class names to YOLO class names
        """
        yolo_classes = []
        
        for genai_class in genai_classes:
            if genai_class in self.class_mapping:
                yolo_class = self.class_mapping[genai_class]
                yolo_classes.append(yolo_class)
            else:
                # Try direct mapping
                base_class = genai_class.replace('_individual', '').replace('_single', '')
                yolo_classes.append(base_class)
        
        return list(set(yolo_classes))
    
    def _run_yolo_with_classes(self, image_path, target_classes):
        """
        Run YOLO with specific target classes
        Dr. Niaki's suggestion: Input class names as parameters to YOLO
        """
        # Get YOLO class IDs for target classes
        yolo_class_ids = []
        for class_name in target_classes:
            for class_id, yolo_class in self.yolo_model.names.items():
                if yolo_class.lower() == class_name.lower():
                    yolo_class_ids.append(class_id)
        
        print(f"🎯 YOLO class IDs: {yolo_class_ids}")
        
        # Run YOLO with specific classes
        if yolo_class_ids:
            results = self.yolo_model(image_path, classes=yolo_class_ids)
        else:
            # Fallback: run on all classes
            results = self.yolo_model(image_path)
        
        # Extract bounding boxes
        yolo_detections = []
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    detection = {
                        'class_name': self.yolo_model.names[int(box.cls)],
                        'confidence': float(box.conf),
                        'bbox': box.xyxy[0].cpu().numpy().tolist()
                    }
                    yolo_detections.append(detection)
        
        return yolo_detections
    
    def _combine_results(self, genai_results, yolo_results):
        """
        Combine GenAI accuracy with YOLO bounding boxes
        Best of both worlds: GenAI accuracy + YOLO precision
        """
        combined_results = {
            'genai_accuracy': genai_results,
            'yolo_bounding_boxes': yolo_results,
            'combined_analysis': []
        }
        
        # For each GenAI detected item, try to find corresponding YOLO box
        for genai_item in genai_results.get('inventory', []):
            genai_class = genai_item.get('item_type', genai_item.get('item_name', ''))
            genai_quantity = genai_item.get('quantity', 1)
            
            # Find matching YOLO detections
            matching_boxes = []
            mapped_class = self.class_mapping.get(genai_class, genai_class.replace('_individual', ''))
            
            for yolo_detection in yolo_results:
                if yolo_detection['class_name'].lower() == mapped_class.lower():
                    matching_boxes.append(yolo_detection)
            
            # Combine information
            combined_item = {
                'item_name': genai_class,
                'genai_quantity': genai_quantity,
                'genai_confidence': genai_item.get('confidence', 0.9),
                'yolo_detections': len(matching_boxes),
                'bounding_boxes': matching_boxes,
                'accuracy_note': f"GenAI: {genai_quantity}, YOLO: {len(matching_boxes)}"
            }
            
            combined_results['combined_analysis'].append(combined_item)
        
        return combined_results
    
    def create_visualization(self, image_path, combined_results, output_path):
        """
        Create visualization showing GenAI + YOLO combined results
        """
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        plt.figure(figsize=(15, 10))
        plt.imshow(image_rgb)
        plt.title('GenAI + YOLO Combined Detection', fontsize=16, fontweight='bold')
        
        # Draw bounding boxes from YOLO
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan']
        color_idx = 0
        
        for item in combined_results['combined_analysis']:
            for bbox_info in item['bounding_boxes']:
                bbox = bbox_info['bbox']
                x1, y1, x2, y2 = bbox
                
                # Draw rectangle
                rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                   fill=False, color=colors[color_idx % len(colors)], 
                                   linewidth=2)
                plt.gca().add_patch(rect)
                
                # Add label
                label = f"{item['item_name'].replace('_', ' ').title()}"
                plt.text(x1, y1-5, label, fontsize=10, 
                        color=colors[color_idx % len(colors)],
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
                
                color_idx += 1
        
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Visualization saved: {output_path}")
    
    def print_combined_summary(self, combined_results):
        """
        Print summary of combined GenAI + YOLO results
        """
        print("\\n" + "="*60)
        print("🔄 GENAI + YOLO COMBINED RESULTS")
        print("="*60)
        
        print(f"🤖 GenAI Analysis:")
        genai_total = combined_results['genai_accuracy'].get('total_items', 0)
        print(f"   Total items detected: {genai_total}")
        
        print(f"\\n🎯 YOLO Bounding Boxes:")
        yolo_total = len(combined_results['yolo_bounding_boxes'])
        print(f"   Total bounding boxes: {yolo_total}")
        
        print(f"\\n📊 COMBINED ANALYSIS:")
        for item in combined_results['combined_analysis']:
            item_name = item['item_name'].replace('_', ' ').title()
            genai_qty = item['genai_quantity']
            yolo_boxes = item['yolo_detections']
            
            print(f"   • {item_name}:")
            print(f"     GenAI Count: {genai_qty}")
            print(f"     YOLO Boxes: {yolo_boxes}")
            print(f"     Match: {'✅' if genai_qty == yolo_boxes else '⚠️'}")
        
        print(f"\\n💡 RESULT:")
        print(f"   GenAI provides accurate counting")
        print(f"   YOLO provides precise bounding boxes")
        print(f"   Combined: Best of both worlds")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='YOLO + GenAI Integration')
    parser.add_argument('--image', type=str, default='data/input/refrigerator.jpg',
                       help='Path to image')
    parser.add_argument('--genai-json', type=str,
                       help='Path to GenAI JSON results')
    
    args = parser.parse_args()
    
    # Find latest GenAI results if not specified
    if not args.genai_json:
        results_dir = Path("data/genai_results")
        json_files = list(results_dir.glob("*.json"))
        if json_files:
            args.genai_json = max(json_files, key=lambda x: x.stat().st_mtime)
            print(f"📋 Using latest GenAI results: {args.genai_json}")
        else:
            print("❌ No GenAI results found. Run genai_analyzer.py first.")
            return
    
    # Initialize integration system
    integrator = YOLOGenAIIntegration()
    
    # Combine GenAI + YOLO
    combined_results = integrator.combine_genai_with_yolo(args.image, args.genai_json)
    
    # Print summary
    integrator.print_combined_summary(combined_results)
    
    # Create visualization
    output_path = Path("data/genai_results") / "combined_genai_yolo_visualization.png"
    integrator.create_visualization(args.image, combined_results, output_path)
    
    print(f"\\n✅ Dr. Niaki's suggestion #5 implemented!")
    print(f"GenAI accuracy + YOLO bounding boxes = Perfect combination")

if __name__ == "__main__":
    main()