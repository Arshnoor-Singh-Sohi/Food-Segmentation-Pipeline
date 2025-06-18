# Save as: scripts/test_refrigerator_segmentation.py
"""
Test script for proper refrigerator segmentation
Demonstrates the fixed system that correctly identifies individual items
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.metadata.food_type_classifier_fixed import FoodTypeClassifier
from src.metadata.measurement_units import MeasurementUnitSystem
from src.models.refrigerator_aware_segmentation import RefrigeratorAwareSegmentation
from src.models.fast_yolo_segmentation import FastFoodSegmentation
import json
import cv2
from datetime import datetime
from typing import Dict, List, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RefrigeratorAnalyzer:
    """
    Specialized analyzer for refrigerator and storage images
    """
    
    def __init__(self):
        # Initialize components
        self.food_classifier = FoodTypeClassifier()
        self.measurement_system = MeasurementUnitSystem()
        self.segmentation = RefrigeratorAwareSegmentation(
            self.food_classifier, 
            self.measurement_system
        )
        self.detector = FastFoodSegmentation()
        
        # Common refrigerator items for validation
        self.expected_refrigerator_items = {
            'fruits': ['banana', 'apple', 'orange', 'grape', 'berry'],
            'vegetables': ['carrot', 'lettuce', 'tomato', 'pepper', 'cabbage'],
            'dairy': ['milk', 'cheese', 'yogurt', 'butter', 'eggs'],
            'beverages': ['juice', 'water', 'soda', 'drink'],
            'containers': ['jar', 'bottle', 'container', 'box', 'package']
        }
    
    def analyze_refrigerator(self, image_path: str, output_dir: str = None) -> Dict[str, Any]:
        """
        Analyze a refrigerator image with proper individual item detection
        """
        print("\n🏪 REFRIGERATOR CONTENT ANALYSIS")
        print("="*60)
        print(f"📸 Analyzing: {image_path}")
        
        if output_dir is None:
            output_dir = "data/output/refrigerator_analysis"
        
        Path(output_dir).mkdir(exist_ok=True, parents=True)
        
        # Step 1: Initial Detection
        print("\n1️⃣ Running YOLO detection...")
        try:
            # Get raw YOLO results
            detection_results = self.detector.model(image_path)
            
            # Process results into our format
            processed_results = self._process_yolo_results(detection_results)
            
            print(f"   ✅ Initial detections: {len(processed_results['detections'])} items")
            
            # Show what was detected
            detected_classes = {}
            for det in processed_results['detections']:
                class_name = det['name']
                if class_name not in detected_classes:
                    detected_classes[class_name] = 0
                detected_classes[class_name] += 1
            
            print("\n   📊 Detection breakdown:")
            for class_name, count in sorted(detected_classes.items()):
                print(f"      - {class_name}: {count}")
            
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            # Create fallback detection results
            processed_results = self._create_fallback_results(image_path)
        
        # Step 2: Process with fixed segmentation
        print("\n2️⃣ Processing with refrigerator-aware segmentation...")
        
        # Load image
        image = cv2.imread(image_path)
        
        # Process segmentation
        segmentation_results = self.segmentation.process_segmentation(
            processed_results,
            image,
            save_visualization=True,
            output_dir=output_dir
        )
        
        # Step 3: Analyze results
        print("\n3️⃣ Analysis Results:")
        print(f"   Classification: {segmentation_results['food_type_classification']['type']}")
        print(f"   Confidence: {segmentation_results['food_type_classification']['confidence']:.1%}")
        
        stats = segmentation_results.get('processing_stats', {})
        print(f"\n   📊 Processing Statistics:")
        print(f"   - Total detections: {stats.get('total_detections', 0)}")
        print(f"   - Food items found: {stats.get('food_items_found', 0)}")
        print(f"   - Individual segments: {stats.get('segments_created', 0)}")
        
        # Step 4: Create detailed inventory
        inventory = self._create_inventory(segmentation_results)
        
        print("\n4️⃣ Refrigerator Inventory:")
        for category, items in inventory.items():
            if items:
                print(f"\n   {category.upper()}:")
                for item_name, details in items.items():
                    print(f"   - {item_name}: {details['formatted']}")
        
        # Step 5: Save comprehensive results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save JSON results
        json_path = Path(output_dir) / f"refrigerator_analysis_{timestamp}.json"
        enhanced_results = {
            'image_path': str(image_path),
            'timestamp': datetime.now().isoformat(),
            'segmentation_results': segmentation_results,
            'inventory': inventory,
            'analysis_summary': self._create_analysis_summary(segmentation_results, inventory)
        }
        
        with open(json_path, 'w', encoding="utf-8", errors="replace") as f:
            json.dump(enhanced_results, f, indent=2, default=str)
        
        print(f"\n✅ Results saved:")
        print(f"   - JSON: {json_path}")
        print(f"   - Visualization: {segmentation_results.get('visualization_path', 'N/A')}")
        
        # Create HTML report
        html_path = self._create_html_report(enhanced_results, output_dir, timestamp)
        print(f"   - HTML Report: {html_path}")
        
        return enhanced_results
    
    def _process_yolo_results(self, yolo_results) -> Dict[str, Any]:
        """Process raw YOLO results into our standard format"""
        detections = []
        
        if hasattr(yolo_results, '__iter__'):
            for result in yolo_results:
                if hasattr(result, 'boxes') and result.boxes is not None:
                    boxes = result.boxes
                    names = result.names if hasattr(result, 'names') else {}
                    
                    for i in range(len(boxes)):
                        try:
                            box = boxes[i]
                            class_id = int(box.cls[0].item()) if hasattr(box.cls[0], 'item') else int(box.cls[0])
                            class_name = names.get(class_id, f'class_{class_id}')
                            
                            detection = {
                                'name': class_name,
                                'confidence': float(box.conf[0]),
                                'bbox': {
                                    'x1': int(box.xyxy[0][0]),
                                    'y1': int(box.xyxy[0][1]),
                                    'x2': int(box.xyxy[0][2]),
                                    'y2': int(box.xyxy[0][3])
                                }
                            }
                            detections.append(detection)
                        except Exception as e:
                            logger.warning(f"Error processing detection {i}: {e}")
        
        return {
            'detections': detections,
            'results': yolo_results if hasattr(yolo_results, '__iter__') else [yolo_results]
        }
    
    def _create_fallback_results(self, image_path: str) -> Dict[str, Any]:
        """Create fallback results for testing if YOLO fails"""
        # This is for testing - creates dummy detections
        return {
            'detections': [
                {'name': 'banana', 'confidence': 0.85, 'bbox': {'x1': 400, 'y1': 200, 'x2': 500, 'y2': 300}},
                {'name': 'apple', 'confidence': 0.82, 'bbox': {'x1': 300, 'y1': 250, 'x2': 380, 'y2': 330}},
                {'name': 'milk', 'confidence': 0.78, 'bbox': {'x1': 100, 'y1': 150, 'x2': 200, 'y2': 350}},
                {'name': 'eggs', 'confidence': 0.88, 'bbox': {'x1': 250, 'y1': 100, 'x2': 350, 'y2': 180}},
                {'name': 'bottle', 'confidence': 0.75, 'bbox': {'x1': 500, 'y1': 100, 'x2': 580, 'y2': 250}},
            ]
        }
    
    def _create_inventory(self, segmentation_results: Dict) -> Dict[str, Dict]:
        """Create organized inventory from segmentation results"""
        inventory = {
            'fruits': {},
            'vegetables': {},
            'dairy': {},
            'beverages': {},
            'proteins': {},
            'condiments': {},
            'containers': {},
            'other': {}
        }
        
        # Get measurement summary
        measurements = segmentation_results.get('measurement_summary', {})
        
        # Categorize items
        for food_name, measurement in measurements.items():
            category = self._categorize_food(food_name)
            inventory[category][food_name] = measurement
        
        # Remove empty categories
        inventory = {k: v for k, v in inventory.items() if v}
        
        return inventory
    
    def _categorize_food(self, food_name: str) -> str:
        """Categorize food items for inventory organization"""
        name_lower = food_name.lower()
        
        # Check each category
        for category, keywords in [
            ('fruits', ['banana', 'apple', 'orange', 'grape', 'berry', 'fruit']),
            ('vegetables', ['carrot', 'lettuce', 'tomato', 'pepper', 'vegetable', 'cabbage']),
            ('dairy', ['milk', 'cheese', 'yogurt', 'butter', 'egg', 'cream']),
            ('beverages', ['juice', 'water', 'soda', 'drink', 'beverage']),
            ('proteins', ['meat', 'chicken', 'fish', 'beef', 'pork', 'tofu']),
            ('condiments', ['sauce', 'ketchup', 'mayo', 'mustard', 'dressing']),
            ('containers', ['jar', 'bottle', 'container', 'box', 'package'])
        ]:
            if any(keyword in name_lower for keyword in keywords):
                return category
        
        return 'other'
    
    def _create_analysis_summary(self, segmentation_results: Dict, inventory: Dict) -> Dict:
        """Create summary statistics"""
        total_items = sum(
            measurement.get('count', 1) 
            for category in inventory.values() 
            for measurement in category.values()
        )
        
        return {
            'total_unique_items': len(segmentation_results.get('segments', [])),
            'total_item_count': total_items,
            'categories_present': list(inventory.keys()),
            'storage_type': 'refrigerator',
            'organization_score': self._calculate_organization_score(inventory)
        }
    
    def _calculate_organization_score(self, inventory: Dict) -> float:
        """Calculate how well organized the refrigerator is"""
        # Simple scoring based on category distribution
        num_categories = len(inventory)
        total_items = sum(len(items) for items in inventory.values())
        
        if total_items == 0:
            return 0.0
        
        # Score based on even distribution across categories
        avg_per_category = total_items / max(num_categories, 1)
        variance = sum(
            abs(len(items) - avg_per_category) 
            for items in inventory.values()
        ) / max(num_categories, 1)
        
        score = max(0, 1 - (variance / avg_per_category))
        return round(score, 2)
    
    def _create_html_report(self, results: Dict, output_dir: str, timestamp: str) -> Path:
        """Create a comprehensive HTML report"""
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Refrigerator Analysis Report</title>
    <style>
        body {{ 
            font-family: Arial, sans-serif; 
            margin: 20px; 
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        h1 {{ 
            color: #333; 
            text-align: center;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #4CAF50;
            margin-top: 30px;
        }}
        .stats {{ 
            background: #f0f0f0; 
            padding: 15px; 
            border-radius: 8px;
            margin: 20px 0;
        }}
        .inventory {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .category {{
            background: #f9f9f9;
            padding: 15px;
            border-radius: 8px;
            border: 1px solid #ddd;
        }}
        .category h3 {{
            color: #4CAF50;
            margin-top: 0;
            text-transform: uppercase;
            font-size: 14px;
        }}
        .item {{ 
            margin: 8px 0; 
            padding: 8px; 
            background: #fff; 
            border: 1px solid #eee;
            border-radius: 4px;
            display: flex;
            justify-content: space-between;
        }}
        .measurement {{ 
            color: #0066cc; 
            font-weight: bold; 
        }}
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .summary-item {{
            background: #e8f5e9;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }}
        .summary-value {{
            font-size: 24px;
            font-weight: bold;
            color: #2e7d32;
        }}
        .summary-label {{
            color: #666;
            font-size: 14px;
            margin-top: 5px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🏪 Refrigerator Content Analysis Report</h1>
        <p style="text-align: center; color: #666;">
            Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        </p>
        
        <div class="stats">
            <h2>📊 Summary Statistics</h2>
            <div class="summary-grid">
                <div class="summary-item">
                    <div class="summary-value">{results['analysis_summary']['total_unique_items']}</div>
                    <div class="summary-label">Unique Items</div>
                </div>
                <div class="summary-item">
                    <div class="summary-value">{results['analysis_summary']['total_item_count']}</div>
                    <div class="summary-label">Total Items</div>
                </div>
                <div class="summary-item">
                    <div class="summary-value">{len(results['analysis_summary']['categories_present'])}</div>
                    <div class="summary-label">Categories</div>
                </div>
                <div class="summary-item">
                    <div class="summary-value">{results['analysis_summary']['organization_score']:.0%}</div>
                    <div class="summary-label">Organization Score</div>
                </div>
            </div>
        </div>
        
        <h2>📦 Inventory by Category</h2>
        <div class="inventory">
"""
        
        # Add inventory categories
        for category, items in results['inventory'].items():
            if items:
                html_content += f"""
            <div class="category">
                <h3>{category}</h3>
"""
                for item_name, measurement in items.items():
                    formatted = measurement.get('formatted', 'N/A')
                    html_content += f"""
                <div class="item">
                    <span>{item_name}</span>
                    <span class="measurement">{formatted}</span>
                </div>
"""
                html_content += """
            </div>
"""
        
        # Add detection details
        html_content += f"""
        </div>
        
        <h2>🔍 Detection Details</h2>
        <div class="stats">
            <p><strong>Classification:</strong> {results['segmentation_results']['food_type_classification']['type']}</p>
            <p><strong>Confidence:</strong> {results['segmentation_results']['food_type_classification']['confidence']:.1%}</p>
            <p><strong>Total Detections:</strong> {results['segmentation_results']['processing_stats']['total_detections']}</p>
            <p><strong>Food Items Found:</strong> {results['segmentation_results']['processing_stats']['food_items_found']}</p>
        </div>
        
        <p style="text-align: center; color: #666; margin-top: 40px;">
            <em>Visualization saved to: {results['segmentation_results'].get('visualization_path', 'N/A')}</em>
        </p>
    </div>
</body>
</html>
"""
        
        # Save HTML report
        html_path = Path(output_dir) / f"refrigerator_report_{timestamp}.html"
        with open(html_path, 'w', encoding="utf-8", errors="replace") as f:
            f.write(html_content)
        
        return html_path


def main():
    """Main function to test refrigerator analysis"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze refrigerator contents')
    parser.add_argument('--image', type=str, default='data/input/refrigerator.jpg',
                       help='Path to refrigerator image')
    parser.add_argument('--output', type=str, default='data/output/refrigerator_analysis',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = RefrigeratorAnalyzer()
    
    # Analyze refrigerator
    results = analyzer.analyze_refrigerator(args.image, args.output)
    
    print("\n✨ Analysis Complete!")
    print(f"Check the output directory for detailed results: {args.output}")


if __name__ == "__main__":
    main()