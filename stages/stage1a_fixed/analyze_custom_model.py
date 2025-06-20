"""
Analyze Custom Model - Stage 1A-Fixed
====================================

Analyzes your 99.5% custom model to understand:
1. What classes it currently detects
2. How many objects it finds in refrigerator
3. Whether it can be enhanced for individual item detection
4. Next steps for fine-tuning

Usage:
python stages/stage1a_fixed/analyze_custom_model.py --model data/models/custom_food_detection.pt
"""

import argparse
import json
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
from datetime import datetime

try:
    from ultralytics import YOLO
except ImportError:
    print("❌ Ultralytics not installed. Run: pip install ultralytics")
    exit(1)

class CustomModelAnalyzer:
    def __init__(self, model_path):
        """Initialize analyzer with custom model"""
        self.model_path = model_path
        self.model = None
        self.load_model()
        
    def load_model(self):
        """Load the custom model"""
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        try:
            self.model = YOLO(self.model_path)
            print(f"✅ Loaded custom model: {self.model_path}")
        except Exception as e:
            raise Exception(f"Failed to load model: {e}")
    
    def analyze_model_capabilities(self):
        """Analyze what the model can detect"""
        print("\n🔍 CUSTOM MODEL ANALYSIS")
        print("=" * 50)
        
        # Basic model info
        model_info = {
            'model_path': str(self.model_path),
            'model_names': self.model.names,
            'num_classes': len(self.model.names),
            'class_list': list(self.model.names.values())
        }
        
        print(f"📊 Model Classes ({model_info['num_classes']} total):")
        for idx, class_name in model_info['model_names'].items():
            print(f"   {idx}: {class_name}")
        
        return model_info
    
    def test_on_refrigerator(self, image_path="data/input/refrigerator.jpg"):
        """Test model on refrigerator image"""
        print(f"\n🧊 TESTING ON REFRIGERATOR IMAGE")
        print("-" * 40)
        
        if not Path(image_path).exists():
            print(f"❌ Refrigerator image not found: {image_path}")
            return None
        
        # Run detection
        results = self.model(image_path)
        
        # Analyze results
        detections = []
        if results[0].boxes is not None:
            for box in results[0].boxes:
                detection = {
                    'class_id': int(box.cls),
                    'class_name': self.model.names[int(box.cls)],
                    'confidence': float(box.conf),
                    'bbox': box.xyxy[0].cpu().numpy().tolist()
                }
                detections.append(detection)
        
        # Count by class
        class_counts = {}
        for detection in detections:
            class_name = detection['class_name']
            if class_name not in class_counts:
                class_counts[class_name] = 0
            class_counts[class_name] += 1
        
        print(f"📊 Detection Results:")
        print(f"   Total detections: {len(detections)}")
        print(f"   Classes found: {len(class_counts)}")
        
        for class_name, count in class_counts.items():
            print(f"   {class_name}: {count} objects")
        
        return {
            'image_path': image_path,
            'total_detections': len(detections),
            'detections': detections,
            'class_counts': class_counts
        }
    
    def compare_with_generic_yolo(self, image_path="data/input/refrigerator.jpg"):
        """Compare custom model with generic YOLO"""
        print(f"\n⚖️ COMPARISON WITH GENERIC YOLO")
        print("-" * 40)
        
        if not Path(image_path).exists():
            return None
        
        # Test generic YOLO
        try:
            generic_model = YOLO('yolov8n.pt')
            generic_results = generic_model(image_path)
            
            generic_detections = []
            if generic_results[0].boxes is not None:
                for box in generic_results[0].boxes:
                    detection = {
                        'class_name': generic_model.names[int(box.cls)],
                        'confidence': float(box.conf)
                    }
                    generic_detections.append(detection)
            
            # Count generic classes
            generic_counts = {}
            for detection in generic_detections:
                class_name = detection['class_name']
                if class_name not in generic_counts:
                    generic_counts[class_name] = 0
                generic_counts[class_name] += 1
            
            print(f"📊 Generic YOLO vs Custom Model:")
            print(f"   Generic YOLO: {len(generic_detections)} detections")
            print(f"   Custom Model: {len(self.test_on_refrigerator(image_path)['detections'])} detections")
            
            print(f"\n📋 Generic YOLO Classes Found:")
            for class_name, count in generic_counts.items():
                print(f"   {class_name}: {count}")
            
            return {
                'generic_detections': len(generic_detections),
                'generic_classes': generic_counts
            }
            
        except Exception as e:
            print(f"⚠️ Could not test generic YOLO: {e}")
            return None
    
    def assess_individual_item_potential(self, refrigerator_results):
        """Assess if model can be enhanced for individual item detection"""
        print(f"\n🎯 INDIVIDUAL ITEM DETECTION ASSESSMENT")
        print("-" * 50)
        
        if not refrigerator_results:
            print("❌ No results to assess")
            return None
        
        total_detections = refrigerator_results['total_detections']
        class_counts = refrigerator_results['class_counts']
        
        # Assessment criteria
        assessment = {
            'detection_count': total_detections,
            'individual_potential': 'unknown',
            'recommendations': [],
            'next_steps': []
        }
        
        if total_detections == 0:
            assessment['individual_potential'] = 'poor'
            assessment['recommendations'].append("Model doesn't detect anything in refrigerator")
            assessment['next_steps'].append("Consider training from scratch")
            
        elif total_detections == 1:
            assessment['individual_potential'] = 'poor'
            assessment['recommendations'].append("Model only detects 1 object (same problem as generic YOLO)")
            assessment['next_steps'].append("Need to add individual item classes")
            
        elif total_detections < 5:
            assessment['individual_potential'] = 'fair'
            assessment['recommendations'].append("Model detects some items but not enough for inventory")
            assessment['next_steps'].append("Fine-tune to add more individual item classes")
            
        elif total_detections >= 5:
            assessment['individual_potential'] = 'good'
            assessment['recommendations'].append("Model detects multiple items - good foundation")
            assessment['next_steps'].append("Enhance with specific ingredient classes")
        
        # Check for individual item indicators
        individual_indicators = ['banana', 'apple', 'bottle', 'orange', 'carrot', 'tomato']
        found_indicators = []
        for class_name in class_counts.keys():
            for indicator in individual_indicators:
                if indicator.lower() in class_name.lower():
                    found_indicators.append(class_name)
        
        if found_indicators:
            assessment['recommendations'].append(f"Already detects some individual items: {found_indicators}")
            assessment['next_steps'].append("Build upon existing individual item detection")
        else:
            assessment['recommendations'].append("No individual item classes detected")
            assessment['next_steps'].append("Add individual item classes through fine-tuning")
        
        print(f"📊 Assessment Results:")
        print(f"   Individual Item Potential: {assessment['individual_potential'].upper()}")
        print(f"   Current Detection Count: {total_detections}")
        
        print(f"\n💡 Recommendations:")
        for rec in assessment['recommendations']:
            print(f"   • {rec}")
        
        print(f"\n🚀 Next Steps:")
        for step in assessment['next_steps']:
            print(f"   1. {step}")
        
        return assessment
    
    def create_enhancement_plan(self, model_info, refrigerator_results, assessment):
        """Create plan for enhancing the model"""
        print(f"\n📋 CUSTOM MODEL ENHANCEMENT PLAN")
        print("=" * 50)
        
        # Target classes for individual items
        target_classes = [
            'banana_single', 'apple_red', 'apple_green', 'orange_single',
            'bottle_milk', 'bottle_juice', 'bottle_water', 'bottle_soda',
            'tomato_single', 'carrot_single', 'lettuce_head', 'broccoli_head',
            'container_plastic', 'jar_glass', 'package_food', 'egg_single'
        ]
        
        enhancement_plan = {
            'current_model': {
                'path': str(self.model_path),
                'classes': model_info['num_classes'],
                'performance_on_refrigerator': refrigerator_results['total_detections'] if refrigerator_results else 0
            },
            'target_classes': target_classes,
            'training_strategy': '',
            'runpod_setup': {},
            'expected_improvement': ''
        }
        
        # Determine training strategy based on assessment
        if assessment and assessment['individual_potential'] in ['good', 'fair']:
            enhancement_plan['training_strategy'] = 'fine_tuning'
            enhancement_plan['expected_improvement'] = f"Increase from {refrigerator_results['total_detections']} to 10-15 individual items"
            print("🎯 Strategy: FINE-TUNING (Recommended)")
            print("   • Build upon your 99.5% accurate model")
            print("   • Add individual item classes")
            print("   • Faster training than from scratch")
            
        else:
            enhancement_plan['training_strategy'] = 'retrain_with_individual_focus'
            enhancement_plan['expected_improvement'] = "Build individual item detection from ground up"
            print("🎯 Strategy: FOCUSED RETRAINING")
            print("   • Train specifically for individual ingredients")
            print("   • Use your model architecture but retrain")
            
        # RunPod setup recommendations
        enhancement_plan['runpod_setup'] = {
            'gpu_type': 'A100 or RTX 4090',
            'training_time_estimate': '2-4 hours',
            'data_requirements': '500-1000 labeled refrigerator images',
            'batch_size': 16,
            'epochs': 50
        }
        
        print(f"\n🏃‍♂️ RunPod Training Setup:")
        print(f"   • GPU: {enhancement_plan['runpod_setup']['gpu_type']}")
        print(f"   • Time: {enhancement_plan['runpod_setup']['training_time_estimate']}")
        print(f"   • Data needed: {enhancement_plan['runpod_setup']['data_requirements']}")
        
        return enhancement_plan
    
    def save_analysis_report(self, model_info, refrigerator_results, assessment, enhancement_plan):
        """Save comprehensive analysis report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        report = {
            'timestamp': timestamp,
            'model_analysis': model_info,
            'refrigerator_test': refrigerator_results,
            'assessment': assessment,
            'enhancement_plan': enhancement_plan,
            'summary': {
                'current_performance': 'poor' if not refrigerator_results or refrigerator_results['total_detections'] < 5 else 'fair',
                'enhancement_needed': True,
                'recommended_approach': enhancement_plan['training_strategy'] if enhancement_plan else 'unknown'
            }
        }
        
        # Save report
        output_dir = Path("data/output/stage1a_fixed_analysis")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        report_file = output_dir / f"custom_model_analysis_{timestamp}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n💾 Analysis report saved: {report_file}")
        return report_file
    
    def create_visualization(self, image_path, refrigerator_results, output_dir):
        """Create visualization of current detection capabilities"""
        if not refrigerator_results or not Path(image_path).exists():
            return None
        
        # Load image
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(15, 10))
        ax.imshow(image_rgb)
        ax.set_title(f'Custom Model Detection - {len(refrigerator_results["detections"])} items found', 
                    fontsize=16, fontweight='bold')
        
        # Draw bounding boxes
        for detection in refrigerator_results['detections']:
            bbox = detection['bbox']
            x1, y1, x2, y2 = bbox
            
            # Draw box
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                               fill=False, color='blue', linewidth=2)
            ax.add_patch(rect)
            
            # Add label
            label = f"{detection['class_name']}: {detection['confidence']:.2f}"
            ax.text(x1, y1-5, label, fontsize=10, color='blue',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        ax.axis('off')
        plt.tight_layout()
        
        # Save visualization
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        viz_file = output_dir / f"custom_model_detection_{timestamp}.png"
        plt.savefig(viz_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Visualization saved: {viz_file}")
        return viz_file

def main():
    parser = argparse.ArgumentParser(description='Analyze Custom Model for Individual Item Detection')
    parser.add_argument('--model', type=str, default='data/models/custom_food_detection.pt',
                       help='Path to custom model')
    parser.add_argument('--image', type=str, default='data/input/refrigerator.jpg',
                       help='Path to test image')
    
    args = parser.parse_args()
    
    print("🔍 CUSTOM MODEL ANALYSIS - Stage 1A-Fixed")
    print("=" * 60)
    
    try:
        # Initialize analyzer
        analyzer = CustomModelAnalyzer(args.model)
        
        # Create output directory
        output_dir = Path("data/output/stage1a_fixed_analysis")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Step 1: Analyze model capabilities
        model_info = analyzer.analyze_model_capabilities()
        
        # Step 2: Test on refrigerator
        refrigerator_results = analyzer.test_on_refrigerator(args.image)
        
        # Step 3: Compare with generic YOLO
        comparison = analyzer.compare_with_generic_yolo(args.image)
        
        # Step 4: Assess individual item potential
        assessment = analyzer.assess_individual_item_potential(refrigerator_results)
        
        # Step 5: Create enhancement plan
        enhancement_plan = analyzer.create_enhancement_plan(model_info, refrigerator_results, assessment)
        
        # Step 6: Save analysis report
        report_file = analyzer.save_analysis_report(model_info, refrigerator_results, assessment, enhancement_plan)
        
        # Step 7: Create visualization
        viz_file = analyzer.create_visualization(args.image, refrigerator_results, output_dir)
        
        # Final summary
        print(f"\n🎯 FINAL RECOMMENDATIONS")
        print("=" * 60)
        
        if refrigerator_results and refrigerator_results['total_detections'] >= 5:
            print("✅ Your custom model shows promise for individual item detection")
            print("🚀 Recommended: Fine-tune with individual ingredient classes")
            print("⏱️ Estimated RunPod training time: 2-4 hours")
        elif refrigerator_results and refrigerator_results['total_detections'] > 0:
            print("⚠️ Your custom model detects some items but needs enhancement")
            print("🚀 Recommended: Add individual ingredient classes through fine-tuning")
            print("⏱️ Estimated RunPod training time: 3-6 hours")
        else:
            print("❌ Custom model not detecting individual items")
            print("🚀 Recommended: Train individual item detector from scratch")
            print("⏱️ Estimated RunPod training time: 6-12 hours")
        
        print(f"\n📁 Results saved in: {output_dir}")
        print(f"📄 Full report: {report_file}")
        if viz_file:
            print(f"📊 Visualization: {viz_file}")
        
        print(f"\n🔄 Next Steps:")
        print("1. Review the analysis report")
        print("2. Set up RunPod training environment")
        print("3. Prepare individual item training dataset")
        print("4. Fine-tune/retrain model for individual detection")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    main()