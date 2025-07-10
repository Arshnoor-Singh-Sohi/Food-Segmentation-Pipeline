#!/usr/bin/env python3
"""
Local Model Testing Suite
========================

Comprehensive testing for your trained local model vs GenAI.
Tests performance, accuracy, speed, and provides recommendations.

Usage:
python test_local_model.py --basic
python test_local_model.py --compare  
python test_local_model.py --visual
python test_local_model.py --full-analysis
"""

import argparse
import time
import json
from pathlib import Path
from datetime import datetime
import sys

# Add genai_system to path
sys.path.append(str(Path(__file__).parent / "genai_system"))

try:
    from ultralytics import YOLO
except ImportError:
    print("❌ Ultralytics not installed. Run: pip install ultralytics")
    sys.exit(1)

class LocalModelTester:
    """
    Comprehensive testing suite for your trained local model
    """
    
    def __init__(self):
        self.local_model_path = "data/models/genai_trained_local_model2/weights/best.pt"
        self.test_image = "data/input/refrigerator.jpg"
        self.results_dir = Path("data/model_test_results")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Load local model
        try:
            self.local_model = YOLO(self.local_model_path)
            print(f"✅ Local model loaded: {self.local_model_path}")
        except Exception as e:
            print(f"❌ Failed to load local model: {e}")
            self.local_model = None
    
    def test_basic_detection(self):
        """Test basic detection functionality"""
        print("\n🧪 BASIC DETECTION TEST")
        print("=" * 50)
        
        if not self.local_model:
            print("❌ Local model not available")
            return None
        
        if not Path(self.test_image).exists():
            print(f"❌ Test image not found: {self.test_image}")
            return None
        
        print(f"📸 Testing image: {self.test_image}")
        
        # Run detection
        start_time = time.time()
        results = self.local_model(self.test_image, conf=0.25)
        detection_time = time.time() - start_time
        
        # Parse results
        detections = []
        total_items = 0
        
        for r in results:
            if r.boxes is not None:
                for box in r.boxes:
                    class_id = int(box.cls)
                    class_name = self.local_model.names[class_id]
                    confidence = float(box.conf)
                    bbox = box.xyxy[0].cpu().numpy().tolist()
                    
                    detections.append({
                        'class_name': class_name,
                        'confidence': confidence,
                        'bbox': bbox
                    })
                    total_items += 1
        
        # Print results
        print(f"\n📊 DETECTION RESULTS:")
        print(f"   Processing time: {detection_time:.2f} seconds")
        print(f"   Total items detected: {total_items}")
        
        if detections:
            # Group by class
            class_counts = {}
            for det in detections:
                class_name = det['class_name']
                if class_name not in class_counts:
                    class_counts[class_name] = []
                class_counts[class_name].append(det['confidence'])
            
            print(f"\n📋 BY FOOD TYPE:")
            for class_name, confidences in class_counts.items():
                count = len(confidences)
                avg_conf = sum(confidences) / count
                print(f"   ✅ {class_name}: {count} items (avg confidence: {avg_conf:.2f})")
        else:
            print("   ❌ No items detected")
        
        # Save results
        results_data = {
            'test_type': 'basic_detection',
            'timestamp': datetime.now().isoformat(),
            'image_path': self.test_image,
            'model_path': self.local_model_path,
            'processing_time': detection_time,
            'total_detections': total_items,
            'detections': detections,
            'class_summary': class_counts if detections else {}
        }
        
        results_file = self.results_dir / f"basic_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"\n💾 Results saved: {results_file}")
        return results_data
    
    def compare_with_genai(self):
        """Compare local model with GenAI results"""
        print("\n⚖️ LOCAL MODEL vs GENAI COMPARISON")
        print("=" * 50)
        
        # Test local model
        print("🔄 Testing Local Model (FREE)...")
        local_results = self.test_basic_detection()
        
        if not local_results:
            print("❌ Local model test failed")
            return None
        
        # Test GenAI
        print("\n🔄 Testing GenAI (EXPENSIVE)...")
        try:
            from genai_analyzer import GenAIAnalyzer
            genai_analyzer = GenAIAnalyzer()
            genai_results = genai_analyzer.analyze_refrigerator(self.test_image)
            
            if genai_results:
                print("✅ GenAI analysis complete")
                genai_total = genai_results.get('total_items', 0)
                genai_inventory = genai_results.get('inventory', [])
            else:
                print("❌ GenAI analysis failed")
                return None
                
        except Exception as e:
            print(f"❌ GenAI test failed: {e}")
            print("💡 Make sure you have API credits and genai_analyzer.py")
            return None
        
        # Compare results
        print(f"\n📊 COMPARISON RESULTS:")
        print(f"{'Metric':<20} {'Local Model':<15} {'GenAI':<15} {'Winner'}")
        print("-" * 65)
        
        local_total = local_results['total_detections']
        local_time = local_results['processing_time']
        genai_time = 2.5  # Approximate GenAI time
        
        print(f"{'Total Items':<20} {local_total:<15} {genai_total:<15} {'GenAI' if genai_total > local_total else 'Local' if local_total > genai_total else 'Tie'}")
        print(f"{'Processing Time':<20} {local_time:.2f}s{'':<9} {genai_time:.2f}s{'':<9} {'Local' if local_time < genai_time else 'GenAI'}")
        print(f"{'Cost per Image':<20} {'$0.00':<15} {'$0.02':<15} {'Local'}")
        print(f"{'Internet Required':<20} {'No':<15} {'Yes':<15} {'Local'}")
        
        # Detailed comparison
        print(f"\n🔍 DETAILED FOOD COMPARISON:")
        local_classes = local_results.get('class_summary', {})
        
        for genai_item in genai_inventory:
            item_type = genai_item.get('item_type', '').replace('_individual', '')
            genai_qty = genai_item.get('quantity', 0)
            genai_conf = genai_item.get('confidence', 0)
            
            # Find matching local detection
            local_qty = 0
            local_conf = 0
            for local_class, local_confs in local_classes.items():
                if item_type in local_class or local_class in item_type:
                    local_qty = len(local_confs)
                    local_conf = sum(local_confs) / len(local_confs)
                    break
            
            match_status = "✅" if abs(local_qty - genai_qty) <= 1 else "⚠️" if abs(local_qty - genai_qty) <= 2 else "❌"
            print(f"   {match_status} {item_type}: Local={local_qty} | GenAI={genai_qty}")
        
        # Calculate overall accuracy
        matches = 0
        total_comparisons = 0
        for genai_item in genai_inventory:
            item_type = genai_item.get('item_type', '').replace('_individual', '')
            genai_qty = genai_item.get('quantity', 0)
            
            for local_class, local_confs in local_classes.items():
                if item_type in local_class or local_class in item_type:
                    local_qty = len(local_confs)
                    if abs(local_qty - genai_qty) <= 1:  # Allow ±1 difference
                        matches += 1
                    total_comparisons += 1
                    break
        
        accuracy = (matches / total_comparisons * 100) if total_comparisons > 0 else 0
        
        print(f"\n🎯 OVERALL ASSESSMENT:")
        print(f"   Local Model Accuracy vs GenAI: {accuracy:.1f}%")
        
        if accuracy >= 80:
            print(f"   🎉 EXCELLENT! Your model is competitive with GenAI")
            recommendation = "deploy"
        elif accuracy >= 60:
            print(f"   ✅ GOOD! Your model is working well")
            recommendation = "minor_improvements"
        else:
            print(f"   ⚠️ NEEDS IMPROVEMENT: Consider more training data")
            recommendation = "major_improvements"
        
        # Save comparison
        comparison_data = {
            'test_type': 'genai_comparison',
            'timestamp': datetime.now().isoformat(),
            'local_results': local_results,
            'genai_results': genai_results,
            'accuracy_vs_genai': accuracy,
            'recommendation': recommendation
        }
        
        comparison_file = self.results_dir / f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(comparison_file, 'w') as f:
            json.dump(comparison_data, f, indent=2)
        
        print(f"\n💾 Comparison saved: {comparison_file}")
        return comparison_data
    
    def create_visual_results(self):
        """Create visual detection results"""
        print("\n📊 CREATING VISUAL RESULTS")
        print("=" * 50)
        
        if not self.local_model:
            print("❌ Local model not available")
            return None
        
        # Create visual results with different confidence thresholds
        confidence_levels = [0.25, 0.5, 0.75]
        
        for conf in confidence_levels:
            print(f"🎯 Creating visualization with confidence {conf}")
            
            # Run detection with visualization
            results = self.local_model(
                self.test_image, 
                save=True, 
                conf=conf,
                project=str(self.results_dir),
                name=f"visual_conf_{conf}",
                exist_ok=True
            )
            
            print(f"   ✅ Saved to: {self.results_dir}/visual_conf_{conf}/")
        
        print(f"\n📁 All visual results saved in: {self.results_dir}/")
        print("💡 Open the image files to see bounding boxes and labels")
        
        return True
    
    def full_performance_analysis(self):
        """Complete performance analysis with recommendations"""
        print("\n🔬 FULL PERFORMANCE ANALYSIS")
        print("=" * 50)
        
        # Run all tests
        basic_results = self.test_basic_detection()
        if not basic_results:
            print("❌ Basic test failed, cannot continue analysis")
            return None
        
        comparison_results = self.compare_with_genai()
        self.create_visual_results()
        
        # Analyze model performance
        print(f"\n📈 PERFORMANCE ANALYSIS:")
        
        total_detections = basic_results['total_detections']
        processing_time = basic_results['processing_time']
        class_summary = basic_results.get('class_summary', {})
        
        print(f"   Detection Count: {total_detections}")
        print(f"   Processing Speed: {processing_time:.2f}s")
        print(f"   Classes Detected: {len(class_summary)}")
        
        # Speed analysis
        if processing_time < 1.0:
            speed_rating = "🚀 Excellent"
        elif processing_time < 3.0:
            speed_rating = "✅ Good"
        else:
            speed_rating = "⚠️ Slow"
        
        print(f"   Speed Rating: {speed_rating}")
        
        # Detection analysis
        if total_detections > 15:
            detection_rating = "🎯 High"
        elif total_detections > 5:
            detection_rating = "✅ Moderate"
        else:
            detection_rating = "⚠️ Low"
        
        print(f"   Detection Rating: {detection_rating}")
        
        # Generate recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        
        recommendations = []
        
        if total_detections < 10:
            recommendations.append("📸 Collect more training images (current: low detection)")
        
        if len(class_summary) < 4:
            recommendations.append("🏷️ Add more food variety to training data")
        
        if processing_time > 2.0:
            recommendations.append("⚡ Consider GPU training for faster inference")
        
        if comparison_results and comparison_results.get('accuracy_vs_genai', 0) < 70:
            recommendations.append("🎯 Increase training epochs (current model: 65.5% mAP50)")
        
        if not recommendations:
            recommendations.append("🎉 Model performing well! Ready for production use")
        
        for rec in recommendations:
            print(f"   {rec}")
        
        # Cost analysis
        print(f"\n💰 COST ANALYSIS:")
        print(f"   With 1000 images/month:")
        print(f"   GenAI Cost: $20.00")
        print(f"   Your Model: $0.00")
        print(f"   Annual Savings: $240.00")
        
        # Final assessment
        overall_score = 0
        if total_detections >= 10: overall_score += 25
        if processing_time <= 2.0: overall_score += 25
        if len(class_summary) >= 4: overall_score += 25
        if comparison_results and comparison_results.get('accuracy_vs_genai', 0) >= 70: overall_score += 25
        
        print(f"\n🏆 OVERALL MODEL SCORE: {overall_score}/100")
        
        if overall_score >= 75:
            final_recommendation = "🎉 DEPLOY! Your model is ready for production"
        elif overall_score >= 50:
            final_recommendation = "✅ GOOD! Minor improvements recommended"
        else:
            final_recommendation = "🔧 NEEDS WORK! Consider retraining with more data"
        
        print(f"   {final_recommendation}")
        
        # Save complete analysis
        analysis_data = {
            'test_type': 'full_analysis',
            'timestamp': datetime.now().isoformat(),
            'basic_results': basic_results,
            'comparison_results': comparison_results,
            'performance_metrics': {
                'detection_count': total_detections,
                'processing_time': processing_time,
                'classes_detected': len(class_summary),
                'speed_rating': speed_rating,
                'detection_rating': detection_rating
            },
            'recommendations': recommendations,
            'overall_score': overall_score,
            'final_recommendation': final_recommendation
        }
        
        analysis_file = self.results_dir / f"full_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(analysis_file, 'w') as f:
            json.dump(analysis_data, f, indent=2)
        
        print(f"\n💾 Complete analysis saved: {analysis_file}")
        return analysis_data

def main():
    parser = argparse.ArgumentParser(description='Local Model Testing Suite')
    parser.add_argument('--basic', action='store_true',
                       help='Run basic detection test')
    parser.add_argument('--compare', action='store_true',
                       help='Compare local model with GenAI')
    parser.add_argument('--visual', action='store_true',
                       help='Create visual detection results')
    parser.add_argument('--full-analysis', action='store_true',
                       help='Run complete performance analysis')
    parser.add_argument('--model-path', type=str,
                       help='Path to trained model (default: auto-detect)')
    parser.add_argument('--test-image', type=str,
                       help='Path to test image (default: data/input/refrigerator.jpg)')
    
    args = parser.parse_args()
    
    print("🧪 LOCAL MODEL TESTING SUITE")
    print("=" * 60)
    
    # Initialize tester
    tester = LocalModelTester()
    
    if args.model_path:
        tester.local_model_path = args.model_path
        tester.local_model = YOLO(args.model_path)
    
    if args.test_image:
        tester.test_image = args.test_image
    
    # Run requested tests
    if args.basic:
        tester.test_basic_detection()
    
    elif args.compare:
        tester.compare_with_genai()
    
    elif args.visual:
        tester.create_visual_results()
    
    elif args.full_analysis:
        tester.full_performance_analysis()
    
    else:
        print("Usage:")
        print("  python test_local_model.py --basic          # Basic detection test")
        print("  python test_local_model.py --compare        # Compare with GenAI")
        print("  python test_local_model.py --visual         # Create visual results")
        print("  python test_local_model.py --full-analysis  # Complete analysis")
        print("\nRecommended: Start with --basic, then --compare")

if __name__ == "__main__":
    main()