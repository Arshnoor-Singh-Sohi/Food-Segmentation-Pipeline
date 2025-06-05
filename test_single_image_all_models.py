#!/usr/bin/env python3
"""Test a single specific image with all models."""

import sys
import time
import json
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_single_image_all_models(image_path, output_dir="data/output/model_comparison"):
    """Test a single image with all available models."""
    
    if not Path(image_path).exists():
        print(f"❌ Image not found: {image_path}")
        return
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Models to test
    models_to_test = [
        'yolov8n-seg.pt',
        'yolov8s-seg.pt', 
        'yolov8n.pt',
        'yolov8s.pt',
        'yolov8n-oiv7.pt',
        'yolov8n-world.pt'
    ]
    
    print(f"🎯 Testing image: {Path(image_path).name}")
    print("="*60)
    
    results = []
    
    for model_name in models_to_test:
        print(f"\n🔧 Testing {model_name}...")
        
        try:
            from ultralytics import YOLO
            
            # Load model
            model = YOLO(model_name)
            
            # Test different confidence levels
            best_result = None
            best_count = 0
            
            for conf in [0.1, 0.25, 0.4, 0.6]:
                start_time = time.time()
                detection_results = model(image_path, conf=conf, verbose=False)
                inference_time = time.time() - start_time
                
                # Count detections
                detection_count = 0
                detections = []
                
                if detection_results and len(detection_results) > 0:
                    result = detection_results[0]
                    
                    if hasattr(result, 'boxes') and result.boxes is not None:
                        detection_count = len(result.boxes)
                        
                        # Extract detection details
                        for i in range(len(result.boxes)):
                            detection = {
                                'class_name': model.names[int(result.boxes.cls[i])],
                                'confidence': float(result.boxes.conf[i]),
                                'bbox': result.boxes.xyxy[i].tolist()
                            }
                            detections.append(detection)
                
                if detection_count > best_count:
                    best_count = detection_count
                    best_result = {
                        'model': model_name,
                        'confidence': conf,
                        'detections': detections,
                        'count': detection_count,
                        'time': inference_time
                    }
                
                print(f"  Conf {conf}: {detection_count} detections")
            
            if best_result:
                results.append(best_result)
                print(f"✅ Best: {best_count} detections at conf {best_result['confidence']}")
                
                # Show what was detected
                if best_result['detections']:
                    detected_items = [d['class_name'] for d in best_result['detections']]
                    print(f"   Found: {', '.join(detected_items)}")
            else:
                print("❌ No detections found")
                
        except Exception as e:
            print(f"❌ {model_name} failed: {e}")
    
    # Sort results by detection count
    results.sort(key=lambda x: x['count'], reverse=True)
    
    # Print summary
    print("\n" + "="*60)
    print("🏆 FINAL RESULTS")
    print("="*60)
    
    for i, result in enumerate(results[:5], 1):
        emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "🏅"
        print(f"{emoji} #{i} {result['model']}")
        print(f"   Detections: {result['count']}")
        print(f"   Best confidence: {result['confidence']}")
        print(f"   Speed: {result['time']:.3f}s")
        if result['detections']:
            items = [d['class_name'] for d in result['detections']]
            print(f"   Items: {', '.join(items)}")
        print()
    
    # Save results
    results_file = output_path / f"single_image_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump({
            'image_path': image_path,
            'timestamp': datetime.now().isoformat(),
            'results': results
        }, f, indent=2)
    
    print(f"📄 Results saved: {results_file}")
    
    # Recommendation
    if results:
        best = results[0]
        print(f"\n🎯 RECOMMENDATION: Use {best['model']}")
        print(f"   Found {best['count']} items with confidence {best['confidence']}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        test_single_image_all_models(image_path)
    else:
        print("Usage: python test_single_image_all_models.py <image_path>")