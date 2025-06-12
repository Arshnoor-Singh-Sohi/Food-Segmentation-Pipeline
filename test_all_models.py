#!/usr/bin/env python3
"""Test and compare multiple YOLO models for food detection."""

import sys
import time
import json
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Any
import shutil

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

class ModelComparison:
    """Test multiple YOLO models and compare their performance."""
    
    def __init__(self, output_dir="data/output/model_comparison"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Models to test (in order of preference)
        self.models_to_test = [
            # Segmentation Models (preferred)
            {
                'name': 'yolov8n-seg',
                'file': 'yolov8n-seg.pt',
                'type': 'segmentation',
                'description': 'YOLOv8 Nano Segmentation',
                'expected_performance': 'Good general detection + segmentation'
            },
            {
                'name': 'yolov8s-seg',
                'file': 'yolov8s-seg.pt', 
                'type': 'segmentation',
                'description': 'YOLOv8 Small Segmentation',
                'expected_performance': 'Better accuracy, slower'
            },
            {
                'name': 'yolov8m-seg',
                'file': 'yolov8m-seg.pt',
                'type': 'segmentation', 
                'description': 'YOLOv8 Medium Segmentation',
                'expected_performance': 'High accuracy, moderate speed'
            },
            
            # Detection Models
            {
                'name': 'yolov8n',
                'file': 'yolov8n.pt',
                'type': 'detection',
                'description': 'YOLOv8 Nano Detection',
                'expected_performance': 'Fast detection, no segmentation'
            },
            {
                'name': 'yolov8s',
                'file': 'yolov8s.pt',
                'type': 'detection',
                'description': 'YOLOv8 Small Detection', 
                'expected_performance': 'Good balance speed/accuracy'
            },
            
            # Specialized Models
            {
                'name': 'yolov8n-oiv7',
                'file': 'yolov8n-oiv7.pt',
                'type': 'detection',
                'description': 'YOLOv8 Open Images V7 (more classes)',
                'expected_performance': 'More object classes, may include more foods'
            },
            {
                'name': 'yolov8n-world',
                'file': 'yolov8n-world.pt',
                'type': 'detection',
                'description': 'YOLOv8 World Model',
                'expected_performance': 'Broader object recognition'
            },
            
            # YOLOv9 Models (if available)
            {
                'name': 'yolov9n',
                'file': 'yolov9n.pt',
                'type': 'detection',
                'description': 'YOLOv9 Nano (newest)',
                'expected_performance': 'Latest YOLO version'
            },
            {
                'name': 'yolov9s',
                'file': 'yolov9s.pt',
                'type': 'detection', 
                'description': 'YOLOv9 Small (newest)',
                'expected_performance': 'Latest YOLO, better accuracy'
            },
            
            # YOLOv10 Models (cutting edge)
            {
                'name': 'yolov10n',
                'file': 'yolov10n.pt',
                'type': 'detection',
                'description': 'YOLOv10 Nano (cutting edge)',
                'expected_performance': 'Newest architecture'
            }
        ]
        
        self.results = []
        self.test_images = []
        
    def setup_test_images(self, input_dir="data/input"):
        """Get test images."""
        input_path = Path(input_dir)
        if not input_path.exists():
            print(f"[FAIL] Input directory not found: {input_dir}")
            return False
            
        extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        self.test_images = []
        
        for ext in extensions:
            self.test_images.extend(list(input_path.glob(f'*{ext}')))
            self.test_images.extend(list(input_path.glob(f'*{ext.upper()}')))
        
        print(f"📸 Found {len(self.test_images)} test images")
        return len(self.test_images) > 0
    
    def test_single_model(self, model_config: Dict[str, str], test_image_path: str) -> Dict[str, Any]:
        """Test a single model on one image."""
        model_name = model_config['name']
        model_file = model_config['file']
        
        print(f"\n[TOOL] Testing {model_name}...")
        
        try:
            from ultralytics import YOLO
            
            # Create model directory
            model_output_dir = self.output_dir / model_name
            model_output_dir.mkdir(exist_ok=True)
            
            start_time = time.time()
            
            # Load model (auto-downloads if needed)
            print(f"📥 Loading {model_file}...")
            model = YOLO(model_file)
            load_time = time.time() - start_time
            
            # Test different confidence thresholds
            confidence_levels = [0.1, 0.25, 0.4, 0.6]
            best_result = None
            best_detection_count = 0
            
            for conf in confidence_levels:
                print(f"  [TARGET] Testing confidence: {conf}")
                
                inference_start = time.time()
                results = model(test_image_path, conf=conf, verbose=False)
                inference_time = time.time() - inference_start
                
                # Extract detections
                detections = []
                if results and len(results) > 0:
                    result = results[0]
                    
                    if hasattr(result, 'boxes') and result.boxes is not None:
                        boxes = result.boxes
                        for i in range(len(boxes)):
                            detection = {
                                'class_id': int(boxes.cls[i].item()),
                                'class_name': model.names[int(boxes.cls[i].item())],
                                'confidence': float(boxes.conf[i].item()),
                                'bbox': boxes.xyxy[i].tolist()
                            }
                            
                            # Add mask info if available
                            if hasattr(result, 'masks') and result.masks is not None:
                                detection['has_mask'] = True
                                detection['mask_area'] = float(np.sum(result.masks.data[i].cpu().numpy()))
                            else:
                                detection['has_mask'] = False
                            
                            detections.append(detection)
                
                # Keep track of best result
                if len(detections) > best_detection_count:
                    best_detection_count = len(detections)
                    best_result = {
                        'model_config': model_config,
                        'confidence_threshold': conf,
                        'detections': detections,
                        'detection_count': len(detections),
                        'load_time_seconds': load_time,
                        'inference_time_seconds': inference_time,
                        'total_time_seconds': load_time + inference_time
                    }
                
                print(f"    [STATS] Found {len(detections)} detections")
            
            # Save detailed results
            if best_result:
                result_file = model_output_dir / f"{Path(test_image_path).stem}_results.json"
                with open(result_file, 'w') as f:
                    json.dump(best_result, f, indent=2)
                
                # Save visualization if possible
                try:
                    self.create_visualization(test_image_path, best_result, model_output_dir)
                except Exception as e:
                    print(f"    [WARN] Visualization failed: {e}")
            
            print(f"[OK] {model_name}: {best_detection_count} detections (best conf: {best_result['confidence_threshold'] if best_result else 'N/A'})")
            return best_result or {'model_config': model_config, 'error': 'No detections found'}
            
        except Exception as e:
            error_result = {
                'model_config': model_config,
                'error': str(e),
                'failed': True
            }
            print(f"[FAIL] {model_name} failed: {e}")
            return error_result
    
    def create_visualization(self, image_path: str, result: Dict, output_dir: Path):
        """Create visualization for model results."""
        import cv2
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            return
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create visualization
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(image_rgb)
        ax.set_title(f"Model: {result['model_config']['name']} - {result['detection_count']} detections")
        ax.axis('off')
        
        # Draw detections
        colors = plt.cm.Set3(np.linspace(0, 1, 12))
        
        for i, detection in enumerate(result['detections']):
            bbox = detection['bbox']
            x1, y1, x2, y2 = bbox
            
            color = colors[i % len(colors)]
            
            # Draw bounding box
            rect = plt.Rectangle(
                (x1, y1), x2-x1, y2-y1,
                linewidth=2, edgecolor=color, facecolor='none'
            )
            ax.add_patch(rect)
            
            # Add label
            label = f"{detection['class_name']} ({detection['confidence']:.2f})"
            if detection['has_mask']:
                label += " [MASK]"
            
            ax.text(x1, y1-10, label, 
                   color=color, fontsize=10, weight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        # Save visualization
        viz_file = output_dir / f"{Path(image_path).stem}_visualization.png"
        plt.savefig(viz_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        return str(viz_file)
    
    def run_comprehensive_test(self, input_dir="data/input"):
        """Run comprehensive test on all models."""
        print("[RUN] Starting Comprehensive Model Comparison")
        print("=" * 60)
        
        # Setup test images
        if not self.setup_test_images(input_dir):
            return
        
        # Test first image with all models (for speed)
        test_image = self.test_images[0]
        print(f"[TARGET] Testing with image: {test_image.name}")
        
        all_results = []
        
        for model_config in self.models_to_test:
            try:
                result = self.test_single_model(model_config, str(test_image))
                result['test_image'] = str(test_image)
                result['timestamp'] = datetime.now().isoformat()
                all_results.append(result)
                
                # Small delay to prevent overwhelming
                time.sleep(1)
                
            except Exception as e:
                print(f"[FAIL] Failed testing {model_config['name']}: {e}")
                continue
        
        # Generate comparison report
        self.generate_comparison_report(all_results)
        
        return all_results
    
    def generate_comparison_report(self, results: List[Dict]):
        """Generate comprehensive comparison report."""
        print(f"\n[STATS] Generating Comparison Report...")
        
        # Filter successful results
        successful_results = [r for r in results if not r.get('failed', False) and 'error' not in r]
        
        if not successful_results:
            print("[FAIL] No successful model results to compare")
            return
        
        # Sort by detection count (descending)
        successful_results.sort(key=lambda x: x.get('detection_count', 0), reverse=True)
        
        # Create summary report
        summary = {
            'test_info': {
                'timestamp': datetime.now().isoformat(),
                'test_image': successful_results[0]['test_image'] if successful_results else None,
                'models_tested': len(results),
                'successful_models': len(successful_results),
                'failed_models': len(results) - len(successful_results)
            },
            'model_rankings': [],
            'detailed_results': successful_results
        }
        
        # Create rankings
        for i, result in enumerate(successful_results):
            ranking = {
                'rank': i + 1,
                'model_name': result['model_config']['name'],
                'model_type': result['model_config']['type'],
                'description': result['model_config']['description'],
                'detections_found': result.get('detection_count', 0),
                'best_confidence': result.get('confidence_threshold', 'N/A'),
                'inference_time': result.get('inference_time_seconds', 'N/A'),
                'total_time': result.get('total_time_seconds', 'N/A'),
                'detected_classes': list(set([d['class_name'] for d in result.get('detections', [])]))
            }
            summary['model_rankings'].append(ranking)
        
        # Save JSON report
        report_file = self.output_dir / f"comparison_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Create HTML report
        self.create_html_report(summary)
        
        # Print summary to console
        self.print_comparison_summary(summary)
        
        print(f"\n[FILE] Full report saved: {report_file}")
    
    def create_html_report(self, summary: Dict):
        """Create HTML comparison report."""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>YOLO Model Comparison Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
                .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; }}
                .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; text-align: center; }}
                .model-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 15px; margin: 20px 0; }}
                .model-card {{ border: 1px solid #ddd; border-radius: 8px; padding: 15px; background: white; }}
                .rank-1 {{ border-left: 4px solid #4caf50; }}
                .rank-2 {{ border-left: 4px solid #ff9800; }}
                .rank-3 {{ border-left: 4px solid #f44336; }}
                .metric {{ display: flex; justify-content: space-between; margin: 5px 0; }}
                .classes {{ background: #f8f9fa; padding: 8px; border-radius: 4px; margin-top: 10px; }}
                .summary-stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; margin: 20px 0; }}
                .stat-card {{ background: #e3f2fd; padding: 15px; border-radius: 8px; text-align: center; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🏆 YOLO Model Comparison Report</h1>
                    <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
                
                <div class="summary-stats">
                    <div class="stat-card">
                        <strong>{summary['test_info']['models_tested']}</strong><br>
                        Models Tested
                    </div>
                    <div class="stat-card">
                        <strong>{summary['test_info']['successful_models']}</strong><br>
                        Successful
                    </div>
                    <div class="stat-card">
                        <strong>{summary['test_info']['failed_models']}</strong><br>
                        Failed
                    </div>
                </div>
                
                <h2>🏅 Model Rankings</h2>
                <div class="model-grid">
        """
        
        for ranking in summary['model_rankings']:
            rank_class = f"rank-{min(ranking['rank'], 3)}"
            
            html_content += f"""
            <div class="model-card {rank_class}">
                <h3>#{ranking['rank']} {ranking['model_name']}</h3>
                <p><em>{ranking['description']}</em></p>
                
                <div class="metric">
                    <span><strong>Type:</strong></span>
                    <span>{ranking['model_type']}</span>
                </div>
                <div class="metric">
                    <span><strong>Detections:</strong></span>
                    <span>{ranking['detections_found']}</span>
                </div>
                <div class="metric">
                    <span><strong>Best Confidence:</strong></span>
                    <span>{ranking['best_confidence']}</span>
                </div>
                <div class="metric">
                    <span><strong>Inference Time:</strong></span>
                    <span>{ranking['inference_time']:.3f}s</span>
                </div>
                
                <div class="classes">
                    <strong>Detected Classes:</strong><br>
                    {', '.join(ranking['detected_classes']) if ranking['detected_classes'] else 'None'}
                </div>
            </div>
            """
        
        html_content += """
                </div>
                
                <div style="margin-top: 30px; padding: 20px; background: #f8f9fa; border-radius: 8px;">
                    <h3>[LOG] Recommendations</h3>
                    <ul>
        """
        
        if summary['model_rankings']:
            best_model = summary['model_rankings'][0]
            html_content += f"""
                        <li><strong>Best Overall:</strong> {best_model['model_name']} - {best_model['detections_found']} detections</li>
                        <li><strong>Fastest:</strong> Look for models with lowest inference time</li>
                        <li><strong>Most Accurate:</strong> Consider models with segmentation capability</li>
            """
        
        html_content += """
                    </ul>
                </div>
            </div>
        </body>
        </html>
        """
        
        html_file = self.output_dir / f"comparison_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"🌐 HTML report saved: {html_file}")
    
    def print_comparison_summary(self, summary: Dict):
        """Print summary to console."""
        print("\n" + "="*80)
        print("🏆 MODEL COMPARISON RESULTS")
        print("="*80)
        
        for ranking in summary['model_rankings'][:5]:  # Top 5
            rank_emoji = "🥇" if ranking['rank'] == 1 else "🥈" if ranking['rank'] == 2 else "🥉" if ranking['rank'] == 3 else "🏅"
            
            print(f"\n{rank_emoji} #{ranking['rank']} {ranking['model_name']}")
            print(f"   Type: {ranking['model_type']}")
            print(f"   Detections: {ranking['detections_found']}")
            print(f"   Classes: {', '.join(ranking['detected_classes'][:3])}{'...' if len(ranking['detected_classes']) > 3 else ''}")
            print(f"   Time: {ranking['inference_time']:.3f}s")
        
        if summary['model_rankings']:
            best = summary['model_rankings'][0]
            print(f"\n[TARGET] RECOMMENDATION: Use {best['model_name']}")
            print(f"   Reason: {best['detections_found']} detections found")
            print(f"   Classes detected: {', '.join(best['detected_classes'])}")
        
        print("="*80)

def main():
    """Run the comprehensive model comparison."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare multiple YOLO models")
    parser.add_argument('--input-dir', default='data/input', help="Directory with test images")
    parser.add_argument('--output-dir', default='data/output/model_comparison', help="Output directory")
    
    args = parser.parse_args()
    
    # Run comparison
    comparator = ModelComparison(args.output_dir)
    results = comparator.run_comprehensive_test(args.input_dir)
    
    if results:
        print(f"\n[OK] Comparison complete! Check {args.output_dir} for detailed results")
    else:
        print("[FAIL] Comparison failed - no results generated")

if __name__ == "__main__":
    main()