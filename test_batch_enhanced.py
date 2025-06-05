#!/usr/bin/env python3
"""Enhanced batch processing with beautiful HTML reports."""

import sys
import time
import json
import webbrowser
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

class EnhancedBatchTester:
    """Enhanced batch tester with comprehensive HTML reports."""
    
    def __init__(self):
        self.models_to_test = [
            {'name': 'yolov8n-seg.pt', 'type': 'Segmentation', 'size': 'Nano'},
            {'name': 'yolov8s-seg.pt', 'type': 'Segmentation', 'size': 'Small'},
            {'name': 'yolov8n.pt', 'type': 'Detection', 'size': 'Nano'},
            {'name': 'yolov8s.pt', 'type': 'Detection', 'size': 'Small'},
            {'name': 'yolov8n-oiv7.pt', 'type': 'Open Images', 'size': 'Nano'}
        ]
        
    def test_batch(self, input_dir="data/input", output_dir="data/output/batch_model_comparison"):
        """Test all models on all images."""
        
        # Get test images
        input_path = Path(input_dir)
        if not input_path.exists():
            print(f"❌ Input directory not found: {input_dir}")
            return
        
        extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        test_images = []
        
        for ext in extensions:
            test_images.extend(list(input_path.glob(f'*{ext}')))
            test_images.extend(list(input_path.glob(f'*{ext.upper()}')))
        
        if not test_images:
            print(f"❌ No images found in {input_dir}")
            return
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"🚀 Testing {len(self.models_to_test)} models on {len(test_images)} images")
        print("="*80)
        
        # Test each model on all images
        all_results = {}
        
        for model_config in self.models_to_test:
            model_name = model_config['name']
            print(f"\n🔧 Testing {model_name} on all images...")
            
            model_results = []
            
            for image_path in test_images:
                try:
                    result = self.test_single_model_single_image(model_config, str(image_path))
                    if result:
                        result['image_name'] = image_path.name
                        model_results.append(result)
                        print(f"  ✅ {image_path.name}: {result['count']} detections")
                    else:
                        print(f"  ❌ {image_path.name}: No detections")
                        
                except Exception as e:
                    print(f"  ❌ {image_path.name}: Failed - {e}")
                    continue
            
            all_results[model_name] = {
                'config': model_config,
                'results': model_results,
                'total_detections': sum(r['count'] for r in model_results),
                'avg_detections': np.mean([r['count'] for r in model_results]) if model_results else 0,
                'avg_time': np.mean([r['time'] for r in model_results]) if model_results else 0,
                'success_rate': len(model_results) / len(test_images) * 100
            }
            
            print(f"  📊 Total: {all_results[model_name]['total_detections']} detections across {len(model_results)} images")
        
        # Generate comprehensive reports
        self.print_batch_summary(all_results, test_images)
        html_file = self.generate_batch_html_report(all_results, test_images, output_path)
        
        # Save JSON results
        json_file = output_path / f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(json_file, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        print(f"\n📄 JSON results: {json_file}")
        print(f"🌐 HTML report: {html_file}")
        
        # Auto-open HTML report
        try:
            webbrowser.open(f'file://{html_file.absolute()}')
            print("🚀 HTML report opened in browser!")
        except:
            print("💡 Open the HTML file manually in your browser")
        
        return all_results
    
    def test_single_model_single_image(self, model_config, image_path):
        """Test a single model on a single image."""
        try:
            from ultralytics import YOLO
            
            model = YOLO(model_config['name'])
            
            # Test with confidence 0.1 for maximum detections
            start_time = time.time()
            results = model(image_path, conf=0.1, verbose=False)
            inference_time = time.time() - start_time
            
            detections = []
            if results and len(results) > 0:
                result = results[0]
                
                if hasattr(result, 'boxes') and result.boxes is not None:
                    for i in range(len(result.boxes)):
                        detection = {
                            'class_name': model.names[int(result.boxes.cls[i])],
                            'confidence': float(result.boxes.conf[i])
                        }
                        detections.append(detection)
            
            return {
                'detections': detections,
                'count': len(detections),
                'time': inference_time
            }
            
        except Exception as e:
            return None
    
    def print_batch_summary(self, all_results, test_images):
        """Print comprehensive batch summary."""
        print("\n" + "="*80)
        print("🏆 BATCH TESTING RESULTS")
        print("="*80)
        
        # Sort models by total detections
        sorted_models = sorted(all_results.items(), 
                             key=lambda x: x[1]['total_detections'], 
                             reverse=True)
        
        for i, (model_name, results) in enumerate(sorted_models, 1):
            emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "🏅"
            
            print(f"\n{emoji} #{i} {model_name}")
            print(f"   Type: {results['config']['type']} ({results['config']['size']})")
            print(f"   Total detections: {results['total_detections']}")
            print(f"   Average per image: {results['avg_detections']:.1f}")
            print(f"   Average time: {results['avg_time']:.3f}s")
            print(f"   Success rate: {results['success_rate']:.1f}%")
            
            # Most common detections
            all_detections = []
            for result in results['results']:
                all_detections.extend([d['class_name'] for d in result['detections']])
            
            from collections import Counter
            common_items = Counter(all_detections).most_common(5)
            if common_items:
                items_str = ', '.join([f"{item}({count})" for item, count in common_items])
                print(f"   Common items: {items_str}")
        
        if sorted_models:
            best_model = sorted_models[0]
            print(f"\n🎯 OVERALL RECOMMENDATION: {best_model[0]}")
            print(f"   Best overall performance with {best_model[1]['total_detections']} total detections")
            print(f"   Average {best_model[1]['avg_detections']:.1f} items per image")
        
        print("="*80)
    
    def generate_batch_html_report(self, all_results, test_images, output_dir):
        """Generate comprehensive batch HTML report."""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        html_file = output_dir / f"batch_comparison_report_{timestamp}.html"
        
        # Sort models by performance
        sorted_models = sorted(all_results.items(), 
                             key=lambda x: x[1]['total_detections'], 
                             reverse=True)
        
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>🚀 Batch Model Comparison Report</title>
            <style>
                * {{ margin: 0; padding: 0; box-sizing: border-box; }}
                
                body {{ 
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    min-height: 100vh;
                    padding: 20px;
                }}
                
                .container {{ 
                    max-width: 1600px; 
                    margin: 0 auto; 
                    background: white; 
                    border-radius: 20px;
                    box-shadow: 0 20px 40px rgba(0,0,0,0.1);
                    overflow: hidden;
                }}
                
                .header {{ 
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white; 
                    padding: 30px; 
                    text-align: center;
                }}
                
                .header h1 {{ font-size: 2.5em; margin-bottom: 10px; }}
                
                .summary-stats {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 20px;
                    padding: 30px;
                    background: #f8f9fa;
                }}
                
                .stat-card {{
                    background: white;
                    padding: 20px;
                    border-radius: 15px;
                    text-align: center;
                    border-left: 4px solid #667eea;
                }}
                
                .stat-value {{
                    font-size: 2em;
                    font-weight: bold;
                    color: #667eea;
                }}
                
                .results-section {{ padding: 30px; }}
                
                .model-comparison {{
                    display: grid;
                    gap: 25px;
                    margin-bottom: 40px;
                }}
                
                .model-summary {{
                    border: 1px solid #ddd;
                    border-radius: 15px;
                    padding: 25px;
                    background: white;
                    transition: transform 0.3s ease;
                }}
                
                .model-summary:hover {{ transform: translateY(-5px); }}
                
                .rank-1 {{ border-left: 6px solid #FFD700; background: linear-gradient(145deg, #fff9e6, #ffffff); }}
                .rank-2 {{ border-left: 6px solid #C0C0C0; background: linear-gradient(145deg, #f8f8f8, #ffffff); }}
                .rank-3 {{ border-left: 6px solid #CD7F32; background: linear-gradient(145deg, #fdf6f0, #ffffff); }}
                
                .model-header {{
                    display: flex;
                    align-items: center;
                    margin-bottom: 20px;
                }}
                
                .rank-emoji {{ font-size: 2.5em; margin-right: 15px; }}
                
                .model-metrics {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
                    gap: 15px;
                    margin: 20px 0;
                }}
                
                .metric {{
                    background: #f8f9fa;
                    padding: 15px;
                    border-radius: 10px;
                    text-align: center;
                }}
                
                .metric-value {{
                    font-size: 1.5em;
                    font-weight: bold;
                    color: #667eea;
                }}
                
                .per-image-results {{
                    margin-top: 20px;
                    padding: 20px;
                    background: #f8f9fa;
                    border-radius: 10px;
                }}
                
                .image-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 10px;
                    margin-top: 15px;
                }}
                
                .image-result {{
                    background: white;
                    padding: 10px;
                    border-radius: 8px;
                    border-left: 3px solid #4caf50;
                }}
                
                .recommendation {{
                    background: linear-gradient(135deg, #4caf50, #45a049);
                    color: white;
                    padding: 25px;
                    border-radius: 15px;
                    margin: 30px;
                    text-align: center;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🚀 Batch Model Comparison Report</h1>
                    <p>Comprehensive analysis of {len(sorted_models)} models tested on {len(test_images)} images</p>
                    <p>Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</p>
                </div>
                
                <div class="summary-stats">
                    <div class="stat-card">
                        <div class="stat-value">{len(test_images)}</div>
                        <div>Images Tested</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{len(sorted_models)}</div>
                        <div>Models Compared</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{max(model_data['total_detections'] for _, model_data in sorted_models) if sorted_models else 0}</div>
                        <div>Max Detections</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{sorted_models[0][0].split('.')[0] if sorted_models else 'None'}</div>
                        <div>Best Model</div>
                    </div>
                </div>
        """
        
        if sorted_models:
            best_model = sorted_models[0]
            html_content += f"""
                <div class="recommendation">
                    <h2>🎯 Recommended Model for Batch Processing</h2>
                    <h3>{best_model[0]}</h3>
                    <p>Total detections: {best_model[1]['total_detections']} across {len(test_images)} images</p>
                    <p>Average: {best_model[1]['avg_detections']:.1f} detections per image</p>
                    <p>Average processing time: {best_model[1]['avg_time']:.3f} seconds per image</p>
                </div>
            """
        
        html_content += """
                <div class="results-section">
                    <h2>🏅 Detailed Model Comparison</h2>
                    <div class="model-comparison">
        """
        
        # Add detailed model results
        for i, (model_name, model_data) in enumerate(sorted_models, 1):
            emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "🏅"
            rank_class = f"rank-{min(i, 3)}" if i <= 3 else ""
            
            html_content += f"""
                        <div class="model-summary {rank_class}">
                            <div class="model-header">
                                <div class="rank-emoji">{emoji}</div>
                                <div>
                                    <h3>#{i} {model_name}</h3>
                                    <p>{model_data['config']['type']} - {model_data['config']['size']} Size</p>
                                </div>
                            </div>
                            
                            <div class="model-metrics">
                                <div class="metric">
                                    <div class="metric-value">{model_data['total_detections']}</div>
                                    <div>Total Detections</div>
                                </div>
                                <div class="metric">
                                    <div class="metric-value">{model_data['avg_detections']:.1f}</div>
                                    <div>Avg per Image</div>
                                </div>
                                <div class="metric">
                                    <div class="metric-value">{model_data['avg_time']:.3f}s</div>
                                    <div>Avg Time</div>
                                </div>
                                <div class="metric">
                                    <div class="metric-value">{model_data['success_rate']:.1f}%</div>
                                    <div>Success Rate</div>
                                </div>
                            </div>
                            
                            <div class="per-image-results">
                                <h4>📊 Per-Image Results</h4>
                                <div class="image-grid">
            """
            
            for result in model_data['results']:
                html_content += f"""
                                    <div class="image-result">
                                        <strong>{result['image_name']}</strong><br>
                                        {result['count']} detections<br>
                                        <small>{result['time']:.3f}s</small>
                                    </div>
                """
            
            html_content += """
                                </div>
                            </div>
                        </div>
            """
        
        html_content += """
                    </div>
                </div>
                
                <div style="text-align: center; padding: 30px; background: #f8f9fa; color: #666;">
                    <p>Generated by Enhanced YOLO Batch Comparison Tool</p>
                    <p style="margin-top: 10px;">🚀 Powered by Ultralytics YOLO</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        # Save HTML file
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return html_file

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced batch model comparison")
    parser.add_argument('--input-dir', default='data/input', help="Directory with test images")
    parser.add_argument('--output-dir', default='data/output/batch_model_comparison', help="Output directory")
    
    args = parser.parse_args()
    
    tester = EnhancedBatchTester()
    tester.test_batch(args.input_dir, args.output_dir)

if __name__ == "__main__":
    main()