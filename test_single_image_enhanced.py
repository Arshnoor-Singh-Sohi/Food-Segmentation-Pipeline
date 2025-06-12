#!/usr/bin/env python3
"""Enhanced single image test with beautiful HTML reports."""

import sys
import time
import json
import webbrowser
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

class EnhancedSingleImageTester:
    """Enhanced single image tester with beautiful HTML reports."""
    
    def __init__(self):
        self.models_to_test = [
            {'name': 'yolov8n-seg.pt', 'type': 'Segmentation', 'size': 'Nano'},
            {'name': 'yolov8s-seg.pt', 'type': 'Segmentation', 'size': 'Small'},
            {'name': 'yolov8m-seg.pt', 'type': 'Segmentation', 'size': 'Medium'},
            {'name': 'yolov8n.pt', 'type': 'Detection', 'size': 'Nano'},
            {'name': 'yolov8s.pt', 'type': 'Detection', 'size': 'Small'},
            {'name': 'yolov8n-oiv7.pt', 'type': 'Open Images', 'size': 'Nano'},
            {'name': 'yolov8n-world.pt', 'type': 'World Model', 'size': 'Nano'},
            {'name': 'yolov9n.pt', 'type': 'YOLOv9', 'size': 'Nano'},
            {'name': 'yolov10n.pt', 'type': 'YOLOv10', 'size': 'Nano'}
        ]
        
    def test_all_models(self, image_path, output_dir="data/output/model_comparison"):
        """Test all models and generate comprehensive reports."""
        
        if not Path(image_path).exists():
            print(f"[FAIL] Image not found: {image_path}")
            return
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"[TARGET] Testing image: {Path(image_path).name}")
        print("="*60)
        
        results = []
        
        for model_config in self.models_to_test:
            model_name = model_config['name']
            print(f"\n[TOOL] Testing {model_name}...")
            
            try:
                result = self.test_single_model(model_config, image_path)
                if result:
                    results.append(result)
                    
            except Exception as e:
                print(f"[FAIL] {model_name} failed: {e}")
                continue
        
        # Sort results by detection count
        results.sort(key=lambda x: x['count'], reverse=True)
        
        # Print console summary (keep the nice console output)
        self.print_console_summary(results)
        
        # Generate HTML report
        html_file = self.generate_html_report(results, image_path, output_path)
        
        # Save JSON results
        json_file = output_path / f"single_image_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(json_file, 'w') as f:
            json.dump({
                'image_path': image_path,
                'timestamp': datetime.now().isoformat(),
                'results': results
            }, f, indent=2)
        
        print(f"[FILE] JSON results: {json_file}")
        print(f"🌐 HTML report: {html_file}")
        
        # Auto-open HTML report
        try:
            webbrowser.open(f'file://{html_file.absolute()}')
            print("[RUN] HTML report opened in browser!")
        except:
            print("[TIP] Open the HTML file manually in your browser")
        
        return results
    
    def test_single_model(self, model_config, image_path):
        """Test a single model."""
        model_name = model_config['name']
        
        try:
            from ultralytics import YOLO
            
            # Load model
            model = YOLO(model_name)
            
            # Test different confidence levels
            best_result = None
            best_count = 0
            
            confidence_tests = []
            
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
                
                confidence_test = {
                    'confidence': conf,
                    'detections': detections,
                    'count': detection_count,
                    'time': inference_time
                }
                confidence_tests.append(confidence_test)
                
                if detection_count > best_count:
                    best_count = detection_count
                    best_result = confidence_test.copy()
                
                print(f"  Conf {conf}: {detection_count} detections")
            
            if best_result:
                final_result = {
                    'model_config': model_config,
                    'model': model_name,
                    'best_result': best_result,
                    'confidence_tests': confidence_tests,
                    'count': best_count,
                    'time': best_result['time'],
                    'confidence': best_result['confidence'],
                    'detections': best_result['detections']
                }
                
                print(f"[OK] Best: {best_count} detections at conf {best_result['confidence']}")
                
                if best_result['detections']:
                    detected_items = [d['class_name'] for d in best_result['detections']]
                    print(f"   Found: {', '.join(detected_items)}")
                
                return final_result
            else:
                print("[FAIL] No detections found")
                return None
                
        except Exception as e:
            print(f"[FAIL] {model_name} failed: {e}")
            return None
    
    def print_console_summary(self, results):
        """Print the nice console summary."""
        print("\n" + "="*60)
        print("🏆 FINAL RESULTS")
        print("="*60)
        
        for i, result in enumerate(results[:10], 1):
            emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "🏅"
            print(f"{emoji} #{i} {result['model']}")
            print(f"   Detections: {result['count']}")
            print(f"   Best confidence: {result['confidence']}")
            print(f"   Speed: {result['time']:.3f}s")
            print(f"   Type: {result['model_config']['type']} ({result['model_config']['size']})")
            if result['detections']:
                items = [d['class_name'] for d in result['detections']]
                print(f"   Items: {', '.join(items)}")
            print()
        
        # Recommendation
        if results:
            best = results[0]
            print(f"[TARGET] RECOMMENDATION: Use {best['model']}")
            print(f"   Found {best['count']} items with confidence {best['confidence']}")
            print("="*60)
    
    def generate_html_report(self, results, image_path, output_dir):
        """Generate beautiful HTML report."""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        html_file = output_dir / f"model_comparison_report_{timestamp}.html"
        
        # Get image for display
        image_name = Path(image_path).name
        
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>🏆 YOLO Model Comparison Report</title>
            <style>
                * {{ margin: 0; padding: 0; box-sizing: border-box; }}
                
                body {{ 
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    min-height: 100vh;
                    padding: 20px;
                }}
                
                .container {{ 
                    max-width: 1400px; 
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
                .header p {{ font-size: 1.1em; opacity: 0.9; }}
                
                .test-info {{
                    background: #f8f9fa;
                    padding: 20px;
                    border-bottom: 1px solid #eee;
                }}
                
                .test-info-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                    gap: 20px;
                }}
                
                .info-card {{
                    background: white;
                    padding: 15px;
                    border-radius: 10px;
                    border-left: 4px solid #667eea;
                }}
                
                .results-container {{ padding: 30px; }}
                
                .ranking-grid {{
                    display: grid;
                    gap: 20px;
                }}
                
                .model-card {{
                    border: 1px solid #ddd;
                    border-radius: 15px;
                    padding: 25px;
                    background: white;
                    transition: transform 0.3s ease, box-shadow 0.3s ease;
                    position: relative;
                }}
                
                .model-card:hover {{
                    transform: translateY(-5px);
                    box-shadow: 0 10px 30px rgba(0,0,0,0.1);
                }}
                
                .rank-1 {{ border-left: 6px solid #FFD700; background: linear-gradient(145deg, #fff9e6, #ffffff); }}
                .rank-2 {{ border-left: 6px solid #C0C0C0; background: linear-gradient(145deg, #f8f8f8, #ffffff); }}
                .rank-3 {{ border-left: 6px solid #CD7F32; background: linear-gradient(145deg, #fdf6f0, #ffffff); }}
                .rank-other {{ border-left: 6px solid #4caf50; }}
                
                .rank-badge {{
                    position: absolute;
                    top: -10px;
                    right: 20px;
                    background: #667eea;
                    color: white;
                    padding: 5px 15px;
                    border-radius: 20px;
                    font-weight: bold;
                    font-size: 0.9em;
                }}
                
                .rank-1 .rank-badge {{ background: #FFD700; color: #333; }}
                .rank-2 .rank-badge {{ background: #C0C0C0; color: #333; }}
                .rank-3 .rank-badge {{ background: #CD7F32; color: white; }}
                
                .model-header {{
                    display: flex;
                    align-items: center;
                    margin-bottom: 20px;
                }}
                
                .model-emoji {{ font-size: 2.5em; margin-right: 15px; }}
                .model-title {{ flex: 1; }}
                .model-title h3 {{ font-size: 1.5em; color: #333; margin-bottom: 5px; }}
                .model-title p {{ color: #666; }}
                
                .metrics-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
                    gap: 15px;
                    margin: 20px 0;
                }}
                
                .metric {{
                    text-align: center;
                    padding: 15px;
                    background: #f8f9fa;
                    border-radius: 10px;
                }}
                
                .metric-value {{
                    font-size: 1.8em;
                    font-weight: bold;
                    color: #667eea;
                    display: block;
                }}
                
                .metric-label {{
                    font-size: 0.9em;
                    color: #666;
                    margin-top: 5px;
                }}
                
                .detections-section {{
                    margin-top: 20px;
                    padding: 20px;
                    background: #f8f9fa;
                    border-radius: 10px;
                }}
                
                .detections-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 10px;
                    margin-top: 15px;
                }}
                
                .detection-item {{
                    background: white;
                    padding: 10px 15px;
                    border-radius: 8px;
                    border-left: 3px solid #4caf50;
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                }}
                
                .confidence-tests {{
                    margin-top: 20px;
                    padding: 20px;
                    background: #f1f3f4;
                    border-radius: 10px;
                }}
                
                .confidence-grid {{
                    display: grid;
                    grid-template-columns: repeat(4, 1fr);
                    gap: 10px;
                    margin-top: 15px;
                }}
                
                .confidence-card {{
                    background: white;
                    padding: 15px;
                    border-radius: 8px;
                    text-align: center;
                    border: 2px solid transparent;
                }}
                
                .confidence-card.best {{ border-color: #4caf50; background: #e8f5e8; }}
                
                .recommendation {{
                    background: linear-gradient(135deg, #4caf50, #45a049);
                    color: white;
                    padding: 25px;
                    border-radius: 15px;
                    margin: 30px 0;
                    text-align: center;
                }}
                
                .recommendation h2 {{ margin-bottom: 15px; }}
                .recommendation p {{ font-size: 1.1em; opacity: 0.9; }}
                
                @media (max-width: 768px) {{
                    .metrics-grid {{ grid-template-columns: repeat(2, 1fr); }}
                    .confidence-grid {{ grid-template-columns: repeat(2, 1fr); }}
                    .header h1 {{ font-size: 1.8em; }}
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🏆 YOLO Model Comparison Report</h1>
                    <p>Comprehensive analysis of {len(results)} models tested on {image_name}</p>
                    <p>Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</p>
                </div>
                
                <div class="test-info">
                    <h2 style="margin-bottom: 20px;">[STATS] Test Summary</h2>
                    <div class="test-info-grid">
                        <div class="info-card">
                            <h4>[TARGET] Test Image</h4>
                            <p>{image_name}</p>
                        </div>
                        <div class="info-card">
                            <h4>[TOOL] Models Tested</h4>
                            <p>{len(results)} successful models</p>
                        </div>
                        <div class="info-card">
                            <h4>🏅 Best Performance</h4>
                            <p>{results[0]['model'] if results else 'None'}</p>
                        </div>
                        <div class="info-card">
                            <h4>🔍 Max Detections</h4>
                            <p>{results[0]['count'] if results else 0} items</p>
                        </div>
                    </div>
                </div>
        """
        
        if results:
            # Add recommendation section
            best_model = results[0]
            html_content += f"""
                <div class="recommendation">
                    <h2>[TARGET] Recommended Model</h2>
                    <h3>{best_model['model']}</h3>
                    <p>Found {best_model['count']} items with {best_model['confidence']} confidence threshold</p>
                    <p>Processing time: {best_model['time']:.3f} seconds</p>
                </div>
            """
        
        html_content += """
                <div class="results-container">
                    <h2 style="margin-bottom: 30px;">🏅 Model Rankings</h2>
                    <div class="ranking-grid">
        """
        
        # Add model cards
        for i, result in enumerate(results, 1):
            emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "🏅"
            rank_class = f"rank-{min(i, 4) if i <= 3 else 'other'}"
            
            html_content += f"""
                        <div class="model-card {rank_class}">
                            <div class="rank-badge">#{i}</div>
                            <div class="model-header">
                                <div class="model-emoji">{emoji}</div>
                                <div class="model-title">
                                    <h3>{result['model']}</h3>
                                    <p>{result['model_config']['type']} - {result['model_config']['size']} Size</p>
                                </div>
                            </div>
                            
                            <div class="metrics-grid">
                                <div class="metric">
                                    <span class="metric-value">{result['count']}</span>
                                    <div class="metric-label">Detections</div>
                                </div>
                                <div class="metric">
                                    <span class="metric-value">{result['confidence']}</span>
                                    <div class="metric-label">Best Confidence</div>
                                </div>
                                <div class="metric">
                                    <span class="metric-value">{result['time']:.3f}s</span>
                                    <div class="metric-label">Processing Time</div>
                                </div>
                                <div class="metric">
                                    <span class="metric-value">{len(set(d['class_name'] for d in result['detections']))}</span>
                                    <div class="metric-label">Unique Classes</div>
                                </div>
                            </div>
            """
            
            # Add detections section
            if result['detections']:
                html_content += f"""
                            <div class="detections-section">
                                <h4>🔍 Detected Items ({len(result['detections'])} total)</h4>
                                <div class="detections-grid">
                """
                
                for detection in result['detections']:
                    html_content += f"""
                                    <div class="detection-item">
                                        <span>{detection['class_name']}</span>
                                        <span style="font-weight: bold; color: #667eea;">{detection['confidence']:.2f}</span>
                                    </div>
                    """
                
                html_content += """
                                </div>
                            </div>
                """
            
            # Add confidence tests
            if 'confidence_tests' in result:
                html_content += f"""
                            <div class="confidence-tests">
                                <h4>[STATS] Confidence Threshold Tests</h4>
                                <div class="confidence-grid">
                """
                
                for test in result['confidence_tests']:
                    is_best = test['confidence'] == result['confidence']
                    best_class = 'best' if is_best else ''
                    
                    html_content += f"""
                                    <div class="confidence-card {best_class}">
                                        <div style="font-weight: bold; color: #667eea;">{test['confidence']}</div>
                                        <div style="font-size: 1.2em; margin: 5px 0;">{test['count']}</div>
                                        <div style="font-size: 0.8em; color: #666;">detections</div>
                                        <div style="font-size: 0.8em; color: #666;">{test['time']:.3f}s</div>
                                    </div>
                    """
                
                html_content += """
                                </div>
                            </div>
                """
            
            html_content += """
                        </div>
            """
        
        html_content += """
                    </div>
                </div>
                
                <div style="text-align: center; padding: 30px; background: #f8f9fa; color: #666;">
                    <p>Generated by YOLO Model Comparison Tool</p>
                    <p style="margin-top: 10px;">[RUN] Powered by Ultralytics YOLO</p>
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
    import sys
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) > 2 else "data/output/model_comparison"
        
        tester = EnhancedSingleImageTester()
        tester.test_all_models(image_path, output_dir)
    else:
        print("Usage: python test_single_image_enhanced.py <image_path> [output_dir]")

if __name__ == "__main__":
    main()