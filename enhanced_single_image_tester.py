#!/usr/bin/env python3
"""Enhanced single image test with beautiful HTML reports and multiple export formats."""

import sys
import time
import json
import webbrowser
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from collections import Counter

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

class EnhancedSingleImageTester:
    """Enhanced single image tester with comprehensive export options."""
    
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
        """Test all models and generate comprehensive reports in multiple formats."""
        
        if not Path(image_path).exists():
            print(f"❌ Image not found: {image_path}")
            return
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"🎯 Testing image: {Path(image_path).name}")
        print("📊 Export formats: JSON, CSV (3 files), Excel (multi-sheet), HTML")
        print("="*60)
        
        results = []
        
        for model_config in self.models_to_test:
            model_name = model_config['name']
            print(f"\n🔧 Testing {model_name}...")
            
            try:
                result = self.test_single_model(model_config, image_path)
                if result:
                    results.append(result)
                    
            except Exception as e:
                print(f"❌ {model_name} failed: {e}")
                continue
        
        # Sort results by detection count
        results.sort(key=lambda x: x['count'], reverse=True)
        
        # Print console summary (keep the nice console output)
        self.print_console_summary(results)
        
        # Generate reports in multiple formats
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 1. HTML report
        html_file = self.generate_html_report(results, image_path, output_path, timestamp)
        
        # 2. JSON results (original functionality)
        json_file = output_path / f"single_image_test_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump({
                'image_path': image_path,
                'image_name': Path(image_path).name,
                'timestamp': datetime.now().isoformat(),
                'results': results,
                'summary': self.create_summary_data(results)
            }, f, indent=2)
        
        # 3. CSV reports
        csv_files = self.save_csv_results(results, image_path, output_path, timestamp)
        
        # 4. Excel report
        excel_file = self.save_excel_results(results, image_path, output_path, timestamp)
        
        # Display all generated files
        print(f"\n📄 Results exported in multiple formats:")
        print(f"  📊 JSON: {json_file}")
        print(f"  📈 Excel: {excel_file}")
        for csv_file in csv_files:
            print(f"  📋 CSV: {csv_file}")
        print(f"  🌐 HTML: {html_file}")
        
        # Auto-open HTML report
        try:
            webbrowser.open(f'file://{html_file.absolute()}')
            print("\n🚀 HTML report opened in browser!")
        except:
            print("\n💡 Open the HTML file manually in your browser")
        
        return results
    
    def test_single_model(self, model_config, image_path):
        """Test a single model with enhanced metrics collection."""
        model_name = model_config['name']
        
        try:
            from ultralytics import YOLO
            
            # Load model
            model = YOLO(model_name)
            
            # Test different confidence levels
            best_result = None
            best_count = 0
            confidence_tests = []
            all_detections = []
            
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
                                'class_id': int(result.boxes.cls[i]),
                                'class_name': model.names[int(result.boxes.cls[i])],
                                'confidence': float(result.boxes.conf[i]),
                                'bbox': result.boxes.xyxy[i].tolist(),
                                'bbox_width': float(result.boxes.xyxy[i][2] - result.boxes.xyxy[i][0]),
                                'bbox_height': float(result.boxes.xyxy[i][3] - result.boxes.xyxy[i][1]),
                                'bbox_area': float((result.boxes.xyxy[i][2] - result.boxes.xyxy[i][0]) * 
                                                 (result.boxes.xyxy[i][3] - result.boxes.xyxy[i][1]))
                            }
                            detections.append(detection)
                            all_detections.append(detection)
                
                confidence_test = {
                    'confidence': conf,
                    'detections': detections,
                    'count': detection_count,
                    'time': inference_time,
                    'unique_classes': len(set(d['class_name'] for d in detections)),
                    'avg_confidence': np.mean([d['confidence'] for d in detections]) if detections else 0,
                    'min_confidence': np.min([d['confidence'] for d in detections]) if detections else 0,
                    'max_confidence': np.max([d['confidence'] for d in detections]) if detections else 0
                }
                confidence_tests.append(confidence_test)
                
                if detection_count > best_count:
                    best_count = detection_count
                    best_result = confidence_test.copy()
                
                print(f"  Conf {conf}: {detection_count} detections")
            
            if best_result:
                # Calculate additional metrics
                unique_classes = list(set(d['class_name'] for d in all_detections))
                class_counts = Counter(d['class_name'] for d in all_detections)
                
                final_result = {
                    'model_config': model_config,
                    'model': model_name,
                    'best_result': best_result,
                    'confidence_tests': confidence_tests,
                    'count': best_count,
                    'time': best_result['time'],
                    'confidence': best_result['confidence'],
                    'detections': best_result['detections'],
                    'all_detections': all_detections,
                    'unique_classes': unique_classes,
                    'unique_class_count': len(unique_classes),
                    'class_counts': dict(class_counts),
                    'avg_detection_confidence': best_result['avg_confidence'],
                    'total_bbox_area': sum(d['bbox_area'] for d in best_result['detections']),
                    'avg_bbox_area': np.mean([d['bbox_area'] for d in best_result['detections']]) if best_result['detections'] else 0
                }
                
                print(f"✅ Best: {best_count} detections at conf {best_result['confidence']}")
                
                if best_result['detections']:
                    detected_items = [d['class_name'] for d in best_result['detections']]
                    print(f"   Found: {', '.join(detected_items)}")
                
                return final_result
            else:
                print("❌ No detections found")
                return None
                
        except Exception as e:
            print(f"❌ {model_name} failed: {e}")
            return None
    
    def create_summary_data(self, results):
        """Create summary statistics for the test."""
        if not results:
            return {}
        
        return {
            'total_models_tested': len(results),
            'best_model': results[0]['model'],
            'max_detections': results[0]['count'],
            'fastest_model': min(results, key=lambda x: x['time'])['model'],
            'fastest_time': min(results, key=lambda x: x['time'])['time'],
            'avg_detections': np.mean([r['count'] for r in results]),
            'avg_processing_time': np.mean([r['time'] for r in results]),
            'total_unique_classes': len(set().union(*[r['unique_classes'] for r in results])),
            'models_by_type': Counter(r['model_config']['type'] for r in results)
        }
    
    def save_csv_results(self, results, image_path, output_path, timestamp):
        """Save results in CSV format."""
        csv_files = []
        image_name = Path(image_path).name
        
        # 1. Model Comparison Summary CSV
        summary_file = output_path / f"model_summary_{timestamp}.csv"
        summary_data = []
        
        for i, result in enumerate(results, 1):
            summary_data.append({
                'Rank': i,
                'Model_Name': result['model'],
                'Model_Type': result['model_config']['type'],
                'Model_Size': result['model_config']['size'],
                'Total_Detections': result['count'],
                'Unique_Classes': result['unique_class_count'],
                'Best_Confidence_Threshold': result['confidence'],
                'Processing_Time_Seconds': round(result['time'], 4),
                'Avg_Detection_Confidence': round(result['avg_detection_confidence'], 4),
                'Total_Bbox_Area': round(result['total_bbox_area'], 2),
                'Avg_Bbox_Area': round(result['avg_bbox_area'], 2),
                'Detected_Classes': ', '.join(result['unique_classes']),
                'Most_Common_Class': max(result['class_counts'].items(), key=lambda x: x[1])[0] if result['class_counts'] else 'None',
                'Most_Common_Class_Count': max(result['class_counts'].values()) if result['class_counts'] else 0
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(summary_file, index=False)
        csv_files.append(summary_file)
        
        # 2. Detailed Detections CSV
        detailed_file = output_path / f"detailed_detections_{timestamp}.csv"
        detailed_data = []
        
        for result in results:
            for detection in result['detections']:
                detailed_data.append({
                    'Model_Name': result['model'],
                    'Model_Type': result['model_config']['type'],
                    'Model_Size': result['model_config']['size'],
                    'Image_Name': image_name,
                    'Best_Confidence_Threshold': result['confidence'],
                    'Class_ID': detection['class_id'],
                    'Class_Name': detection['class_name'],
                    'Detection_Confidence': round(detection['confidence'], 4),
                    'Bbox_X1': round(detection['bbox'][0], 2),
                    'Bbox_Y1': round(detection['bbox'][1], 2),
                    'Bbox_X2': round(detection['bbox'][2], 2),
                    'Bbox_Y2': round(detection['bbox'][3], 2),
                    'Bbox_Width': round(detection['bbox_width'], 2),
                    'Bbox_Height': round(detection['bbox_height'], 2),
                    'Bbox_Area': round(detection['bbox_area'], 2)
                })
        
        if detailed_data:
            detailed_df = pd.DataFrame(detailed_data)
            detailed_df.to_csv(detailed_file, index=False)
            csv_files.append(detailed_file)
        
        # 3. Confidence Threshold Analysis CSV
        confidence_file = output_path / f"confidence_analysis_{timestamp}.csv"
        confidence_data = []
        
        for result in results:
            for conf_test in result['confidence_tests']:
                confidence_data.append({
                    'Model_Name': result['model'],
                    'Model_Type': result['model_config']['type'],
                    'Model_Size': result['model_config']['size'],
                    'Confidence_Threshold': conf_test['confidence'],
                    'Detection_Count': conf_test['count'],
                    'Unique_Classes': conf_test['unique_classes'],
                    'Processing_Time_Seconds': round(conf_test['time'], 4),
                    'Avg_Detection_Confidence': round(conf_test['avg_confidence'], 4),
                    'Min_Detection_Confidence': round(conf_test['min_confidence'], 4),
                    'Max_Detection_Confidence': round(conf_test['max_confidence'], 4),
                    'Is_Best_Threshold': conf_test['confidence'] == result['confidence']
                })
        
        if confidence_data:
            confidence_df = pd.DataFrame(confidence_data)
            confidence_df.to_csv(confidence_file, index=False)
            csv_files.append(confidence_file)
        
        return csv_files
    
    def save_excel_results(self, results, image_path, output_path, timestamp):
        """Save results in Excel format with multiple sheets."""
        excel_file = output_path / f"single_image_test_{timestamp}.xlsx"
        image_name = Path(image_path).name
        
        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
            # Sheet 1: Model Summary
            summary_data = []
            for i, result in enumerate(results, 1):
                summary_data.append({
                    'Rank': i,
                    'Model': result['model'],
                    'Type': result['model_config']['type'],
                    'Size': result['model_config']['size'],
                    'Detections': result['count'],
                    'Unique Classes': result['unique_class_count'],
                    'Best Confidence': result['confidence'],
                    'Processing Time (s)': round(result['time'], 4),
                    'Avg Confidence': round(result['avg_detection_confidence'], 4),
                    'Total Bbox Area': round(result['total_bbox_area'], 2),
                    'Avg Bbox Area': round(result['avg_bbox_area'], 2),
                    'Detected Classes': ', '.join(result['unique_classes']),
                    'Performance Score': round(result['count'] * result['avg_detection_confidence'] / max(result['time'], 0.001), 2)
                })
            
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Model Summary', index=False)
            
            # Sheet 2: Test Information
            test_info = pd.DataFrame([{
                'Test Image': image_name,
                'Test Date': datetime.now().strftime('%Y-%m-%d'),
                'Test Time': datetime.now().strftime('%H:%M:%S'),
                'Total Models Tested': len(results),
                'Best Model': results[0]['model'] if results else 'None',
                'Max Detections Found': results[0]['count'] if results else 0,
                'Fastest Model': min(results, key=lambda x: x['time'])['model'] if results else 'None',
                'Fastest Time (s)': round(min(results, key=lambda x: x['time'])['time'], 4) if results else 0,
                'Average Detections': round(np.mean([r['count'] for r in results]), 2) if results else 0,
                'Average Processing Time (s)': round(np.mean([r['time'] for r in results]), 4) if results else 0
            }])
            test_info.to_excel(writer, sheet_name='Test Info', index=False)
            
            # Sheet 3: Detailed Detections
            detailed_data = []
            for result in results:
                for detection in result['detections']:
                    detailed_data.append({
                        'Model': result['model'],
                        'Type': result['model_config']['type'],
                        'Size': result['model_config']['size'],
                        'Class': detection['class_name'],
                        'Confidence': round(detection['confidence'], 4),
                        'X1': round(detection['bbox'][0], 2),
                        'Y1': round(detection['bbox'][1], 2),
                        'X2': round(detection['bbox'][2], 2),
                        'Y2': round(detection['bbox'][3], 2),
                        'Width': round(detection['bbox_width'], 2),
                        'Height': round(detection['bbox_height'], 2),
                        'Area': round(detection['bbox_area'], 2)
                    })
            
            if detailed_data:
                detailed_df = pd.DataFrame(detailed_data)
                detailed_df.to_excel(writer, sheet_name='Detailed Detections', index=False)
            
            # Sheet 4: Confidence Analysis
            confidence_data = []
            for result in results:
                for conf_test in result['confidence_tests']:
                    confidence_data.append({
                        'Model': result['model'],
                        'Type': result['model_config']['type'],
                        'Confidence Threshold': conf_test['confidence'],
                        'Detections': conf_test['count'],
                        'Unique Classes': conf_test['unique_classes'],
                        'Processing Time (s)': round(conf_test['time'], 4),
                        'Avg Confidence': round(conf_test['avg_confidence'], 4),
                        'Min Confidence': round(conf_test['min_confidence'], 4),
                        'Max Confidence': round(conf_test['max_confidence'], 4),
                        'Is Best': conf_test['confidence'] == result['confidence']
                    })
            
            if confidence_data:
                confidence_df = pd.DataFrame(confidence_data)
                confidence_df.to_excel(writer, sheet_name='Confidence Analysis', index=False)
            
            # Sheet 5: Class Detection Summary
            class_summary = {}
            for result in results:
                model_name = result['model']
                for class_name, count in result['class_counts'].items():
                    if class_name not in class_summary:
                        class_summary[class_name] = {}
                    class_summary[class_name][model_name] = count
            
            if class_summary:
                class_data = []
                for class_name, model_counts in class_summary.items():
                    row = {'Class': class_name}
                    total_detections = 0
                    models_detecting = 0
                    
                    for result in results:
                        model_name = result['model']
                        count = model_counts.get(model_name, 0)
                        row[f'{model_name}'] = count
                        if count > 0:
                            models_detecting += 1
                        total_detections += count
                    
                    row['Total Detections'] = total_detections
                    row['Models Detecting'] = models_detecting
                    row['Detection Rate'] = f"{(models_detecting/len(results)*100):.1f}%" if results else "0%"
                    class_data.append(row)
                
                class_df = pd.DataFrame(class_data)
                class_df = class_df.sort_values('Total Detections', ascending=False)
                class_df.to_excel(writer, sheet_name='Class Summary', index=False)
            
            # Sheet 6: Performance Comparison
            if results:
                perf_data = []
                for result in results:
                    efficiency = result['count'] / max(result['time'], 0.001)
                    accuracy_score = result['avg_detection_confidence']
                    diversity_score = result['unique_class_count']
                    overall_score = (efficiency * 0.4) + (accuracy_score * 0.4) + (diversity_score * 0.2)
                    
                    perf_data.append({
                        'Model': result['model'],
                        'Type': result['model_config']['type'],
                        'Overall Score': round(overall_score, 3),
                        'Efficiency (det/sec)': round(efficiency, 2),
                        'Accuracy Score': round(accuracy_score, 3),
                        'Diversity Score': diversity_score,
                        'Speed Rank': 0,  # Will be filled after sorting
                        'Accuracy Rank': 0,
                        'Detection Rank': 0
                    })
                
                perf_df = pd.DataFrame(perf_data)
                
                # Add rankings
                perf_df['Speed Rank'] = perf_df['Efficiency (det/sec)'].rank(ascending=False, method='min').astype(int)
                perf_df['Accuracy Rank'] = perf_df['Accuracy Score'].rank(ascending=False, method='min').astype(int)
                perf_df['Detection Rank'] = perf_df['Diversity Score'].rank(ascending=False, method='min').astype(int)
                
                perf_df = perf_df.sort_values('Overall Score', ascending=False)
                perf_df.to_excel(writer, sheet_name='Performance Analysis', index=False)
        
        return excel_file
    
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
            print(f"   Avg confidence: {result['avg_detection_confidence']:.3f}")
            if result['detections']:
                items = [d['class_name'] for d in result['detections']]
                print(f"   Items: {', '.join(items)}")
            print()
        
        # Recommendation
        if results:
            best = results[0]
            print(f"🎯 RECOMMENDATION: Use {best['model']}")
            print(f"   Found {best['count']} items with confidence {best['confidence']}")
            print(f"   Average detection confidence: {best['avg_detection_confidence']:.3f}")
            print("="*60)
    
    def generate_html_report(self, results, image_path, output_dir, timestamp):
        """Generate beautiful HTML report with export format information."""
        
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
                
                .export-info {{
                    background: #e3f2fd;
                    padding: 20px;
                    margin: 0;
                    border-bottom: 1px solid #ddd;
                }}
                
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
                
                <div class="export-info">
                    <h3>📊 Available Export Formats:</h3>
                    <p><strong>Excel:</strong> Multi-sheet workbook with comprehensive analysis and performance metrics</p>
                    <p><strong>CSV:</strong> Model summary, detailed detections, and confidence threshold analysis (3 files)</p>
                    <p><strong>JSON:</strong> Complete structured data for programmatic access</p>
                    <p><strong>HTML:</strong> This visual report with interactive elements</p>
                </div>
                
                <div class="test-info">
                    <h2 style="margin-bottom: 20px;">📊 Test Summary</h2>
                    <div class="test-info-grid">
                        <div class="info-card">
                            <h4>🎯 Test Image</h4>
                            <p>{image_name}</p>
                        </div>
                        <div class="info-card">
                            <h4>🔧 Models Tested</h4>
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
                        <div class="info-card">
                            <h4>⚡ Fastest Model</h4>
                            <p>{min(results, key=lambda x: x['time'])['model'] if results else 'None'}</p>
                        </div>
                        <div class="info-card">
                            <h4>🎯 Avg Accuracy</h4>
                            <p>{np.mean([r['avg_detection_confidence'] for r in results]):.3f} confidence</p>
                        </div>
                    </div>
                </div>
        """
        
        if results:
            # Add recommendation section
            best_model = results[0]
            fastest_model = min(results, key=lambda x: x['time'])
            html_content += f"""
                <div class="recommendation">
                    <h2>🎯 Recommended Model</h2>
                    <h3>{best_model['model']}</h3>
                    <p>Found {best_model['count']} items with {best_model['confidence']} confidence threshold</p>
                    <p>Average detection confidence: {best_model['avg_detection_confidence']:.3f}</p>
                    <p>Processing time: {best_model['time']:.3f} seconds</p>
                    <p style="margin-top: 10px;"><strong>For speed:</strong> Consider {fastest_model['model']} ({fastest_model['time']:.3f}s)</p>
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
                                <div class="metric">
                                    <span class="metric-value">{result['avg_detection_confidence']:.3f}</span>
                                    <div class="metric-label">Avg Confidence</div>
                                </div>
                                <div class="metric">
                                    <span class="metric-value">{result['avg_bbox_area']:.0f}</span>
                                    <div class="metric-label">Avg Bbox Area</div>
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
                                <h4>📊 Confidence Threshold Tests</h4>
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
                                        <div style="font-size: 0.8em; color: #666;">avg: {test['avg_confidence']:.2f}</div>
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
                    <p>Generated by Enhanced YOLO Model Comparison Tool</p>
                    <p style="margin-top: 10px;">🚀 Powered by Ultralytics YOLO | 📊 Multiple Export Formats Available</p>
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
        
        print("🚀 Enhanced Single Image Tester")
        print("📊 Export formats: JSON, CSV (3 files), Excel (multi-sheet), HTML")
        print("="*60)
        
        tester = EnhancedSingleImageTester()
        tester.test_all_models(image_path, output_dir)
    else:
        print("Usage: python test_single_image_enhanced.py <image_path> [output_dir]")
        print("\nExport formats available:")
        print("  📊 JSON - Complete structured data")
        print("  📈 Excel - Multi-sheet workbook with analysis")
        print("  📋 CSV - 3 files: summary, detections, confidence analysis")
        print("  🌐 HTML - Interactive visual report")

if __name__ == "__main__":
    main()