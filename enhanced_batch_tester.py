#!/usr/bin/env python3
"""Enhanced batch processing with beautiful HTML reports and multiple export formats."""

import sys
import time
import json
import webbrowser
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter
import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

class EnhancedBatchTester:
    """Enhanced batch tester with comprehensive HTML reports and multiple export formats."""
    
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
                'success_rate': len(model_results) / len(test_images) * 100,
                'std_detections': np.std([r['count'] for r in model_results]) if model_results else 0,
                'min_detections': min([r['count'] for r in model_results]) if model_results else 0,
                'max_detections': max([r['count'] for r in model_results]) if model_results else 0
            }
            
            print(f"  📊 Total: {all_results[model_name]['total_detections']} detections across {len(model_results)} images")
        
        # Generate comprehensive reports
        self.print_batch_summary(all_results, test_images)
        html_file = self.generate_batch_html_report(all_results, test_images, output_path)
        
        # Save results in multiple formats
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save JSON results (original functionality)
        json_file = output_path / f"batch_results_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        # Save CSV results
        csv_files = self.save_csv_results(all_results, test_images, output_path, timestamp)
        
        # Save Excel results
        excel_file = self.save_excel_results(all_results, test_images, output_path, timestamp)
        
        print(f"\n📄 Results saved in multiple formats:")
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
        
        return all_results
    
    def save_csv_results(self, all_results, test_images, output_path, timestamp):
        """Save results in CSV format."""
        csv_files = []
        
        # 1. Model Summary CSV
        summary_file = output_path / f"model_summary_{timestamp}.csv"
        summary_data = []
        
        for model_name, model_data in all_results.items():
            # Get most common detections
            all_detections = []
            for result in model_data['results']:
                all_detections.extend([d['class_name'] for d in result['detections']])
            
            common_items = Counter(all_detections).most_common(3)
            top_detections = ', '.join([f"{item}({count})" for item, count in common_items])
            
            summary_data.append({
                'Model': model_name,
                'Type': model_data['config']['type'],
                'Size': model_data['config']['size'],
                'Total_Detections': model_data['total_detections'],
                'Avg_Detections_Per_Image': round(model_data['avg_detections'], 2),
                'Std_Detections': round(model_data['std_detections'], 2),
                'Min_Detections': model_data['min_detections'],
                'Max_Detections': model_data['max_detections'],
                'Avg_Processing_Time_Seconds': round(model_data['avg_time'], 3),
                'Success_Rate_Percent': round(model_data['success_rate'], 1),
                'Images_Processed': len(model_data['results']),
                'Total_Images': len(test_images),
                'Top_Detected_Objects': top_detections
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('Total_Detections', ascending=False)
        summary_df.to_csv(summary_file, index=False)
        csv_files.append(summary_file)
        
        # 2. Detailed Results CSV (all models combined)
        detailed_file = output_path / f"detailed_results_{timestamp}.csv"
        detailed_data = []
        
        for model_name, model_data in all_results.items():
            for result in model_data['results']:
                for detection in result['detections']:
                    detailed_data.append({
                        'Model': model_name,
                        'Model_Type': model_data['config']['type'],
                        'Model_Size': model_data['config']['size'],
                        'Image_Name': result['image_name'],
                        'Processing_Time_Seconds': round(result['time'], 3),
                        'Total_Detections_In_Image': result['count'],
                        'Detected_Object': detection['class_name'],
                        'Confidence': round(detection['confidence'], 3)
                    })
        
        if detailed_data:
            detailed_df = pd.DataFrame(detailed_data)
            detailed_df.to_csv(detailed_file, index=False)
            csv_files.append(detailed_file)
        
        # 3. Per-Image Comparison CSV
        comparison_file = output_path / f"per_image_comparison_{timestamp}.csv"
        comparison_data = []
        
        # Get all unique images
        all_images = set()
        for model_data in all_results.values():
            for result in model_data['results']:
                all_images.add(result['image_name'])
        
        for image_name in sorted(all_images):
            row = {'Image_Name': image_name}
            for model_name, model_data in all_results.items():
                # Find result for this image
                image_result = next((r for r in model_data['results'] if r['image_name'] == image_name), None)
                if image_result:
                    row[f"{model_name}_Detections"] = image_result['count']
                    row[f"{model_name}_Time_Seconds"] = round(image_result['time'], 3)
                else:
                    row[f"{model_name}_Detections"] = 0
                    row[f"{model_name}_Time_Seconds"] = 0
            comparison_data.append(row)
        
        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            comparison_df.to_csv(comparison_file, index=False)
            csv_files.append(comparison_file)
        
        return csv_files
    
    def save_excel_results(self, all_results, test_images, output_path, timestamp):
        """Save results in Excel format with multiple sheets."""
        excel_file = output_path / f"batch_results_{timestamp}.xlsx"
        
        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
            # Sheet 1: Model Summary
            summary_data = []
            for model_name, model_data in all_results.items():
                all_detections = []
                for result in model_data['results']:
                    all_detections.extend([d['class_name'] for d in result['detections']])
                
                common_items = Counter(all_detections).most_common(3)
                top_detections = ', '.join([f"{item}({count})" for item, count in common_items])
                
                summary_data.append({
                    'Rank': 0,  # Will be filled after sorting
                    'Model': model_name,
                    'Type': model_data['config']['type'],
                    'Size': model_data['config']['size'],
                    'Total Detections': model_data['total_detections'],
                    'Avg Detections/Image': round(model_data['avg_detections'], 2),
                    'Std Detections': round(model_data['std_detections'], 2),
                    'Min Detections': model_data['min_detections'],
                    'Max Detections': model_data['max_detections'],
                    'Avg Time (s)': round(model_data['avg_time'], 3),
                    'Success Rate (%)': round(model_data['success_rate'], 1),
                    'Images Processed': len(model_data['results']),
                    'Total Images': len(test_images),
                    'Top Detected Objects': top_detections
                })
            
            summary_df = pd.DataFrame(summary_data)
            summary_df = summary_df.sort_values('Total Detections', ascending=False)
            summary_df['Rank'] = range(1, len(summary_df) + 1)
            summary_df.to_excel(writer, sheet_name='Model Summary', index=False)
            
            # Sheet 2: Per-Image Comparison
            all_images = set()
            for model_data in all_results.values():
                for result in model_data['results']:
                    all_images.add(result['image_name'])
            
            comparison_data = []
            for image_name in sorted(all_images):
                row = {'Image Name': image_name}
                for model_name, model_data in all_results.items():
                    image_result = next((r for r in model_data['results'] if r['image_name'] == image_name), None)
                    if image_result:
                        row[f"{model_name} (Detections)"] = image_result['count']
                        row[f"{model_name} (Time)"] = round(image_result['time'], 3)
                    else:
                        row[f"{model_name} (Detections)"] = 0
                        row[f"{model_name} (Time)"] = 0
                comparison_data.append(row)
            
            if comparison_data:
                comparison_df = pd.DataFrame(comparison_data)
                comparison_df.to_excel(writer, sheet_name='Per-Image Comparison', index=False)
            
            # Sheet 3-N: Individual model sheets
            for model_name, model_data in all_results.items():
                model_sheet_data = []
                for result in model_data['results']:
                    for detection in result['detections']:
                        model_sheet_data.append({
                            'Image Name': result['image_name'],
                            'Processing Time (s)': round(result['time'], 3),
                            'Total Detections': result['count'],
                            'Detected Object': detection['class_name'],
                            'Confidence': round(detection['confidence'], 3)
                        })
                
                if model_sheet_data:
                    model_df = pd.DataFrame(model_sheet_data)
                    # Truncate sheet name if too long for Excel
                    sheet_name = model_name.replace('.pt', '')[:31]
                    model_df.to_excel(writer, sheet_name=sheet_name, index=False)
            
            # Sheet: Object Detection Summary
            object_summary = {}
            for model_name, model_data in all_results.items():
                all_detections = []
                for result in model_data['results']:
                    all_detections.extend([d['class_name'] for d in result['detections']])
                object_counts = Counter(all_detections)
                for obj, count in object_counts.items():
                    if obj not in object_summary:
                        object_summary[obj] = {}
                    object_summary[obj][model_name] = count
            
            if object_summary:
                object_data = []
                for obj, model_counts in object_summary.items():
                    row = {'Object': obj}
                    total = 0
                    for model_name in all_results.keys():
                        count = model_counts.get(model_name, 0)
                        row[model_name] = count
                        total += count
                    row['Total Across All Models'] = total
                    object_data.append(row)
                
                object_df = pd.DataFrame(object_data)
                object_df = object_df.sort_values('Total Across All Models', ascending=False)
                object_df.to_excel(writer, sheet_name='Object Detection Summary', index=False)
        
        return excel_file
    
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
                
                .export-info {{
                    background: #e3f2fd;
                    padding: 20px;
                    margin: 30px;
                    border-radius: 15px;
                    border-left: 4px solid #2196f3;
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
                
                <div class="export-info">
                    <h3>📊 Data Export Formats Available:</h3>
                    <p><strong>Excel:</strong> Comprehensive workbook with multiple sheets for detailed analysis</p>
                    <p><strong>CSV:</strong> Model summary, detailed results, and per-image comparison files</p>
                    <p><strong>JSON:</strong> Raw data for programmatic access</p>
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
                    <p style="margin-top: 10px;">🚀 Powered by Ultralytics YOLO | 📊 Multiple Export Formats</p>
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
    
    parser = argparse.ArgumentParser(description="Enhanced batch model comparison with multiple export formats")
    parser.add_argument('--input-dir', default='data/input', help="Directory with test images")
    parser.add_argument('--output-dir', default='data/output/batch_model_comparison', help="Output directory")
    
    args = parser.parse_args()
    
    print("🚀 Enhanced Batch Tester with Multiple Export Formats")
    print("📊 Exports: JSON, CSV (3 files), Excel (multi-sheet), HTML")
    print("="*60)
    
    tester = EnhancedBatchTester()
    tester.test_batch(args.input_dir, args.output_dir)

if __name__ == "__main__":
    main()