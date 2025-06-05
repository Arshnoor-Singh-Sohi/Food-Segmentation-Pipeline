#!/usr/bin/env python3
"""Test and compare multiple YOLO models for food detection with multiple export formats."""

import sys
import time
import json
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Dict, List, Any
import shutil
from collections import Counter

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

class ModelComparison:
    """Test multiple YOLO models and compare their performance with comprehensive export options."""
    
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
            print(f"❌ Input directory not found: {input_dir}")
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
        
        print(f"\n🔧 Testing {model_name}...")
        
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
            all_confidence_results = []
            
            for conf in confidence_levels:
                print(f"  🎯 Testing confidence: {conf}")
                
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
                
                # Store result for this confidence level
                conf_result = {
                    'confidence_threshold': conf,
                    'detections': detections,
                    'detection_count': len(detections),
                    'inference_time_seconds': inference_time
                }
                all_confidence_results.append(conf_result)
                
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
                        'total_time_seconds': load_time + inference_time,
                        'all_confidence_results': all_confidence_results
                    }
                
                print(f"    📊 Found {len(detections)} detections")
            
            # Calculate additional metrics
            if best_result:
                # Average confidence of detections
                if best_result['detections']:
                    confidences = [d['confidence'] for d in best_result['detections']]
                    best_result['avg_confidence'] = np.mean(confidences)
                    best_result['min_confidence'] = np.min(confidences)
                    best_result['max_confidence'] = np.max(confidences)
                    best_result['std_confidence'] = np.std(confidences)
                else:
                    best_result['avg_confidence'] = 0
                    best_result['min_confidence'] = 0
                    best_result['max_confidence'] = 0
                    best_result['std_confidence'] = 0
                
                # Unique classes detected
                unique_classes = list(set([d['class_name'] for d in best_result['detections']]))
                best_result['unique_classes'] = unique_classes
                best_result['unique_class_count'] = len(unique_classes)
                
                # Save detailed results
                result_file = model_output_dir / f"{Path(test_image_path).stem}_results.json"
                with open(result_file, 'w') as f:
                    json.dump(best_result, f, indent=2)
                
                # Save visualization if possible
                try:
                    self.create_visualization(test_image_path, best_result, model_output_dir)
                except Exception as e:
                    print(f"    ⚠️ Visualization failed: {e}")
            
            print(f"✅ {model_name}: {best_detection_count} detections (best conf: {best_result['confidence_threshold'] if best_result else 'N/A'})")
            return best_result or {'model_config': model_config, 'error': 'No detections found'}
            
        except Exception as e:
            error_result = {
                'model_config': model_config,
                'error': str(e),
                'failed': True
            }
            print(f"❌ {model_name} failed: {e}")
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
    
    def save_csv_results(self, summary: Dict, timestamp: str):
        """Save results in CSV format."""
        csv_files = []
        
        # 1. Model Comparison Summary CSV
        summary_file = self.output_dir / f"model_comparison_{timestamp}.csv"
        summary_data = []
        
        for ranking in summary['model_rankings']:
            summary_data.append({
                'Rank': ranking['rank'],
                'Model_Name': ranking['model_name'],
                'Model_Type': ranking['model_type'],
                'Description': ranking['description'],
                'Detections_Found': ranking['detections_found'],
                'Unique_Classes': len(ranking['detected_classes']),
                'Best_Confidence_Threshold': ranking['best_confidence'],
                'Inference_Time_Seconds': round(ranking['inference_time'], 4),
                'Total_Time_Seconds': round(ranking['total_time'], 4),
                'Detected_Classes': ', '.join(ranking['detected_classes']),
                'Average_Confidence': ranking.get('avg_confidence', 0),
                'Min_Confidence': ranking.get('min_confidence', 0),
                'Max_Confidence': ranking.get('max_confidence', 0)
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(summary_file, index=False)
        csv_files.append(summary_file)
        
        # 2. Detailed Detections CSV
        detailed_file = self.output_dir / f"detailed_detections_{timestamp}.csv"
        detailed_data = []
        
        for result in summary['detailed_results']:
            if result.get('failed') or 'error' in result:
                continue
                
            model_name = result['model_config']['name']
            model_type = result['model_config']['type']
            
            for detection in result.get('detections', []):
                detailed_data.append({
                    'Model_Name': model_name,
                    'Model_Type': model_type,
                    'Test_Image': Path(result['test_image']).name,
                    'Confidence_Threshold': result['confidence_threshold'],
                    'Class_ID': detection['class_id'],
                    'Class_Name': detection['class_name'],
                    'Confidence': round(detection['confidence'], 4),
                    'Has_Mask': detection['has_mask'],
                    'Bbox_X1': round(detection['bbox'][0], 2),
                    'Bbox_Y1': round(detection['bbox'][1], 2),
                    'Bbox_X2': round(detection['bbox'][2], 2),
                    'Bbox_Y2': round(detection['bbox'][3], 2),
                    'Bbox_Width': round(detection['bbox'][2] - detection['bbox'][0], 2),
                    'Bbox_Height': round(detection['bbox'][3] - detection['bbox'][1], 2),
                    'Mask_Area': detection.get('mask_area', 0) if detection['has_mask'] else 0
                })
        
        if detailed_data:
            detailed_df = pd.DataFrame(detailed_data)
            detailed_df.to_csv(detailed_file, index=False)
            csv_files.append(detailed_file)
        
        # 3. Confidence Threshold Analysis CSV
        confidence_file = self.output_dir / f"confidence_analysis_{timestamp}.csv"
        confidence_data = []
        
        for result in summary['detailed_results']:
            if result.get('failed') or 'error' in result:
                continue
                
            model_name = result['model_config']['name']
            
            for conf_result in result.get('all_confidence_results', []):
                confidence_data.append({
                    'Model_Name': model_name,
                    'Model_Type': result['model_config']['type'],
                    'Confidence_Threshold': conf_result['confidence_threshold'],
                    'Detections_Count': conf_result['detection_count'],
                    'Inference_Time_Seconds': round(conf_result['inference_time_seconds'], 4),
                    'Unique_Classes': len(set([d['class_name'] for d in conf_result['detections']])),
                    'Avg_Detection_Confidence': round(np.mean([d['confidence'] for d in conf_result['detections']]), 4) if conf_result['detections'] else 0
                })
        
        if confidence_data:
            confidence_df = pd.DataFrame(confidence_data)
            confidence_df.to_csv(confidence_file, index=False)
            csv_files.append(confidence_file)
        
        return csv_files
    
    def save_excel_results(self, summary: Dict, timestamp: str):
        """Save results in Excel format with multiple sheets."""
        excel_file = self.output_dir / f"model_comparison_{timestamp}.xlsx"
        
        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
            # Sheet 1: Model Summary
            summary_data = []
            for ranking in summary['model_rankings']:
                summary_data.append({
                    'Rank': ranking['rank'],
                    'Model Name': ranking['model_name'],
                    'Type': ranking['model_type'],
                    'Description': ranking['description'],
                    'Detections': ranking['detections_found'],
                    'Unique Classes': len(ranking['detected_classes']),
                    'Best Confidence': ranking['best_confidence'],
                    'Inference Time (s)': round(ranking['inference_time'], 4),
                    'Total Time (s)': round(ranking['total_time'], 4),
                    'Avg Confidence': round(ranking.get('avg_confidence', 0), 4),
                    'Min Confidence': round(ranking.get('min_confidence', 0), 4),
                    'Max Confidence': round(ranking.get('max_confidence', 0), 4),
                    'Detected Classes': ', '.join(ranking['detected_classes'])
                })
            
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Model Summary', index=False)
            
            # Sheet 2: Detailed Detections
            detailed_data = []
            for result in summary['detailed_results']:
                if result.get('failed') or 'error' in result:
                    continue
                    
                for detection in result.get('detections', []):
                    detailed_data.append({
                        'Model': result['model_config']['name'],
                        'Type': result['model_config']['type'],
                        'Image': Path(result['test_image']).name,
                        'Confidence Threshold': result['confidence_threshold'],
                        'Class': detection['class_name'],
                        'Confidence': round(detection['confidence'], 4),
                        'Has Mask': detection['has_mask'],
                        'X1': round(detection['bbox'][0], 2),
                        'Y1': round(detection['bbox'][1], 2),
                        'X2': round(detection['bbox'][2], 2),
                        'Y2': round(detection['bbox'][3], 2),
                        'Width': round(detection['bbox'][2] - detection['bbox'][0], 2),
                        'Height': round(detection['bbox'][3] - detection['bbox'][1], 2)
                    })
            
            if detailed_data:
                detailed_df = pd.DataFrame(detailed_data)
                detailed_df.to_excel(writer, sheet_name='Detailed Detections', index=False)
            
            # Sheet 3: Confidence Analysis
            confidence_data = []
            for result in summary['detailed_results']:
                if result.get('failed') or 'error' in result:
                    continue
                    
                for conf_result in result.get('all_confidence_results', []):
                    confidence_data.append({
                        'Model': result['model_config']['name'],
                        'Type': result['model_config']['type'],
                        'Confidence Threshold': conf_result['confidence_threshold'],
                        'Detections': conf_result['detection_count'],
                        'Inference Time (s)': round(conf_result['inference_time_seconds'], 4),
                        'Unique Classes': len(set([d['class_name'] for d in conf_result['detections']])),
                        'Avg Confidence': round(np.mean([d['confidence'] for d in conf_result['detections']]), 4) if conf_result['detections'] else 0
                    })
            
            if confidence_data:
                confidence_df = pd.DataFrame(confidence_data)
                confidence_df.to_excel(writer, sheet_name='Confidence Analysis', index=False)
            
            # Sheet 4: Class Detection Summary
            class_summary = {}
            for result in summary['detailed_results']:
                if result.get('failed') or 'error' in result:
                    continue
                    
                model_name = result['model_config']['name']
                for detection in result.get('detections', []):
                    class_name = detection['class_name']
                    if class_name not in class_summary:
                        class_summary[class_name] = {}
                    if model_name not in class_summary[class_name]:
                        class_summary[class_name][model_name] = []
                    class_summary[class_name][model_name].append(detection['confidence'])
            
            class_data = []
            for class_name, model_data in class_summary.items():
                row = {'Class': class_name}
                total_detections = 0
                model_names = set()
                
                for model_name, confidences in model_data.items():
                    model_names.add(model_name)
                    count = len(confidences)
                    avg_conf = round(np.mean(confidences), 4)
                    total_detections += count
                    row[f'{model_name} Count'] = count
                    row[f'{model_name} Avg Conf'] = avg_conf
                
                row['Total Detections'] = total_detections
                row['Models Detecting'] = len(model_names)
                class_data.append(row)
            
            if class_data:
                class_df = pd.DataFrame(class_data)
                class_df = class_df.sort_values('Total Detections', ascending=False)
                class_df.to_excel(writer, sheet_name='Class Summary', index=False)
            
            # Sheet 5: Performance Metrics
            if summary['model_rankings']:
                perf_data = []
                for ranking in summary['model_rankings']:
                    perf_data.append({
                        'Model': ranking['model_name'],
                        'Type': ranking['model_type'],
                        'Rank': ranking['rank'],
                        'Performance Score': ranking['detections_found'] * 10 + (1 / max(ranking['inference_time'], 0.001)),  # Simple scoring
                        'Detection Count': ranking['detections_found'],
                        'Speed Score': round(1 / max(ranking['inference_time'], 0.001), 2),
                        'Class Diversity': len(ranking['detected_classes']),
                        'Efficiency': round(ranking['detections_found'] / max(ranking['total_time'], 0.001), 2)
                    })
                
                perf_df = pd.DataFrame(perf_data)
                perf_df.to_excel(writer, sheet_name='Performance Metrics', index=False)
        
        return excel_file
    
    def run_comprehensive_test(self, input_dir="data/input"):
        """Run comprehensive test on all models."""
        print("🚀 Starting Comprehensive Model Comparison")
        print("📊 Export formats: JSON, CSV (3 files), Excel (multi-sheet), HTML")
        print("=" * 60)
        
        # Setup test images
        if not self.setup_test_images(input_dir):
            return
        
        # Test first image with all models (for speed)
        test_image = self.test_images[0]
        print(f"🎯 Testing with image: {test_image.name}")
        
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
                print(f"❌ Failed testing {model_config['name']}: {e}")
                continue
        
        # Generate comparison report
        self.generate_comparison_report(all_results)
        
        return all_results
    
    def generate_comparison_report(self, results: List[Dict]):
        """Generate comprehensive comparison report in multiple formats."""
        print(f"\n📊 Generating Comparison Report in Multiple Formats...")
        
        # Filter successful results
        successful_results = [r for r in results if not r.get('failed', False) and 'error' not in r]
        
        if not successful_results:
            print("❌ No successful model results to compare")
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
                'inference_time': result.get('inference_time_seconds', 0),
                'total_time': result.get('total_time_seconds', 0),
                'detected_classes': list(set([d['class_name'] for d in result.get('detections', [])])),
                'avg_confidence': result.get('avg_confidence', 0),
                'min_confidence': result.get('min_confidence', 0),
                'max_confidence': result.get('max_confidence', 0)
            }
            summary['model_rankings'].append(ranking)
        
        # Generate timestamp for file naming
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save results in multiple formats
        
        # 1. JSON report (original functionality)
        json_file = self.output_dir / f"comparison_report_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # 2. CSV reports
        csv_files = self.save_csv_results(summary, timestamp)
        
        # 3. Excel report
        excel_file = self.save_excel_results(summary, timestamp)
        
        # 4. HTML report
        html_file = self.create_html_report(summary, timestamp)
        
        # Print summary to console
        self.print_comparison_summary(summary)
        
        # Display all generated files
        print(f"\n📄 Results exported in multiple formats:")
        print(f"  📊 JSON: {json_file}")
        print(f"  📈 Excel: {excel_file}")
        for csv_file in csv_files:
            print(f"  📋 CSV: {csv_file}")
        print(f"  🌐 HTML: {html_file}")
        
        print(f"\n📁 All files saved to: {self.output_dir}")
    
    def create_html_report(self, summary: Dict, timestamp: str):
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
                .export-info {{ background: #e3f2fd; padding: 20px; margin: 20px 0; border-radius: 10px; border-left: 4px solid #2196f3; }}
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
                
                <div class="export-info">
                    <h3>📊 Available Export Formats:</h3>
                    <p><strong>Excel:</strong> Multi-sheet workbook with comprehensive analysis</p>
                    <p><strong>CSV:</strong> Model comparison, detailed detections, and confidence analysis</p>
                    <p><strong>JSON:</strong> Raw structured data for programmatic access</p>
                    <p><strong>HTML:</strong> This visual report with interactive elements</p>
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
                    <div class="stat-card">
                        <strong>{max([r['detections_found'] for r in summary['model_rankings']], default=0)}</strong><br>
                        Max Detections
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
                <div class="metric">
                    <span><strong>Avg Confidence:</strong></span>
                    <span>{ranking['avg_confidence']:.3f}</span>
                </div>
                
                <div class="classes">
                    <strong>Detected Classes ({len(ranking['detected_classes'])}):</strong><br>
                    {', '.join(ranking['detected_classes']) if ranking['detected_classes'] else 'None'}
                </div>
            </div>
            """
        
        html_content += """
                </div>
                
                <div style="margin-top: 30px; padding: 20px; background: #f8f9fa; border-radius: 8px;">
                    <h3>📝 Recommendations</h3>
                    <ul>
        """
        
        if summary['model_rankings']:
            best_model = summary['model_rankings'][0]
            fastest_model = min(summary['model_rankings'], key=lambda x: x['inference_time'])
            most_classes = max(summary['model_rankings'], key=lambda x: len(x['detected_classes']))
            
            html_content += f"""
                        <li><strong>Best Overall:</strong> {best_model['model_name']} - {best_model['detections_found']} detections</li>
                        <li><strong>Fastest:</strong> {fastest_model['model_name']} - {fastest_model['inference_time']:.3f}s inference time</li>
                        <li><strong>Most Diverse:</strong> {most_classes['model_name']} - {len(most_classes['detected_classes'])} different classes</li>
                        <li><strong>For Production:</strong> Consider segmentation models for detailed analysis</li>
                        <li><strong>For Speed:</strong> Use nano models for real-time applications</li>
            """
        
        html_content += """
                    </ul>
                </div>
                
                <div style="text-align: center; padding: 30px; background: #f8f9fa; color: #666; margin-top: 20px;">
                    <p>Generated by Enhanced YOLO Model Comparison Tool</p>
                    <p style="margin-top: 10px;">🚀 Powered by Ultralytics YOLO | 📊 Multiple Export Formats Available</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        html_file = self.output_dir / f"comparison_report_{timestamp}.html"
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return html_file
    
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
            print(f"   Avg Conf: {ranking['avg_confidence']:.3f}")
        
        if summary['model_rankings']:
            best = summary['model_rankings'][0]
            print(f"\n🎯 RECOMMENDATION: Use {best['model_name']}")
            print(f"   Reason: {best['detections_found']} detections found")
            print(f"   Classes detected: {', '.join(best['detected_classes'])}")
            print(f"   Average confidence: {best['avg_confidence']:.3f}")
        
        print("="*80)

def main():
    """Run the comprehensive model comparison."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare multiple YOLO models with comprehensive export options")
    parser.add_argument('--input-dir', default='data/input', help="Directory with test images")
    parser.add_argument('--output-dir', default='data/output/model_comparison', help="Output directory")
    
    args = parser.parse_args()
    
    print("🚀 Enhanced Model Comparison Tool")
    print("📊 Export formats: JSON, CSV (3 files), Excel (multi-sheet), HTML")
    print("⚡ Testing models on first image for quick comparison")
    print("="*60)
    
    # Run comparison
    comparator = ModelComparison(args.output_dir)
    results = comparator.run_comprehensive_test(args.input_dir)
    
    if results:
        print(f"\n✅ Comparison complete! Check {args.output_dir} for all exported files")
        print("💡 Open the Excel file for detailed analysis or HTML for visual overview")
    else:
        print("❌ Comparison failed - no results generated")

if __name__ == "__main__":
    main()