#!/usr/bin/env python3
"""Simple batch processor without seaborn dependency."""

import sys
import json
import time
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from collections import Counter

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.fast_yolo_segmentation import FastFoodSegmentation

def main():
    input_dir = "data/input"
    output_dir = Path("data/output/yolo_results")
    
    # Get image files
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    image_files = []
    
    for ext in extensions:
        image_files.extend(Path(input_dir).glob(f'*{ext}'))
        image_files.extend(Path(input_dir).glob(f'*{ext.upper()}'))
    
    if not image_files:
        print(f"No images found in {input_dir}")
        return
    
    print(f"Found {len(image_files)} images to process")
    
    # Initialize processor
    processor = FastFoodSegmentation(model_size='n')
    
    # Process images
    results = []
    failed = []
    
    start_time = time.time()
    
    with tqdm(total=len(image_files), desc="Processing images", unit="img") as pbar:
        for image_path in image_files:
            try:
                result = processor.process_single_image(str(image_path))
                
                if 'error' in result:
                    failed.append({'image': str(image_path), 'error': result['error']})
                else:
                    results.append(result)
                    
                    # Save individual result
                    output_dir.mkdir(parents=True, exist_ok=True)
                    json_file = output_dir / f"{image_path.stem}_results.json"
                    with open(json_file, 'w') as f:
                        json.dump(result, f, indent=2)
                
                pbar.set_postfix({
                    'Current': image_path.name[:15],
                    'Success': len(results),
                    'Failed': len(failed)
                })
                pbar.update(1)
                
            except Exception as e:
                failed.append({'image': str(image_path), 'error': str(e)})
                pbar.update(1)
    
    total_time = time.time() - start_time
    
    # Generate summary
    generate_summary_report(results, failed, total_time, output_dir)
    
    # Print final results
    print_final_summary(results, failed, total_time)

def generate_summary_report(results, failed, total_time, output_dir):
    """Generate summary report."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Calculate stats
    total_food_items = sum(r['analysis_summary']['food_items_count'] for r in results)
    total_calories = sum(r['nutrition_totals']['calories'] for r in results)
    
    # Most common foods
    all_foods = []
    for result in results:
        all_foods.extend([item['name'] for item in result['food_items'] if item['is_food']])
    
    food_counts = Counter(all_foods)
    
    # Create summary
    summary = {
        'batch_info': {
            'timestamp': timestamp,
            'total_images': len(results) + len(failed),
            'successful': len(results),
            'failed': len(failed),
            'success_rate': round(len(results) / (len(results) + len(failed)) * 100, 1) if (len(results) + len(failed)) > 0 else 0,
            'total_processing_time_minutes': round(total_time / 60, 2),
            'average_time_per_image': round(total_time / (len(results) + len(failed)), 2) if (len(results) + len(failed)) > 0 else 0
        },
        'analysis_summary': {
            'total_food_items_detected': total_food_items,
            'total_calories': round(total_calories, 1),
            'average_items_per_image': round(total_food_items / len(results), 1) if results else 0,
            'average_calories_per_image': round(total_calories / len(results), 1) if results else 0,
            'most_common_foods': dict(food_counts.most_common(10))
        },
        'detailed_results': results,
        'failed_images': failed
    }
    
    # Save JSON report
    report_dir = output_dir / 'batch_reports'
    report_dir.mkdir(exist_ok=True)
    
    report_file = report_dir / f'batch_summary_{timestamp}.json'
    with open(report_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Create simple HTML report
    create_simple_html_report(summary, report_dir, timestamp)
    
    print(f"Reports saved to: {report_dir}")

def create_simple_html_report(summary, report_dir, timestamp):
    """Create simple HTML report."""
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Batch Food Analysis Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
            .container {{ max-width: 1000px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; }}
            .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; text-align: center; }}
            .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }}
            .stat-card {{ background: #e3f2fd; padding: 20px; border-radius: 8px; text-align: center; border-left: 4px solid #2196f3; }}
            .stat-number {{ font-size: 2.5em; font-weight: bold; color: #1976d2; }}
            .stat-label {{ color: #666; margin-top: 10px; font-size: 1.1em; }}
            .section {{ margin: 30px 0; padding: 20px; background: #f8f9fa; border-radius: 8px; }}
            .food-list {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; }}
            .food-item {{ background: white; padding: 10px; border-radius: 5px; text-align: center; border-left: 3px solid #4caf50; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🍽️ Batch Food Analysis Report</h1>
                <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="stats">
                <div class="stat-card">
                    <div class="stat-number">{summary['batch_info']['total_images']}</div>
                    <div class="stat-label">Total Images</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{summary['batch_info']['successful']}</div>
                    <div class="stat-label">Successfully Processed</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{summary['batch_info']['success_rate']:.1f}%</div>
                    <div class="stat-label">Success Rate</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{summary['analysis_summary']['total_food_items_detected']}</div>
                    <div class="stat-label">Food Items Found</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{summary['analysis_summary']['total_calories']:.0f}</div>
                    <div class="stat-label">Total Calories</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{summary['batch_info']['total_processing_time_minutes']:.1f}</div>
                    <div class="stat-label">Processing Time (min)</div>
                </div>
            </div>
            
            <div class="section">
                <h2>📊 Most Common Foods</h2>
                <div class="food-list">
    """
    
    for food, count in summary['analysis_summary']['most_common_foods'].items():
        html_content += f"""
        <div class="food-item">
            <strong>{food.title()}</strong><br>
            <span style="color: #666;">{count} detected</span>
        </div>
        """
    
    html_content += f"""
                </div>
            </div>
            
            <div class="section">
                <h2>📈 Processing Statistics</h2>
                <p><strong>Average processing time per image:</strong> {summary['batch_info']['average_time_per_image']:.1f} seconds</p>
                <p><strong>Average food items per image:</strong> {summary['analysis_summary']['average_items_per_image']:.1f}</p>
                <p><strong>Average calories per image:</strong> {summary['analysis_summary']['average_calories_per_image']:.1f}</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    html_file = report_dir / f'dashboard_{timestamp}.html'
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"HTML dashboard: {html_file}")

def print_final_summary(results, failed, total_time):
    """Print final summary."""
    total = len(results) + len(failed)
    success_rate = (len(results) / total * 100) if total > 0 else 0
    
    print("\n" + "="*70)
    print("BATCH PROCESSING COMPLETE!")
    print("="*70)
    print(f"Total Images: {total}")
    print(f"Successful: {len(results)}")
    print(f"Failed: {len(failed)}")
    print(f"Success Rate: {success_rate:.1f}%")
    print(f"Total Time: {total_time/60:.1f} minutes")
    print(f"Average Time per Image: {total_time/total:.1f} seconds")
    
    if results:
        total_food_items = sum(r['analysis_summary']['food_items_count'] for r in results)
        total_calories = sum(r['nutrition_totals']['calories'] for r in results)
        
        print(f"\nAnalysis Results:")
        print(f"   Total Food Items: {total_food_items}")
        print(f"   Total Calories: {total_calories:.1f}")
        print(f"   Average Items per Image: {total_food_items/len(results):.1f}")
        print(f"   Average Calories per Image: {total_calories/len(results):.1f}")
    
    print("="*70)

if __name__ == "__main__":
    main()