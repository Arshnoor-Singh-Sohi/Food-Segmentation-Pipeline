#!/usr/bin/env python3
"""Batch process all images with YOLO - Generate comprehensive reports!"""

import sys
import argparse
import json
import time
from pathlib import Path
from datetime import datetime
import pandas as pd
from tqdm import tqdm
import concurrent.futures
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.fast_yolo_segmentation import FastFoodSegmentation

class BatchFoodProcessor:
    """Batch process food images with comprehensive reporting."""
    
    def __init__(self, model_size='n', output_dir='data/output/yolo_results'):
        self.processor = FastFoodSegmentation(model_size=model_size)
        self.output_dir = Path(output_dir)
        self.results = []
        self.failed_images = []
        
        # Create directory structure
        self.setup_output_directories()
    
    def setup_output_directories(self):
        """Create organized output directory structure."""
        directories = [
            'individual_results',
            'visualizations', 
            'batch_reports',
            'csv_exports',
            'failed_images'
        ]
        
        for dir_name in directories:
            (self.output_dir / dir_name).mkdir(parents=True, exist_ok=True)
    
    def get_image_files(self, input_dir):
        """Get all image files from input directory."""
        extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        image_files = []
        
        for ext in extensions:
            image_files.extend(Path(input_dir).glob(f'*{ext}'))
            image_files.extend(Path(input_dir).glob(f'*{ext.upper()}'))
        
        return sorted(image_files)
    
    def process_batch(self, input_dir, parallel=False, max_workers=2):
        """Process all images in batch with progress tracking."""
        image_files = self.get_image_files(input_dir)
        
        if not image_files:
            print(f"[FAIL] No images found in {input_dir}")
            return
        
        print(f"🔍 Found {len(image_files)} images to process")
        start_time = time.time()
        
        if parallel:
            self._process_parallel(image_files, max_workers)
        else:
            self._process_sequential(image_files)
        
        total_time = time.time() - start_time
        
        # Generate comprehensive reports
        self._generate_reports(total_time, len(image_files))
        
        # Print summary
        self._print_final_summary(total_time, len(image_files))
    
    def _process_sequential(self, image_files):
        """Process images one by one with progress bar."""
        with tqdm(total=len(image_files), desc="Processing images", unit="img") as pbar:
            for image_path in image_files:
                result = self._process_single_with_error_handling(image_path)
                self._save_individual_result(result, image_path)
                
                pbar.set_postfix({
                    'Current': image_path.name[:20],
                    'Success': len(self.results),
                    'Failed': len(self.failed_images)
                })
                pbar.update(1)
    
    def _process_parallel(self, image_files, max_workers):
        """Process images in parallel (careful with memory)."""
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_image = {
                executor.submit(self._process_single_with_error_handling, img): img 
                for img in image_files
            }
            
            with tqdm(total=len(image_files), desc="Processing images", unit="img") as pbar:
                for future in concurrent.futures.as_completed(future_to_image):
                    image_path = future_to_image[future]
                    result = future.result()
                    self._save_individual_result(result, image_path)
                    
                    pbar.set_postfix({
                        'Success': len(self.results),
                        'Failed': len(self.failed_images)
                    })
                    pbar.update(1)
    
    def _process_single_with_error_handling(self, image_path):
        """Process single image with error handling."""
        try:
            result = self.processor.process_single_image(str(image_path))
            
            if 'error' in result:
                self.failed_images.append({
                    'image': str(image_path),
                    'error': result['error']
                })
            else:
                self.results.append(result)
            
            return result
            
        except Exception as e:
            error_result = {
                'image_info': {'path': str(image_path), 'filename': image_path.name},
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            self.failed_images.append({
                'image': str(image_path),
                'error': str(e)
            })
            return error_result
    
    def _save_individual_result(self, result, image_path):
        """Save individual result to JSON file."""
        if 'error' not in result:
            output_file = self.output_dir / 'individual_results' / f"{image_path.stem}_results.json"
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2)
    
    def _generate_reports(self, total_time, total_images):
        """Generate comprehensive batch reports."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 1. JSON Summary Report
        self._generate_json_report(timestamp, total_time, total_images)
        
        # 2. CSV Exports
        self._generate_csv_reports(timestamp)
        
        # 3. HTML Dashboard
        self._generate_html_dashboard(timestamp)
        
        # 4. Visual Analytics
        self._generate_visual_analytics(timestamp)
    
    def _generate_json_report(self, timestamp, total_time, total_images):
        """Generate comprehensive JSON report."""
        successful_results = [r for r in self.results if 'error' not in r]
        
        # Calculate statistics
        total_food_items = sum(r['analysis_summary']['food_items_count'] for r in successful_results)
        total_calories = sum(r['nutrition_totals']['calories'] for r in successful_results)
        
        # Most common foods
        all_foods = []
        for result in successful_results:
            all_foods.extend([item['name'] for item in result['food_items'] if item['is_food']])
        
        food_counts = Counter(all_foods)
        
        summary_report = {
            'batch_info': {
                'timestamp': timestamp,
                'total_images': total_images,
                'successful': len(successful_results),
                'failed': len(self.failed_images),
                'success_rate': round(len(successful_results) / total_images * 100, 1),
                'total_processing_time_minutes': round(total_time / 60, 2),
                'average_time_per_image': round(total_time / total_images, 2)
            },
            'analysis_summary': {
                'total_food_items_detected': total_food_items,
                'total_calories': round(total_calories, 1),
                'average_items_per_image': round(total_food_items / len(successful_results), 1) if successful_results else 0,
                'average_calories_per_image': round(total_calories / len(successful_results), 1) if successful_results else 0,
                'most_common_foods': dict(food_counts.most_common(10))
            },
            'detailed_results': successful_results,
            'failed_images': self.failed_images
        }
        
        report_file = self.output_dir / 'batch_reports' / f'batch_summary_{timestamp}.json'
        with open(report_file, 'w') as f:
            json.dump(summary_report, f, indent=2)
        
        print(f"[STATS] JSON report saved: {report_file}")
    
    def _generate_csv_reports(self, timestamp):
        """Generate CSV reports for easy analysis."""
        if not self.results:
            return
        
        # Image-level summary CSV
        image_data = []
        for result in self.results:
            if 'error' not in result:
                image_data.append({
                    'filename': result['image_info']['filename'],
                    'processing_time_seconds': result['image_info']['processing_time_seconds'],
                    'total_items': result['analysis_summary']['total_items_detected'],
                    'food_items': result['analysis_summary']['food_items_count'],
                    'avg_confidence': result['analysis_summary']['avg_confidence'],
                    'total_calories': result['nutrition_totals']['calories'],
                    'total_protein': result['nutrition_totals']['protein_g'],
                    'total_carbs': result['nutrition_totals']['carbs_g'],
                    'total_fat': result['nutrition_totals']['fat_g']
                })
        
        df_images = pd.DataFrame(image_data)
        csv_file = self.output_dir / 'csv_exports' / f'image_summary_{timestamp}.csv'
        df_images.to_csv(csv_file, index=False)
        
        # Food items detail CSV
        food_data = []
        for result in self.results:
            if 'error' not in result:
                for item in result['food_items']:
                    food_data.append({
                        'image_filename': result['image_info']['filename'],
                        'item_name': item['name'],
                        'confidence': item['confidence'],
                        'is_food': item['is_food'],
                        'estimated_grams': item['portion_estimate']['estimated_grams'],
                        'calories': item['nutrition']['calories'],
                        'protein_g': item['nutrition']['protein_g'],
                        'carbs_g': item['nutrition']['carbs_g'],
                        'fat_g': item['nutrition']['fat_g']
                    })
        
        df_foods = pd.DataFrame(food_data)
        csv_file = self.output_dir / 'csv_exports' / f'food_items_{timestamp}.csv'
        df_foods.to_csv(csv_file, index=False)
        
        print(f"📈 CSV reports saved to: {self.output_dir / 'csv_exports'}")
    
    def _generate_html_dashboard(self, timestamp):
        """Generate an HTML dashboard for easy viewing."""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Food Analysis Batch Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f8ff; padding: 20px; border-radius: 10px; }}
                .summary {{ display: flex; justify-content: space-around; margin: 20px 0; }}
                .stat-box {{ background-color: #e6f3ff; padding: 15px; border-radius: 8px; text-align: center; }}
                .image-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }}
                .image-card {{ border: 1px solid #ddd; border-radius: 8px; padding: 15px; }}
                .food-list {{ list-style-type: none; padding: 0; }}
                .food-item {{ background-color: #f9f9f9; margin: 5px 0; padding: 8px; border-radius: 4px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🍽️ Food Analysis Batch Report</h1>
                <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="summary">
                <div class="stat-box">
                    <h3>{len(self.results)}</h3>
                    <p>Images Processed</p>
                </div>
                <div class="stat-box">
                    <h3>{sum(r['analysis_summary']['food_items_count'] for r in self.results if 'error' not in r)}</h3>
                    <p>Food Items Found</p>
                </div>
                <div class="stat-box">
                    <h3>{sum(r['nutrition_totals']['calories'] for r in self.results if 'error' not in r):.0f}</h3>
                    <p>Total Calories</p>
                </div>
            </div>
            
            <div class="image-grid">
        """
        
        for result in self.results[:20]:  # Show first 20 images
            if 'error' not in result:
                html_content += f"""
                <div class="image-card">
                    <h4>{result['image_info']['filename']}</h4>
                    <p><strong>Processing time:</strong> {result['image_info']['processing_time_seconds']}s</p>
                    <p><strong>Total calories:</strong> {result['nutrition_totals']['calories']:.1f}</p>
                    <ul class="food-list">
                """
                
                for item in result['food_items']:
                    if item['is_food']:
                        html_content += f"""
                        <li class="food-item">
                            {item['name']} - {item['nutrition']['calories']:.0f} cal
                        </li>
                        """
                
                html_content += """
                    </ul>
                </div>
                """
        
        html_content += """
            </div>
        </body>
        </html>
        """
        
        html_file = self.output_dir / 'batch_reports' / f'dashboard_{timestamp}.html'
        with open(html_file, 'w') as f:
            f.write(html_content)
        
        print(f"🌐 HTML dashboard saved: {html_file}")
    
    def _generate_visual_analytics(self, timestamp):
        """Generate visual analytics charts."""
        if not self.results:
            return
        
        # Create analytics plots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Food items distribution
        all_foods = []
        for result in self.results:
            if 'error' not in result:
                all_foods.extend([item['name'] for item in result['food_items'] if item['is_food']])
        
        if all_foods:
            food_counts = Counter(all_foods)
            top_foods = dict(food_counts.most_common(10))
            
            ax1.bar(top_foods.keys(), top_foods.values())
            ax1.set_title('Top 10 Most Detected Foods')
            ax1.set_ylabel('Count')
            ax1.tick_params(axis='x', rotation=45)
        
        # 2. Calories distribution
        calories_per_image = [r['nutrition_totals']['calories'] for r in self.results if 'error' not in r]
        if calories_per_image:
            ax2.hist(calories_per_image, bins=20, alpha=0.7)
            ax2.set_title('Calories Distribution per Image')
            ax2.set_xlabel('Calories')
            ax2.set_ylabel('Number of Images')
        
        # 3. Processing time distribution
        processing_times = [r['image_info']['processing_time_seconds'] for r in self.results if 'error' not in r]
        if processing_times:
            ax3.hist(processing_times, bins=20, alpha=0.7, color='green')
            ax3.set_title('Processing Time Distribution')
            ax3.set_xlabel('Processing Time (seconds)')
            ax3.set_ylabel('Number of Images')
        
        # 4. Items per image
        items_per_image = [r['analysis_summary']['food_items_count'] for r in self.results if 'error' not in r]
        if items_per_image:
            ax4.hist(items_per_image, bins=range(0, max(items_per_image)+2), alpha=0.7, color='orange')
            ax4.set_title('Food Items per Image Distribution')
            ax4.set_xlabel('Number of Food Items')
            ax4.set_ylabel('Number of Images')
        
        plt.tight_layout()
        
        # Save analytics chart
        analytics_file = self.output_dir / 'batch_reports' / f'analytics_{timestamp}.png'
        plt.savefig(analytics_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"[STATS] Analytics chart saved: {analytics_file}")
    
    def _print_final_summary(self, total_time, total_images):
        """Print final comprehensive summary."""
        successful = len(self.results)
        failed = len(self.failed_images)
        success_rate = (successful / total_images * 100) if total_images > 0 else 0
        
        print("\n" + "="*80)
        print("[SUCCESS] BATCH PROCESSING COMPLETE!")
        print("="*80)
        print(f"[STATS] Total Images: {total_images}")
        print(f"[OK] Successful: {successful}")
        print(f"[FAIL] Failed: {failed}")
        print(f"📈 Success Rate: {success_rate:.1f}%")
        print(f"[TIMER]  Total Time: {total_time/60:.1f} minutes")
        print(f"⚡ Average Time per Image: {total_time/total_images:.1f} seconds")
        
        if successful > 0:
            total_food_items = sum(r['analysis_summary']['food_items_count'] for r in self.results)
            total_calories = sum(r['nutrition_totals']['calories'] for r in self.results)
            
            print(f"\n🍎 Analysis Results:")
            print(f"   Total Food Items: {total_food_items}")
            print(f"   Total Calories: {total_calories:.1f}")
            print(f"   Average Items per Image: {total_food_items/successful:.1f}")
            print(f"   Average Calories per Image: {total_calories/successful:.1f}")
        
        print(f"\n[FOLDER] All results saved to: {self.output_dir}")
        print("="*80)

def main():
    parser = argparse.ArgumentParser(description="Batch process food images with YOLO")
    parser.add_argument('--input-dir', default='data/input',
                       help="Input directory containing images")
    parser.add_argument('--output-dir', default='data/output/yolo_results',
                       help="Output directory for results")
    parser.add_argument('--model-size', choices=['n', 's', 'm', 'l', 'x'],
                       default='n', help="YOLO model size")
    parser.add_argument('--parallel', action='store_true',
                       help="Process images in parallel")
    parser.add_argument('--workers', type=int, default=2,
                       help="Number of parallel workers")
    
    args = parser.parse_args()
    
    # Check input directory
    if not Path(args.input_dir).exists():
        print(f"[FAIL] Input directory not found: {args.input_dir}")
        return 1
    
    try:
        # Initialize and run batch processor
        processor = BatchFoodProcessor(args.model_size, args.output_dir)
        processor.process_batch(args.input_dir, args.parallel, args.workers)
        
        return 0
        
    except Exception as e:
        print(f"[FAIL] Batch processing failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())