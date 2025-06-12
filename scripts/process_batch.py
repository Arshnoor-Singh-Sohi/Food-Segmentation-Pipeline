#!/usr/bin/env python3
"""Batch process all food images in the input directory."""

import sys
import argparse
import logging
from pathlib import Path
import yaml
import time
from tqdm import tqdm
import json
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models.combined_pipeline import FoodAnalysisPipeline
from utils.visualization import FoodVisualization

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BatchProcessor:
    """Batch process all food images with progress tracking and error handling."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.pipeline = None
        self.processed_count = 0
        self.failed_count = 0
        self.results_summary = []
        self.lock = threading.Lock()
    
    def _load_config(self) -> dict:
        """Load configuration file."""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.error(f"Config file not found: {self.config_path}")
            raise
    
    def initialize_pipeline(self):
        """Initialize the food analysis pipeline once."""
        logger.info("Initializing food analysis pipeline...")
        self.pipeline = FoodAnalysisPipeline(self.config)
        logger.info("Pipeline initialized successfully!")
    
    def get_image_files(self, input_dir: Path) -> list:
        """Get all image files from input directory."""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        
        image_files = []
        for ext in image_extensions:
            image_files.extend(list(input_dir.glob(f'*{ext}')))
            image_files.extend(list(input_dir.glob(f'*{ext.upper()}')))
        
        return sorted(image_files)
    
    def process_single_image(self, image_path: Path) -> dict:
        """Process a single image and return results."""
        try:
            start_time = time.time()
            
            # Analyze the image
            results = self.pipeline.analyze_meal_image(
                str(image_path),
                interactive_points=None,
                save_results=True
            )
            
            processing_time = time.time() - start_time
            
            # Create summary for this image
            summary = {
                'image_name': image_path.name,
                'image_path': str(image_path),
                'status': 'success',
                'processing_time': round(processing_time, 2),
                'total_items': results['meal_summary']['total_items'],
                'detected_items': results['meal_summary']['detected_items'],
                'total_calories': round(results['total_nutrition']['calories'], 1),
                'food_types': results['meal_summary']['food_types'],
                'avg_quality_score': round(results['meal_summary']['avg_quality_score'], 3),
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            with self.lock:
                self.processed_count += 1
                self.results_summary.append(summary)
            
            logger.info(f"[OK] Processed: {image_path.name} ({self.processed_count} completed)")
            return summary
            
        except Exception as e:
            error_summary = {
                'image_name': image_path.name,
                'image_path': str(image_path),
                'status': 'failed',
                'error': str(e),
                'processing_time': 0,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            with self.lock:
                self.failed_count += 1
                self.results_summary.append(error_summary)
            
            logger.error(f"[FAIL] Failed: {image_path.name} - {str(e)}")
            return error_summary
    
    def process_batch_sequential(self, image_files: list) -> list:
        """Process images sequentially with progress bar."""
        logger.info(f"Processing {len(image_files)} images sequentially...")
        
        with tqdm(total=len(image_files), desc="Processing images", unit="img") as pbar:
            for image_path in image_files:
                result = self.process_single_image(image_path)
                pbar.set_postfix({
                    'Current': image_path.name[:20],
                    'Success': self.processed_count,
                    'Failed': self.failed_count
                })
                pbar.update(1)
        
        return self.results_summary
    
    def process_batch_parallel(self, image_files: list, max_workers: int = 2) -> list:
        """Process images in parallel (use carefully with GPU memory)."""
        logger.info(f"Processing {len(image_files)} images with {max_workers} workers...")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_image = {
                executor.submit(self.process_single_image, img_path): img_path 
                for img_path in image_files
            }
            
            # Process completed tasks with progress bar
            with tqdm(total=len(image_files), desc="Processing images", unit="img") as pbar:
                for future in as_completed(future_to_image):
                    image_path = future_to_image[future]
                    try:
                        result = future.result()
                        pbar.set_postfix({
                            'Success': self.processed_count,
                            'Failed': self.failed_count
                        })
                    except Exception as e:
                        logger.error(f"Unexpected error processing {image_path}: {e}")
                    
                    pbar.update(1)
        
        return self.results_summary
    
    def save_batch_summary(self, output_dir: Path):
        """Save comprehensive batch processing summary."""
        try:
            # Create summary directory
            summary_dir = output_dir / "batch_summary"
            summary_dir.mkdir(exist_ok=True)
            
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            
            # Save detailed JSON summary
            json_file = summary_dir / f"batch_results_{timestamp}.json"
            with open(json_file, 'w') as f:
                json.dump({
                    'processing_summary': {
                        'total_images': len(self.results_summary),
                        'successful': self.processed_count,
                        'failed': self.failed_count,
                        'success_rate': round(self.processed_count / len(self.results_summary) * 100, 1) if self.results_summary else 0,
                        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                    },
                    'detailed_results': self.results_summary
                }, f, indent=2)
            
            # Save CSV summary for easy analysis
            csv_file = summary_dir / f"batch_summary_{timestamp}.csv"
            df = pd.DataFrame(self.results_summary)
            df.to_csv(csv_file, index=False)
            
            # Save nutrition summary for successful images
            successful_results = [r for r in self.results_summary if r['status'] == 'success']
            if successful_results:
                nutrition_summary = {
                    'total_images_analyzed': len(successful_results),
                    'total_food_items': sum(r.get('total_items', 0) for r in successful_results),
                    'total_calories': sum(r.get('total_calories', 0) for r in successful_results),
                    'average_items_per_image': round(sum(r.get('total_items', 0) for r in successful_results) / len(successful_results), 1),
                    'average_calories_per_image': round(sum(r.get('total_calories', 0) for r in successful_results) / len(successful_results), 1),
                    'most_common_foods': self._get_most_common_foods(successful_results)
                }
                
                nutrition_file = summary_dir / f"nutrition_summary_{timestamp}.json"
                with open(nutrition_file, 'w') as f:
                    json.dump(nutrition_summary, f, indent=2)
            
            logger.info(f"[STATS] Batch summary saved to: {summary_dir}")
            
        except Exception as e:
            logger.error(f"Failed to save batch summary: {e}")
    
    def _get_most_common_foods(self, successful_results: list) -> dict:
        """Get most common food types across all images."""
        from collections import Counter
        
        all_foods = []
        for result in successful_results:
            if 'food_types' in result:
                all_foods.extend(result['food_types'])
        
        food_counts = Counter(all_foods)
        return dict(food_counts.most_common(10))
    
    def print_final_summary(self):
        """Print final batch processing summary."""
        total = len(self.results_summary)
        success_rate = (self.processed_count / total * 100) if total > 0 else 0
        
        print("\n" + "="*70)
        print("🍽️  BATCH PROCESSING COMPLETE!")
        print("="*70)
        print(f"[STATS] Total Images: {total}")
        print(f"[OK] Successful: {self.processed_count}")
        print(f"[FAIL] Failed: {self.failed_count}")
        print(f"📈 Success Rate: {success_rate:.1f}%")
        
        if self.processed_count > 0:
            successful_results = [r for r in self.results_summary if r['status'] == 'success']
            total_items = sum(r.get('total_items', 0) for r in successful_results)
            total_calories = sum(r.get('total_calories', 0) for r in successful_results)
            
            print(f"\n🔍 Analysis Results:")
            print(f"   Total Food Items Detected: {total_items}")
            print(f"   Total Calories: {total_calories:.1f}")
            print(f"   Average Items per Image: {total_items/self.processed_count:.1f}")
            print(f"   Average Calories per Image: {total_calories/self.processed_count:.1f}")
        
        print("="*70)

def main():
    parser = argparse.ArgumentParser(description="Batch process all food images")
    parser.add_argument(
        '--input-dir',
        default='data/input',
        help="Input directory containing food images"
    )
    parser.add_argument(
        '--config',
        default='config/config.yaml',
        help="Path to config file"
    )
    parser.add_argument(
        '--parallel',
        action='store_true',
        help="Process images in parallel (faster but uses more GPU memory)"
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=2,
        help="Number of parallel workers (if using --parallel)"
    )
    parser.add_argument(
        '--continue-on-error',
        action='store_true',
        default=True,
        help="Continue processing even if some images fail"
    )
    
    args = parser.parse_args()
    
    # Check input directory
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        return 1
    
    try:
        # Initialize batch processor
        processor = BatchProcessor(args.config)
        processor.initialize_pipeline()
        
        # Get all image files
        image_files = processor.get_image_files(input_dir)
        
        if not image_files:
            logger.warning(f"No image files found in {input_dir}")
            return 1
        
        logger.info(f"Found {len(image_files)} images to process")
        
        # Process images
        start_time = time.time()
        
        if args.parallel:
            processor.process_batch_parallel(image_files, args.workers)
        else:
            processor.process_batch_sequential(image_files)
        
        total_time = time.time() - start_time
        
        # Save results
        output_dir = Path(processor.config['paths']['output_dir'])
        processor.save_batch_summary(output_dir)
        
        # Print summary
        processor.print_final_summary()
        print(f"[TIMER]  Total Processing Time: {total_time/60:.1f} minutes")
        print(f"⚡ Average Time per Image: {total_time/len(image_files):.1f} seconds")
        
        return 0
        
    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())