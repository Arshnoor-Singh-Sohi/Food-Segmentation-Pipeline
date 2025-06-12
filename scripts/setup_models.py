#!/usr/bin/env python3
"""Setup script to download and configure models for the food segmentation pipeline."""

import os
import sys
import requests
from pathlib import Path
import yaml
import logging
import argparse
from urllib.parse import urlparse
from tqdm import tqdm

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelSetup:
    """Handle downloading and setting up models for the pipeline."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.models_dir = Path(self.config['paths']['models_dir'])
        self.models_dir.mkdir(exist_ok=True, parents=True)
    
    def _load_config(self) -> dict:
        """Load configuration file."""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.error(f"Config file not found: {self.config_path}")
            raise
    
    def download_file(self, url: str, destination: Path, description: str = ""):
        """Download file with progress bar."""
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(destination, 'wb') as f, tqdm(
                desc=description,
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    size = f.write(chunk)
                    pbar.update(size)
            
            logger.info(f"Downloaded: {destination}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to download {url}: {e}")
            return False
    
    def setup_sam2(self):
        """Set up SAM 2 model."""
        logger.info("Setting up SAM 2...")
        
        # SAM 2 model URLs
        sam2_models = {
            "sam2.1_hiera_tiny.pt": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt",
            "sam2.1_hiera_small.pt": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt",
            "sam2.1_hiera_base_plus.pt": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt",
            "sam2.1_hiera_large.pt": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
        }
        
        # Download default model (base_plus)
        default_model = "sam2.1_hiera_base_plus.pt"
        model_path = self.models_dir / default_model
        
        if not model_path.exists():
            logger.info(f"Downloading SAM 2 model: {default_model}")
            success = self.download_file(
                sam2_models[default_model], 
                model_path,
                f"SAM 2 {default_model}"
            )
            if not success:
                logger.error("Failed to download SAM 2 model")
                return False
        else:
            logger.info(f"SAM 2 model already exists: {model_path}")
        
        # Clone SAM 2 repository if not exists
        sam2_repo_dir = Path("sam2")
        if not sam2_repo_dir.exists():
            logger.info("Cloning SAM 2 repository...")
            os.system("git clone https://github.com/facebookresearch/sam2.git")
            
            # Install SAM 2
            logger.info("Installing SAM 2...")
            os.chdir("sam2")
            os.system("pip install -e .")
            os.system("pip install -e '.[notebooks]'")
            os.chdir("..")
        
        return True
    
    def setup_yolo(self):
        """Set up YOLO model for food detection."""
        logger.info("Setting up YOLO food detection model...")
        
        # Try to download food-specific YOLO model
        # Note: You'll need to replace this with actual food model URLs
        food_model_urls = {
            # Example URLs - replace with actual food-trained models
            "yolo_food_v8.pt": "https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5s.pt"
        }
        
        # For now, we'll use the standard YOLOv8 model
        # The pipeline will enhance it for food detection
        logger.info("Using YOLOv8n as base model (will be enhanced for food detection)")
        
        try:
            from ultralytics import YOLO
            # This will automatically download YOLOv8n if not present
            model = YOLO('yolov8n.pt')
            
            # Save to our models directory
            yolo_path = self.models_dir / "yolo_food_v8.pt"
            model.save(str(yolo_path))
            
            logger.info(f"YOLO model ready: {yolo_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup YOLO: {e}")
            return False
    
    def create_nutrition_database(self):
        """Create sample nutrition database."""
        logger.info("Creating nutrition database...")
        
        nutrition_data = {
            "apple": {
                "calories_per_100g": 52,
                "protein": 0.3,
                "carbohydrates": 14,
                "fat": 0.2,
                "fiber": 2.4,
                "sugar": 10.4,
                "sodium": 1
            },
            "banana": {
                "calories_per_100g": 89,
                "protein": 1.1,
                "carbohydrates": 23,
                "fat": 0.3,
                "fiber": 2.6,
                "sugar": 12,
                "sodium": 1
            },
            "orange": {
                "calories_per_100g": 47,
                "protein": 0.9,
                "carbohydrates": 12,
                "fat": 0.1,
                "fiber": 2.4,
                "sugar": 9.4,
                "sodium": 0
            },
            "pizza": {
                "calories_per_100g": 266,
                "protein": 11,
                "carbohydrates": 33,
                "fat": 10,
                "fiber": 2.3,
                "sugar": 3.6,
                "sodium": 598
            },
            "sandwich": {
                "calories_per_100g": 250,
                "protein": 12,
                "carbohydrates": 30,
                "fat": 8,
                "fiber": 3,
                "sugar": 4,
                "sodium": 500
            },
            "salad": {
                "calories_per_100g": 15,
                "protein": 1.4,
                "carbohydrates": 2.9,
                "fat": 0.2,
                "fiber": 1.3,
                "sugar": 2,
                "sodium": 28
            },
            "chicken": {
                "calories_per_100g": 239,
                "protein": 27,
                "carbohydrates": 0,
                "fat": 14,
                "fiber": 0,
                "sugar": 0,
                "sodium": 82
            },
            "broccoli": {
                "calories_per_100g": 34,
                "protein": 2.8,
                "carbohydrates": 7,
                "fat": 0.4,
                "fiber": 2.6,
                "sugar": 1.5,
                "sodium": 33
            },
            "carrot": {
                "calories_per_100g": 41,
                "protein": 0.9,
                "carbohydrates": 10,
                "fat": 0.2,
                "fiber": 2.8,
                "sugar": 4.7,
                "sodium": 69
            },
            "rice": {
                "calories_per_100g": 130,
                "protein": 2.7,
                "carbohydrates": 28,
                "fat": 0.3,
                "fiber": 0.4,
                "sugar": 0.1,
                "sodium": 5
            }
        }
        
        # Save nutrition database
        db_path = Path(self.config['nutrition']['database_path'])
        db_path.parent.mkdir(exist_ok=True, parents=True)
        
        import json
        with open(db_path, 'w') as f:
            json.dump(nutrition_data, f, indent=2)
        
        logger.info(f"Nutrition database created: {db_path}")
        return True
    
    def verify_setup(self):
        """Verify that all models are properly set up."""
        logger.info("Verifying setup...")
        
        # Check SAM 2
        sam2_checkpoint = Path(self.config['models']['sam2']['checkpoint_path'])
        if not sam2_checkpoint.exists():
            logger.error(f"SAM 2 checkpoint not found: {sam2_checkpoint}")
            return False
        
        # Check YOLO
        yolo_model = Path(self.config['models']['yolo']['model_path'])
        if not yolo_model.exists():
            logger.warning(f"YOLO model not found: {yolo_model}")
        
        # Check nutrition database
        nutrition_db = Path(self.config['nutrition']['database_path'])
        if not nutrition_db.exists():
            logger.error(f"Nutrition database not found: {nutrition_db}")
            return False
        
        logger.info("Setup verification complete!")
        return True
    
    def run_setup(self, components: list = None):
        """Run complete setup process."""
        if components is None:
            components = ['sam2', 'yolo', 'nutrition']
        
        success = True
        
        if 'sam2' in components:
            success &= self.setup_sam2()
        
        if 'yolo' in components:
            success &= self.setup_yolo()
        
        if 'nutrition' in components:
            success &= self.create_nutrition_database()
        
        if success:
            success &= self.verify_setup()
        
        if success:
            logger.info("[OK] Setup completed successfully!")
            logger.info("You can now run: python scripts/process_single_image.py <image_path>")
        else:
            logger.error("[FAIL] Setup failed. Please check the errors above.")
        
        return success

def main():
    parser = argparse.ArgumentParser(description="Setup models for food segmentation pipeline")
    parser.add_argument(
        '--components', 
        nargs='+', 
        choices=['sam2', 'yolo', 'nutrition'],
        default=['sam2', 'yolo', 'nutrition'],
        help="Components to setup"
    )
    parser.add_argument(
        '--config', 
        default='config/config.yaml',
        help="Path to config file"
    )
    
    args = parser.parse_args()
    
    setup = ModelSetup(args.config)
    setup.run_setup(args.components)

if __name__ == "__main__":
    main()