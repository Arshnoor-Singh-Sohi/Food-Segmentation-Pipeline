#!/usr/bin/env python3
"""
Complete setup script for metadata labeling system
Run this FIRST to set up everything
"""

import os
import sys
import subprocess
import json
import yaml
from pathlib import Path
import requests
from tqdm import tqdm
import zipfile
import shutil
import sys
sys.stdout.reconfigure(encoding='utf-8')


class MetadataSystemSetup:
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.setup_complete = False
        
    def run_complete_setup(self):
        """Run the complete setup process"""
        print("🚀 METADATA LABELING SYSTEM SETUP")
        print("="*50)
        
        steps = [
            ("Creating directories", self.create_directory_structure),
            ("Installing dependencies", self.install_dependencies),
            ("Downloading models", self.download_models),
            ("Building databases", self.build_databases),
            ("Creating configurations", self.create_configurations),
            ("Setting up RunPod files", self.setup_runpod_files),
            ("Validating setup", self.validate_setup)
        ]
        
        for step_name, step_func in steps:
            print(f"\n📋 {step_name}...")
            try:
                step_func()
                print(f"✅ {step_name} completed")
            except Exception as e:
                print(f"❌ {step_name} failed: {e}")
                return False
        
        self.setup_complete = True
        print("\n🎉 SETUP COMPLETE!")
        self.print_next_steps()
        return True
    
    def create_directory_structure(self):
        """Create all necessary directories"""
        directories = [
            "data/databases/nutrition",
            "data/databases/food_taxonomy",
            "data/databases/cuisine_mapping",
            "data/models/metadata_models",
            "data/output/metadata_results",
            "src/metadata",
            "src/databases",
            "src/pipeline",
            "runpod",
            "config"
        ]
        
        for dir_path in directories:
            full_path = self.project_root / dir_path
            full_path.mkdir(parents=True, exist_ok=True)
            print(f"  📁 Created: {dir_path}")
    
    def install_dependencies(self):
        """Install required Python packages"""
        packages = [
            "transformers>=4.30.0",
            "torch>=2.0.0",
            "torchvision>=0.15.0",
            "timm>=0.9.0",
            "scikit-learn>=1.0.0",
            "pandas>=2.0.0",
            "sqlalchemy>=2.0.0",
            "pillow>=10.0.0",
            "opencv-python>=4.8.0",
            "tqdm>=4.65.0",
            "pyyaml>=6.0",
            "requests>=2.31.0",
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0"
        ]
        
        # Create requirements file
        req_file = self.project_root / "requirements_metadata.txt"
        with open(req_file, 'w') as f:
            f.write('\n'.join(packages))
        
        print(f"  📝 Created requirements file: {req_file}")
        
        # Install packages
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(req_file)])
    
    def download_models(self):
        """Download pre-trained models for metadata extraction"""
        
        # Download Food-101 classifier
        print("  🤖 Downloading Food-101 classifier...")
        try:
            from transformers import AutoModelForImageClassification, AutoImageProcessor
            
            model = AutoModelForImageClassification.from_pretrained("nateraw/food")
            processor = AutoImageProcessor.from_pretrained("nateraw/food")
            
            # Save locally
            model_path = self.project_root / "data/models/metadata_models/food101"
            model.save_pretrained(model_path)
            processor.save_pretrained(model_path)
            
            print("  ✅ Food-101 classifier downloaded")
        except Exception as e:
            print(f"  ⚠️  Could not download Food-101 model: {e}")
    
    def build_databases(self):
        """Build all necessary databases"""
        # This creates the actual database builder
        db_builder_file = self.project_root / "src/databases/build_nutrition_db.py"
        db_builder_file.parent.mkdir(parents=True, exist_ok=True)
        
        db_builder_code = '''#!/usr/bin/env python3
"""Build comprehensive nutrition database"""

import json
import sqlite3
import pandas as pd
from pathlib import Path

class NutritionDatabaseBuilder:
    def __init__(self, db_path="data/databases/nutrition/nutrition_expanded.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
    def build_database(self):
        """Build the complete nutrition database"""
        conn = sqlite3.connect(self.db_path)
        
        # Create tables
        conn.execute("""
            CREATE TABLE IF NOT EXISTS foods (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                category TEXT,
                cuisine TEXT,
                common_names TEXT
            )
        """)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS nutrition (
                food_id INTEGER PRIMARY KEY,
                calories REAL,
                protein REAL,
                carbohydrates REAL,
                fat REAL,
                fiber REAL,
                sugar REAL,
                sodium REAL,
                FOREIGN KEY (food_id) REFERENCES foods (id)
            )
        """)
        
        # Add sample data (expand this with real data)
        sample_foods = [
            ("pizza", "main_dish", "italian", "pizza pie,za"),
            ("hamburger", "main_dish", "american", "burger,beef burger"),
            ("sushi", "main_dish", "japanese", "maki,nigiri"),
            ("pasta", "main_dish", "italian", "spaghetti,noodles"),
            ("salad", "side_dish", "international", "green salad,garden salad"),
        ]
        
        sample_nutrition = [
            (1, 266, 11, 33, 10, 2.3, 3.6, 598),  # pizza
            (2, 295, 17, 30, 14, 2.1, 6.8, 414),  # hamburger
            (3, 200, 9, 38, 0.7, 1.5, 8, 428),    # sushi
            (4, 220, 8, 43, 1.3, 2.5, 2, 1),      # pasta
            (5, 20, 1.3, 3.7, 0.2, 1.6, 2.5, 32), # salad
        ]
        
        # Insert data
        for food_data in sample_foods:
            conn.execute("INSERT OR IGNORE INTO foods (name, category, cuisine, common_names) VALUES (?, ?, ?, ?)", 
                        food_data)
        
        # Get food IDs and insert nutrition
        for i, nutrition_data in enumerate(sample_nutrition):
            conn.execute("INSERT OR REPLACE INTO nutrition VALUES (?, ?, ?, ?, ?, ?, ?, ?)", 
                        nutrition_data)
        
        conn.commit()
        conn.close()
        
        print(f"  ✅ Nutrition database created at {self.db_path}")
        
        # Also create JSON version
        self.export_to_json()
        
    def export_to_json(self):
        """Export database to JSON"""
        conn = sqlite3.connect(self.db_path)
        
        query = """
            SELECT f.name, f.category, f.cuisine, n.*
            FROM foods f
            JOIN nutrition n ON f.id = n.food_id
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        # Convert to dictionary format
        nutrition_dict = {}
        for _, row in df.iterrows():
            nutrition_dict[row['name']] = {
                'category': row['category'],
                'cuisine': row['cuisine'],
                'calories': row['calories'],
                'protein': row['protein'],
                'carbohydrates': row['carbohydrates'],
                'fat': row['fat'],
                'fiber': row['fiber'],
                'sugar': row['sugar'],
                'sodium': row['sodium']
            }
        
        json_path = self.db_path.parent / "nutrition_expanded.json"
        with open(json_path, 'w') as f:
            json.dump(nutrition_dict, f, indent=2)
        
        print(f"  ✅ JSON database exported to {json_path}")

if __name__ == "__main__":
    builder = NutritionDatabaseBuilder()
    builder.build_database()
'''
        
        with open(db_builder_file, 'w') as f:
            f.write(db_builder_code)
        
        # Run the builder
        subprocess.check_call([sys.executable, str(db_builder_file)])
    
    def create_configurations(self):
        """Create all configuration files"""
        
        # Metadata configuration
        metadata_config = {
            'models': {
                'food_classifier': {
                    'model_path': 'data/models/metadata_models/food101',
                    'confidence_threshold': 0.7
                },
                'portion_estimator': {
                    'reference_size_cm': 23.0,  # Average plate diameter
                    'density_factors': {
                        'default': 1.0,
                        'salad': 0.3,
                        'soup': 0.9,
                        'meat': 1.2
                    }
                }
            },
            'databases': {
                'nutrition': 'data/databases/nutrition/nutrition_expanded.db',
                'cuisine_mapping': 'data/databases/cuisine_mapping/cuisine_patterns.json'
            },
            'output': {
                'save_crops': True,
                'save_metadata_json': True,
                'save_visualization': True
            }
        }
        
        config_path = self.project_root / "config/metadata_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(metadata_config, f, default_flow_style=False)
        
        print(f"  ✅ Created metadata configuration: {config_path}")
    
    def setup_runpod_files(self):
        """Create RunPod-specific files"""
        
        # Dockerfile
        dockerfile_content = '''FROM runpod/pytorch:2.0.1-py3.10-cuda11.8.0-devel

WORKDIR /workspace

# Copy requirements
COPY requirements_metadata.txt .
RUN pip install -r requirements_metadata.txt

# Copy project files
COPY . .

# Download models at build time
RUN python scripts/setup_metadata_system.py --download-only

# Expose ports
EXPOSE 8888 5000

# Start script
CMD ["/workspace/runpod/start_server.sh"]
'''
        
        dockerfile_path = self.project_root / "runpod/Dockerfile"
        with open(dockerfile_path, 'w') as f:
            f.write(dockerfile_content)
        
        # Start script
        start_script = '''#!/bin/bash
# Start Jupyter Lab in background
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root &

# Start Flask API server
python src/api/metadata_api.py

# Keep container running
tail -f /dev/null
'''
        
        start_script_path = self.project_root / "runpod/start_server.sh"
        with open(start_script_path, 'w') as f:
            f.write(start_script)
        
        os.chmod(start_script_path, 0o755)
        
        print("  ✅ RunPod files created")
    
    def validate_setup(self):
        """Validate that everything is set up correctly"""
        checks = {
            "Directories": [
                "data/databases/nutrition",
                "data/models/metadata_models",
                "src/metadata",
                "runpod"
            ],
            "Files": [
                "config/metadata_config.yaml",
                "runpod/Dockerfile",
                "requirements_metadata.txt"
            ],
            "Databases": [
                "data/databases/nutrition/nutrition_expanded.db",
                "data/databases/nutrition/nutrition_expanded.json"
            ]
        }
        
        all_good = True
        for category, items in checks.items():
            print(f"\n  Checking {category}:")
            for item in items:
                path = self.project_root / item
                if path.exists():
                    print(f"    ✅ {item}")
                else:
                    print(f"    ❌ {item} - MISSING")
                    all_good = False
        
        return all_good
    
    def print_next_steps(self):
        """Print what to do next"""
        print("\n" + "="*50)
        print("🎯 NEXT STEPS:")
        print("="*50)
        print("\n1. LOCAL TESTING:")
        print("   python scripts/process_with_metadata.py --image data/input/pizza.jpg")
        print("\n2. RUNPOD DEPLOYMENT:")
        print("   - Login to RunPod")
        print("   - Deploy Pod with: runpod/pytorch:2.0.1-py3.10-cuda11.8.0-devel")
        print("   - Upload your code")
        print("   - Run: bash runpod/start_server.sh")
        print("\n3. API TESTING:")
        print("   curl -X POST http://your-runpod-url:5000/analyze \\")
        print("     -F 'image=@data/input/pizza.jpg'")

if __name__ == "__main__":
    setup = MetadataSystemSetup()
    setup.run_complete_setup()