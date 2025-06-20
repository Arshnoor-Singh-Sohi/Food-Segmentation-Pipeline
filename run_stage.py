"""
Universal Stage Runner for Food Segmentation Pipeline
====================================================

Single command interface for all stages. Place this file in project root.

Usage:
python run_stage.py 1a --image data/input/refrigerator.jpg
python run_stage.py 1a --refrigerator  
python run_stage.py 1a --test-all
python run_stage.py 1b --display-test
python run_stage.py 2a --bottles

This keeps all stage commands consistent and easy to remember.
"""

import argparse
import sys
import os
from pathlib import Path
import importlib.util

class StageRunner:
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.stages_dir = self.project_root / "stages"
        
    def setup_project_structure(self):
        """Create the organized directory structure"""
        stages = [
            "stage1a_detection_fixes",
            "stage1b_display_fixes", 
            "stage1c_enhanced_detection",
            "stage2a_bottle_detection",
            "stage2b_bottle_ocr",
            "stage2c_bottle_classification",
            "stage3a_receipt_ocr",
            "stage3b_package_ocr"
        ]
        
        # Create stages directories
        for stage in stages:
            stage_dir = self.stages_dir / stage
            stage_dir.mkdir(parents=True, exist_ok=True)
            
            # Create output directory for each stage
            output_dir = self.project_root / "data" / "output" / f"{stage}_results"
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create other necessary directories
        other_dirs = [
            "data/input",
            "data/models", 
            "config"
        ]
        
        for directory in other_dirs:
            (self.project_root / directory).mkdir(parents=True, exist_ok=True)
        
        print("✅ Project structure created")
        
    def run_stage(self, stage_id, args):
        """Run a specific stage with given arguments"""
        stage_map = {
            "1a": "stage1a_detection_fixes",
            "1b": "stage1b_display_fixes",
            "1c": "stage1c_enhanced_detection", 
            "2a": "stage2a_bottle_detection",
            "2b": "stage2b_bottle_ocr",
            "2c": "stage2c_bottle_classification",
            "3a": "stage3a_receipt_ocr",
            "3b": "stage3b_package_ocr"
        }
        
        if stage_id not in stage_map:
            print(f"❌ Invalid stage: {stage_id}")
            print(f"Available stages: {', '.join(stage_map.keys())}")
            return False
            
        stage_folder = stage_map[stage_id]
        stage_path = self.stages_dir / stage_folder
        
        if not stage_path.exists():
            print(f"❌ Stage folder not found: {stage_path}")
            print(f"Run: python run_stage.py setup")
            return False
            
        # Look for stage runner file
        runner_files = [
            f"{stage_id}_runner.py",
            f"stage{stage_id}_runner.py", 
            "runner.py",
            "main.py"
        ]
        
        runner_file = None
        for filename in runner_files:
            potential_file = stage_path / filename
            if potential_file.exists():
                runner_file = potential_file
                break
                
        if not runner_file:
            print(f"❌ No runner file found in {stage_path}")
            print(f"Expected one of: {runner_files}")
            return False
            
        # Import and run the stage
        try:
            spec = importlib.util.spec_from_file_location("stage_module", runner_file)
            stage_module = importlib.util.module_from_spec(spec)
            sys.modules["stage_module"] = stage_module
            spec.loader.exec_module(stage_module)
            
            # Call the stage's main function with arguments
            if hasattr(stage_module, 'run_stage'):
                return stage_module.run_stage(args)
            elif hasattr(stage_module, 'main'):
                return stage_module.main(args)
            else:
                print(f"❌ Stage {stage_id} missing run_stage() or main() function")
                return False
                
        except Exception as e:
            print(f"❌ Error running stage {stage_id}: {str(e)}")
            return False
    
    def list_stages(self):
        """List all available stages"""
        print("📋 Available Stages:")
        print("==================")
        print("PHASE 1: Detection Fixes")
        print("  1a - Detection accuracy fixes (bottle, banana counting)")
        print("  1b - Display formatting fixes") 
        print("  1c - Enhanced item detection")
        print()
        print("PHASE 2: Bottle Enhancement")
        print("  2a - Specialized bottle detection")
        print("  2b - OCR for labeled bottles")
        print("  2c - Non-labeled bottle classification")
        print()
        print("PHASE 3: OCR Integration")
        print("  3a - Grocery receipt OCR")
        print("  3b - Package label OCR")
        print()
        print("Commands:")
        print("  python run_stage.py setup                    # Create structure")
        print("  python run_stage.py 1a --refrigerator       # Run stage 1a")
        print("  python run_stage.py 1a --image path.jpg     # Run on specific image")
        print("  python run_stage.py list                    # Show this list")

def main():
    parser = argparse.ArgumentParser(description='Universal Stage Runner')
    parser.add_argument('stage', nargs='?', help='Stage to run (1a, 1b, 2a, etc.) or "setup"/"list"')
    parser.add_argument('--image', type=str, help='Path to image file')
    parser.add_argument('--refrigerator', action='store_true', help='Test refrigerator scenario')
    parser.add_argument('--test-all', action='store_true', help='Run all tests')
    parser.add_argument('--bottles', action='store_true', help='Focus on bottle detection')
    parser.add_argument('--display-test', action='store_true', help='Test display formatting')
    parser.add_argument('--setup', action='store_true', help='Setup project structure')
    
    args = parser.parse_args()
    
    runner = StageRunner()
    
    # Handle special commands
    if args.stage == 'setup' or args.setup:
        runner.setup_project_structure()
        return
        
    if args.stage == 'list' or not args.stage:
        runner.list_stages()
        return
    
    # Run the specified stage
    print(f"🚀 Running Stage {args.stage.upper()}")
    print("=" * 40)
    
    success = runner.run_stage(args.stage, args)
    
    if success:
        print(f"\n✅ Stage {args.stage.upper()} completed successfully!")
    else:
        print(f"\n❌ Stage {args.stage.upper()} failed!")

if __name__ == "__main__":
    main()