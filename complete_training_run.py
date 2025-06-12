#!/usr/bin/env python3
"""
Complete Training Run - One script to rule them all!
This script runs the entire training process from setup to completion with full logging

Usage:
    python complete_training_run.py --mode setup_only
    python complete_training_run.py --mode quick_test
    python complete_training_run.py --mode full_training
    python complete_training_run.py --mode everything  # Does setup + quick test + full training
"""

import sys
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
import os

# Fix encoding issues for Windows
def setup_encoding():
    """Set up proper encoding for Windows compatibility"""
    try:
        # Set console to UTF-8 if possible
        if sys.platform.startswith('win'):
            os.system('chcp 65001 > nul 2>&1')
        
        # Reconfigure stdout/stderr with UTF-8 encoding
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
            sys.stderr.reconfigure(encoding='utf-8', errors='replace')
        else:
            # Fallback for older Python versions
            import io
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except Exception:
        # If UTF-8 setup fails, we'll use ASCII alternatives
        pass

# Call encoding setup immediately
setup_encoding()

def safe_print(text, use_ascii_fallback=True):
    """Print text with encoding fallback"""
    try:
        print(text)
    except UnicodeEncodeError:
        if use_ascii_fallback:
            # Replace Unicode characters with ASCII alternatives
            ascii_text = (text
                .replace('[FOOD]', '[FOOD]')
                .replace('🔍', '[SEARCH]')
                .replace('[OK]', '[OK]')
                .replace('[FAIL]', '[FAIL]')
                .replace('[RUN]', '[RUN]')
                .replace('[TIP]', '[TIP]')
                .replace('[TIME]', '[TIME]')
                .replace('[INFO]', '[INFO]')
                .replace('[ERROR]', '[ERROR]')
                .replace('[WARN]', '[WARN]')
                .replace('[SUCCESS]', '[SUCCESS]')
                .replace('[STATS]', '[STATS]')
                .replace('[FOLDER]', '[FOLDER]')
                .replace('[TEST]', '[TEST]')
                .replace('[STEP]', '[REFRESH]')
                .replace('[DONE]', '[DONE]')
                .replace('[TIMER]', '[TIMER]')
                .replace('[DATE]', '[DATE]')
                .replace('[TARGET]', '[TARGET]')
                .replace('[CLOCK]', '[CLOCK]')
                .replace('[STOP]', '[STOP]')
            )
            print(ascii_text)
        else:
            print("[ENCODING ERROR - Cannot display special characters]")

def print_banner():
    """Print an attractive banner"""
    banner = """
+==============================================================+
|                 [FOOD] TRAINING PIPELINE [FOOD]             |
|              Complete Training with Full Logging            |
+==============================================================+
    """
    safe_print(banner)

def check_prerequisites():
    """Check if all prerequisite files exist"""
    required_files = [
        "run_with_logging.py",
        "fix_training_issues.py", 
        "scripts/train_custom_food_model.py",
        "src/training/food_dataset_preparer.py",
        "src/training/food_yolo_trainer.py"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        safe_print("[FAIL] MISSING REQUIRED FILES:")
        for file_path in missing_files:
            safe_print(f"   - {file_path}")
        safe_print("\n[TIP] Please create these files first using the provided code!")
        return False
    
    safe_print("[OK] All required files found")
    return True

def run_logged_command(command, description):
    """Run a command using the logging script"""
    safe_print(f"\n[RUN] {description}")
    safe_print("=" * 60)
    
    full_command = f"python run_with_logging.py --command \"{command}\""
    
    try:
        result = subprocess.run(full_command, shell=True, check=False)
        return result.returncode == 0
    except Exception as e:
        safe_print(f"[FAIL] Error running command: {e}")
        return False

def main():
    """Main orchestrator for complete training runs"""
    print_banner()
    
    parser = argparse.ArgumentParser(
        description="Complete food model training pipeline with logging",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Training Modes:
  setup_only    - Just fix setup issues and check configuration
  quick_test    - Setup + quick test training (5-15 minutes)
  full_training - Setup + full training (1-3 hours)
  everything    - Setup + quick test + full training (complete pipeline)

Examples:
  python complete_training_run.py --mode quick_test
  python complete_training_run.py --mode full_training --epochs 100
  python complete_training_run.py --mode everything --quick-epochs 5 --full-epochs 50
        """
    )
    
    parser.add_argument(
        '--mode',
        choices=['setup_only', 'quick_test', 'full_training', 'everything'],
        required=True,
        help='Training mode to execute'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        help='Number of epochs for training (applies to both quick and full)'
    )
    
    parser.add_argument(
        '--quick-epochs',
        type=int,
        default=5,
        help='Epochs for quick test (default: 5)'
    )
    
    parser.add_argument(
        '--full-epochs',
        type=int,
        default=50,
        help='Epochs for full training (default: 50)'
    )
    
    args = parser.parse_args()
    
    # Check prerequisites
    safe_print("[SEARCH] Checking prerequisites...")
    if not check_prerequisites():
        safe_print("\n[TIP] Create the missing files and run again!")
        return
    
    # Start the training pipeline
    start_time = datetime.now()
    safe_print(f"\n[TIME] Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    success = True
    
    # Phase 1: Setup (always run)
    safe_print("\n" + "="*60)
    safe_print("[INFO] PHASE 1: SETUP AND VALIDATION")
    safe_print("="*60)
    
    if not run_logged_command("python fix_training_issues.py", "Fixing setup issues"):
        safe_print("[FAIL] Setup troubleshooting failed")
        success = False
    
    if success and not run_logged_command(
        "python scripts/train_custom_food_model.py --mode check_setup", 
        "Validating training setup"
    ):
        safe_print("[FAIL] Setup validation failed")
        success = False
    
    if not success:
        safe_print("\n[ERROR] Setup phase failed. Check the logs for details.")
        return
    
    safe_print("\n[OK] Setup phase completed successfully!")
    
    # Phase 2: Quick test (if requested)
    if args.mode in ['quick_test', 'everything'] and success:
        safe_print("\n" + "="*60)
        safe_print("[RUN] PHASE 2: QUICK TEST TRAINING")
        safe_print("="*60)
        
        quick_epochs = args.epochs or args.quick_epochs
        command = f"python scripts/train_custom_food_model.py --mode quick_test --epochs {quick_epochs}"
        
        if not run_logged_command(command, f"Quick test training ({quick_epochs} epochs)"):
            safe_print("[FAIL] Quick test training failed")
            success = False
            
            # Ask if user wants to continue to full training
            if args.mode == 'everything':
                response = input("\nQuick test failed. Continue to full training anyway? (y/N): ").strip().lower()
                if response not in ['y', 'yes']:
                    safe_print("[STOP] Stopping due to user choice")
                    return
        else:
            safe_print("\n[OK] Quick test training completed successfully!")
    
    # Phase 3: Full training (if requested)
    if args.mode in ['full_training', 'everything'] and success:
        safe_print("\n" + "="*60)
        safe_print("[TARGET] PHASE 3: FULL TRAINING")
        safe_print("="*60)
        
        full_epochs = args.epochs or args.full_epochs
        command = f"python scripts/train_custom_food_model.py --mode full_training --epochs {full_epochs}"
        
        safe_print(f"[CLOCK] This will take substantial time ({full_epochs} epochs)")
        safe_print("[TIP] You can monitor progress in real-time below")
        
        if not run_logged_command(command, f"Full training ({full_epochs} epochs)"):
            safe_print("[FAIL] Full training failed")
            success = False
        else:
            safe_print("\n[OK] Full training completed successfully!")
    
    # Final summary
    end_time = datetime.now()
    duration = end_time - start_time
    
    safe_print("\n" + "="*60)
    safe_print("[DONE] TRAINING PIPELINE COMPLETE")
    safe_print("="*60)
    safe_print(f"[TIMER] Total Duration: {duration}")
    safe_print(f"[DATE] Completed: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    if success:
        safe_print("[SUCCESS] ALL PHASES COMPLETED SUCCESSFULLY!")
        safe_print("\n[STATS] What you now have:")
        safe_print("   [OK] Fixed and validated setup")
        
        if args.mode in ['quick_test', 'everything']:
            safe_print("   [OK] Quick test model (validates your pipeline)")
        
        if args.mode in ['full_training', 'everything']:
            safe_print("   [OK] Production-ready custom food model")
            safe_print("   [FOLDER] Model saved in: data/models/custom_food_detection.pt")
        
        safe_print("\n[TIP] Next steps:")
        safe_print("   [INFO] Review logs in: logs/training_session_*/")
        safe_print("   [TEST] Test your model with: python enhanced_batch_tester.py")
        safe_print("   [REFRESH] Use your custom model in your existing pipeline")
        
    else:
        safe_print("[WARN] Some phases had issues")
        safe_print("[INFO] Check the detailed logs for troubleshooting information")
        safe_print("[TIP] You can re-run individual phases after fixing issues")
    
    safe_print(f"\n[FOLDER] All detailed logs available in: logs/")
    safe_print("[SEARCH] Check the logs for complete details of everything that happened!")

if __name__ == "__main__":
    main()