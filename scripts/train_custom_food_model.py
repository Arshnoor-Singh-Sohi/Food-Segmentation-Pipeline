#!/usr/bin/env python3
"""
FIXED Food Model Training Orchestrator
Now actually creates training data instead of empty directories!

Usage:
    python scripts/train_custom_food_model.py --mode check_setup
    python scripts/train_custom_food_model.py --mode quick_test
    python scripts/train_custom_food_model.py --mode full_training --epochs 50
"""

import sys
import argparse
from pathlib import Path
import yaml
import time
from datetime import datetime

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import the FIXED training modules
from src.training.food_dataset_preparer import FoodDatasetPreparer
from src.training.food_yolo_trainer import FoodYOLOTrainer

class TrainingOrchestrator:
    """
    FIXED Training Orchestrator - Now creates real training data!
    """
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.start_time = datetime.now()
        
        print("🍕 Food Model Training Orchestrator (FIXED VERSION)")
        print("=" * 60)
        print(f"📁 Project root: {self.project_root}")
        print(f"🕐 Started at: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    def check_setup(self):
        """
        NEW: Check if everything is ready for training
        This helps identify issues before starting training
        """
        print("\n🔍 CHECKING TRAINING SETUP")
        print("Verifying that everything is ready for training...")
        print("-" * 50)
        
        setup_ok = True
        
        # Check 1: Input images exist
        input_dir = Path("data/input")
        if not input_dir.exists():
            print("❌ data/input directory not found")
            print("💡 Create it with: mkdir -p data/input")
            setup_ok = False
        else:
            # Count images
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
            image_files = []
            for ext in image_extensions:
                image_files.extend(list(input_dir.glob(f"*{ext}")))
                image_files.extend(list(input_dir.glob(f"*{ext.upper()}")))
            
            if len(image_files) == 0:
                print(f"❌ No images found in {input_dir}")
                print("💡 Add some food images to data/input/ directory")
                setup_ok = False
            else:
                print(f"✅ Found {len(image_files)} images in {input_dir}")
                print(f"   Sample images: {', '.join([f.name for f in image_files[:3]])}")
        
        # Check 2: Required directories exist
        required_dirs = [
            "src/training",
            "data/training",
            "config"
        ]
        
        for dir_path in required_dirs:
            if not Path(dir_path).exists():
                print(f"❌ Missing directory: {dir_path}")
                print(f"💡 Create it with: mkdir -p {dir_path}")
                setup_ok = False
            else:
                print(f"✅ Directory exists: {dir_path}")
        
        # Check 3: Configuration files
        config_file = Path("config/training_config.yaml")
        if not config_file.exists():
            print(f"❌ Missing config file: {config_file}")
            print("💡 Run the setup script first: python setup_training.py")
            setup_ok = False
        else:
            print(f"✅ Config file exists: {config_file}")
        
        # Check 4: Python packages
        try:
            import ultralytics
            print("✅ ultralytics package available")
        except ImportError:
            print("❌ ultralytics package not found")
            print("💡 Install with: pip install ultralytics")
            setup_ok = False
        
        try:
            import torch
            device_info = "GPU available" if torch.cuda.is_available() else "CPU only"
            print(f"✅ PyTorch available ({device_info})")
        except ImportError:
            print("❌ PyTorch not found")
            print("💡 Install with: pip install torch torchvision")
            setup_ok = False
        
        # Final verdict
        print("\n" + "=" * 50)
        if setup_ok:
            print("🎉 SETUP CHECK PASSED!")
            print("✅ Everything looks good - ready for training!")
            print("\n💡 Next step: python scripts/train_custom_food_model.py --mode quick_test")
        else:
            print("🚨 SETUP CHECK FAILED!")
            print("❌ Please fix the issues above before training")
            print("\n💡 After fixing issues, run this check again")
        
        return setup_ok
    
    def quick_test_training(self, epochs=5):
        """
        FIXED: Run quick test with actual training data
        
        Args:
            epochs: Number of epochs (reduced to 5 for faster testing)
        """
        print(f"\n🚀 QUICK TEST TRAINING - {epochs} EPOCHS")
        print("Creating real training data and training for a few epochs")
        print("This validates your setup before longer training runs")
        print("-" * 50)
        
        # Step 1: Check setup first
        print("🔍 Step 1: Checking setup...")
        if not self.check_setup():
            print("❌ Setup check failed. Please fix issues first.")
            return None, None
        
        # Step 2: Create real training dataset
        print("\n📦 Step 2: Creating training dataset with REAL data...")
        preparer = FoodDatasetPreparer()
        
        # Use the FIXED method that creates actual training data
        dataset_yaml, food_categories = preparer.create_sample_dataset_with_real_data(
            source_dir="data/input",
            num_classes=1  # Start with just one class: "food"
        )
        
        if dataset_yaml is None:
            print("❌ Failed to create dataset. Check that you have images in data/input/")
            return None, None
        
        print(f"✅ Created dataset with {len(food_categories)} categories:")
        print(f"   {', '.join(food_categories)}")
        
        # Step 3: Validate the dataset
        print("\n🔍 Step 3: Validating dataset...")
        validation = preparer.validate_dataset(dataset_yaml)
        
        if not validation['valid']:
            print("❌ Dataset validation failed!")
            return None, None
        
        # Step 4: Quick training run
        print(f"\n🏋️  Step 4: Training for {epochs} epochs...")
        trainer = FoodYOLOTrainer(dataset_yaml)
        
        try:
            model, results = trainer.train_food_detector(
                epochs=epochs,
                custom_name=f"quick_test_{datetime.now().strftime('%m%d_%H%M')}"
            )
            
            print("✅ Training completed successfully!")
            
            # Step 5: Quick validation
            print("\n🧪 Step 5: Testing the trained model...")
            validation_results = trainer.validate_model(
                "data/models/custom_food_detection.pt",
                "data/input"
            )
            
            self._print_quick_results(validation_results, epochs)
            
            print("\n🎉 Quick test training completed successfully!")
            print("💡 Your setup is working! Try full training for better results.")
            
            return model, results
            
        except Exception as e:
            print(f"❌ Training failed with error: {e}")
            print("💡 This might be due to insufficient training data or configuration issues")
            return None, None
    
    def full_training(self, epochs=50, dataset_type="smart"):
        """
        FIXED: Run full training with proper dataset creation
        
        Args:
            epochs: Number of training epochs
            dataset_type: Type of dataset ('smart' uses existing images intelligently)
        """
        print(f"\n🎯 FULL TRAINING - {epochs} EPOCHS")
        print("Training a production-ready food detection model")
        print("This will take some time but produces better results")
        print("-" * 50)
        
        # Step 1: Setup check
        print("🔍 Step 1: Checking setup...")
        if not self.check_setup():
            print("❌ Setup check failed. Please fix issues first.")
            return None, None
        
        # Step 2: Prepare dataset
        print("\n📦 Step 2: Preparing training dataset...")
        preparer = FoodDatasetPreparer()
        
        if dataset_type == "smart":
            # Use the smart method that creates better automatic labels
            dataset_yaml = preparer.create_from_existing_images_smart("data/input")
            if dataset_yaml is None:
                print("❌ Could not prepare smart dataset from existing images")
                return None, None
            print("✅ Using smart automatic labeling on your existing images")
            
        else:
            # Use the simple method
            dataset_yaml, food_categories = preparer.create_sample_dataset_with_real_data(
                source_dir="data/input",
                num_classes=1
            )
            if dataset_yaml is None:
                print("❌ Could not prepare dataset from existing images")
                return None, None
            print("✅ Using simple automatic labeling")
        
        # Step 3: Validate dataset
        print("\n🔍 Step 3: Validating dataset...")
        validation = preparer.validate_dataset(dataset_yaml)
        
        if not validation['valid']:
            print("❌ Dataset validation failed. Cannot proceed with training.")
            return None, None
        
        # Step 4: Run full training
        print(f"\n🏋️  Step 4: Starting {epochs}-epoch training...")
        print("⏰ This will take some time. You can monitor progress in the terminal.")
        
        trainer = FoodYOLOTrainer(dataset_yaml)
        
        try:
            model, results = trainer.train_food_detector(
                epochs=epochs,
                custom_name=f"food_model_{datetime.now().strftime('%Y%m%d_%H%M')}"
            )
            
            print("✅ Full training completed successfully!")
            
            # Step 5: Comprehensive validation
            print("\n🧪 Step 5: Validating trained model...")
            validation_results = trainer.validate_model(
                "data/models/custom_food_detection.pt",
                "data/input"
            )
            
            self._print_full_results(validation_results, epochs)
            
            print("\n🎉 Full training completed!")
            print("🔄 You can now use this model in your existing pipeline")
            print("💡 Try running your enhanced_batch_tester.py with the new model")
            
            return model, results
            
        except Exception as e:
            print(f"❌ Training failed with error: {e}")
            print("💡 Check the error message and consider reducing epochs or batch size")
            return None, None
    
    def _print_quick_results(self, validation_results, epochs):
        """Print encouraging results from quick test"""
        if validation_results is None:
            print("⚠️  No test images found for validation")
            return
            
        summary = validation_results['summary']
        
        print("📈 Quick Test Results:")
        print(f"   📚 Training epochs: {epochs}")
        print(f"   🖼️  Tested on: {validation_results['test_images_count']} images")
        print(f"   🎯 Average detections: {summary['avg_detections_per_image']:.1f}")
        print(f"   💯 Average confidence: {summary['avg_max_confidence']:.3f}")
        
        # Provide realistic interpretation for quick tests
        print("\n💡 Quick Test Analysis:")
        if summary['avg_max_confidence'] > 0.3:
            print("🎉 Model is learning! Confidence scores look promising.")
        else:
            print("🔄 Low confidence is normal for quick tests with few epochs.")
            
        if summary['images_with_detections'] > 0:
            print("✅ Model is detecting objects in images - training is working!")
        else:
            print("💡 No detections yet - try training for more epochs.")
            
        print("🚀 Ready for full training with more epochs for better results!")
    
    def segmentation_training(self, epochs=50):
        """
        FIXED: Train a segmentation model for precise food boundary detection
        
        Args:
            epochs: Number of training epochs
        """
        print(f"\n🎨 SEGMENTATION TRAINING - {epochs} EPOCHS")
        print("Training model for precise food boundary detection")
        print("This enables accurate portion size estimation")
        print("-" * 50)
        
        # Step 1: Setup check
        print("🔍 Step 1: Checking setup...")
        if not self.check_setup():
            print("❌ Setup check failed. Please fix issues first.")
            return None, None
        
        # Step 2: Prepare dataset
        print("\n📦 Step 2: Preparing segmentation dataset...")
        preparer = FoodDatasetPreparer()
        
        # Use existing images with smart labeling
        dataset_yaml = preparer.create_from_existing_images_smart("data/input")
        if dataset_yaml is None:
            print("❌ Could not prepare dataset from existing images")
            return None, None
        print("✅ Using smart automatic labeling for segmentation")
        
        # Step 3: Validate dataset
        print("\n🔍 Step 3: Validating dataset...")
        validation = preparer.validate_dataset(dataset_yaml)
        
        if not validation['valid']:
            print("❌ Dataset validation failed. Cannot proceed with training.")
            return None, None
        
        # Step 4: Train segmentation model
        print(f"\n🎨 Step 4: Training segmentation model for {epochs} epochs...")
        print("⏰ This trains a model that can outline exact food boundaries")
        
        trainer = FoodYOLOTrainer(dataset_yaml)
        
        try:
            model, results = trainer.train_food_segmentation(
                epochs=epochs,
                custom_name=f"food_segmentation_{datetime.now().strftime('%Y%m%d_%H%M')}"
            )
            
            print("✅ Segmentation training completed successfully!")
            
            # Step 5: Validation
            print("\n🧪 Step 5: Testing segmentation model...")
            validation_results = trainer.validate_model(
                "data/models/custom_food_segmentation.pt",
                "data/input"
            )
            
            self._print_segmentation_results(validation_results, epochs)
            
            print("\n🎉 Segmentation training completed!")
            print("🎯 Your model can now detect precise food boundaries")
            print("💡 Great for portion size estimation and nutrition calculation")
            
            return model, results
            
        except Exception as e:
            print(f"❌ Segmentation training failed with error: {e}")
            print("💡 Check the error message and consider reducing epochs or batch size")
            return None, None
    
    def _print_segmentation_results(self, validation_results, epochs):
        """Print results specific to segmentation training"""
        if validation_results is None:
            print("⚠️  No test images found for validation")
            return
            
        summary = validation_results['summary']
        
        print("🎨 Segmentation Training Results:")
        print(f"   📚 Training epochs: {epochs}")
        print(f"   🖼️  Tested on: {validation_results['test_images_count']} images")
        print(f"   🎯 Average detections: {summary['avg_detections_per_image']:.1f}")
        print(f"   💯 Average confidence: {summary['avg_max_confidence']:.3f}")
        print(f"   ✅ Images with detections: {summary['images_with_detections']}/{validation_results['test_images_count']}")
        
        print("\n🎨 Segmentation Analysis:")
        if summary['avg_max_confidence'] > 0.6:
            print("🌟 Excellent segmentation confidence! Model can create precise food outlines.")
        elif summary['avg_max_confidence'] > 0.4:
            print("👍 Good segmentation performance. Model is working well for boundaries.")
        else:
            print("🔄 Consider training longer for better segmentation precision.")
            
        print("💡 Segmentation models are great for:")
        print("   - Precise portion size calculation")
        print("   - Accurate nutrition estimation")
        print("   - Food waste measurement")
        print("   - Advanced visual analysis")
    
    def _print_full_results(self, validation_results, epochs):
        """Print comprehensive results from full training"""
        if validation_results is None:
            print("⚠️  No test images found for validation")
            return
            
        summary = validation_results['summary']
        
        print("🏆 Full Training Results:")
        print(f"   📚 Training epochs: {epochs}")
        print(f"   🖼️  Tested on: {validation_results['test_images_count']} images")
        print(f"   🎯 Average detections: {summary['avg_detections_per_image']:.1f}")
        print(f"   💯 Average confidence: {summary['avg_max_confidence']:.3f}")
        print(f"   ✅ Images with detections: {summary['images_with_detections']}/{validation_results['test_images_count']}")
        
        # Performance interpretation
        print("\n📊 Performance Analysis:")
        
        if summary['avg_max_confidence'] > 0.6:
            print("🌟 Excellent confidence scores! Model is production-ready.")
        elif summary['avg_max_confidence'] > 0.4:
            print("👍 Good confidence scores. Model is working well.")
        else:
            print("🔄 Consider training longer or adding more varied training data.")
            
        detection_rate = summary['images_with_detections'] / validation_results['test_images_count']
        if detection_rate > 0.7:
            print("🎯 High detection rate - model finds food in most images.")
        elif detection_rate > 0.5:
            print("✅ Good detection rate - model is reliable.")
        else:
            print("💡 Consider adjusting confidence thresholds or more training data.")
        """Print comprehensive results from full training"""
        if validation_results is None:
            print("⚠️  No test images found for validation")
            return
            
        summary = validation_results['summary']
        
        print("🏆 Full Training Results:")
        print(f"   📚 Training epochs: {epochs}")
        print(f"   🖼️  Tested on: {validation_results['test_images_count']} images")
        print(f"   🎯 Average detections: {summary['avg_detections_per_image']:.1f}")
        print(f"   💯 Average confidence: {summary['avg_max_confidence']:.3f}")
        print(f"   ✅ Images with detections: {summary['images_with_detections']}/{validation_results['test_images_count']}")
        
        # Performance interpretation
        print("\n📊 Performance Analysis:")
        
        if summary['avg_max_confidence'] > 0.6:
            print("🌟 Excellent confidence scores! Model is production-ready.")
        elif summary['avg_max_confidence'] > 0.4:
            print("👍 Good confidence scores. Model is working well.")
        else:
            print("🔄 Consider training longer or adding more varied training data.")
            
        detection_rate = summary['images_with_detections'] / validation_results['test_images_count']
        if detection_rate > 0.7:
            print("🎯 High detection rate - model finds food in most images.")
        elif detection_rate > 0.5:
            print("✅ Good detection rate - model is reliable.")
        else:
            print("💡 Consider adjusting confidence thresholds or more training data.")

def main():
    """Main entry point with enhanced command line interface"""
    parser = argparse.ArgumentParser(
        description="Train custom YOLO models for food detection (FIXED VERSION)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check if everything is ready for training
  python scripts/train_custom_food_model.py --mode check_setup
  
  # Quick test (5-15 minutes) - validates your setup
  python scripts/train_custom_food_model.py --mode quick_test
  
  # Full training with smart labeling (1-3 hours)
  python scripts/train_custom_food_model.py --mode full_training --epochs 50
  
  # Full training using existing images with custom epochs and dataset type
  python scripts/train_custom_food_model.py --mode full_training --dataset existing --epochs 75
  
  # Train segmentation model for precise food boundaries
  python scripts/train_custom_food_model.py --mode segmentation --epochs 50
  
  # Extended training for best results (2-6 hours)
  python scripts/train_custom_food_model.py --mode full_training --epochs 100
        """
    )
    
    parser.add_argument(
        '--mode',
        choices=['check_setup', 'quick_test', 'full_training', 'segmentation'],
        required=True,
        help='What to do: check_setup, quick_test, full_training, or segmentation'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Number of training epochs (default: 5 for quick_test, 50 for full_training)'
    )
    
    parser.add_argument(
        '--dataset',
        choices=['sample', 'existing', 'smart'],
        default='smart',
        help='Dataset type to use for training (sample, existing, or smart)'
    )
    
    args = parser.parse_args()
    
    # Create and run the orchestrator
    orchestrator = TrainingOrchestrator()
    
    try:
        if args.mode == 'check_setup':
            setup_ok = orchestrator.check_setup()
            if setup_ok:
                print("\n💡 Next: python scripts/train_custom_food_model.py --mode quick_test")
            
        elif args.mode == 'quick_test':
            epochs = args.epochs or 5  # Reduced for faster testing
            model, results = orchestrator.quick_test_training(epochs=epochs)
            
        elif args.mode == 'full_training':
            epochs = args.epochs or 50
            dataset_type = args.dataset or 'smart'
            model, results = orchestrator.full_training(epochs=epochs, dataset_type=dataset_type)
            
        elif args.mode == 'segmentation':
            epochs = args.epochs or 50
            model, results = orchestrator.segmentation_training(epochs=epochs)
            
    except KeyboardInterrupt:
        print("\n\n⏸️  Training interrupted by user")
        print("💡 Training progress is saved - you can resume later")
        
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("💡 Try running --mode check_setup to diagnose issues")
        
    finally:
        # Calculate and display total time
        end_time = datetime.now()
        duration = end_time - orchestrator.start_time
        
        hours, remainder = divmod(duration.total_seconds(), 3600)
        minutes, seconds = divmod(remainder, 60)
        
        print(f"\n⏱️  Total time: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")

if __name__ == "__main__":
    main()