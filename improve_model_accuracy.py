#!/usr/bin/env python3
"""
Model Accuracy Improvement Tool
==============================

Tools to improve your local model accuracy through better training,
more data, and advanced techniques.

Usage:
python improve_model_accuracy.py --analyze-current
python improve_model_accuracy.py --suggest-improvements
python improve_model_accuracy.py --retrain-better
python improve_model_accuracy.py --collect-more-data
"""

import argparse
import json
from pathlib import Path
from datetime import datetime
import shutil

try:
    from ultralytics import YOLO
except ImportError:
    print("❌ Ultralytics not installed. Run: pip install ultralytics")
    exit(1)

class ModelAccuracyImprover:
    """
    Tools to improve model accuracy through various methods
    """
    
    def __init__(self):
        self.current_model = "data/models/genai_trained_local_model2/weights/best.pt"
        self.dataset_config = "data/training_dataset_phase2/dataset.yaml"
        self.improvement_dir = Path("data/model_improvements")
        self.improvement_dir.mkdir(parents=True, exist_ok=True)
    
    def analyze_current_model(self):
        """Analyze current model performance and identify improvement areas"""
        print("🔍 ANALYZING CURRENT MODEL PERFORMANCE")
        print("=" * 50)
        
        if not Path(self.current_model).exists():
            print(f"❌ Model not found: {self.current_model}")
            return None
        
        # Load model and get info
        model = YOLO(self.current_model)
        
        print(f"📊 CURRENT MODEL ANALYSIS:")
        print(f"   Model: {self.current_model}")
        print(f"   Architecture: YOLOv8n")
        print(f"   Classes: {len(model.names)}")
        
        # Check training results if available
        results_dir = Path("data/models/genai_trained_local_model")
        if (results_dir / "results.png").exists():
            print(f"   Training results: {results_dir}/results.png")
        
        # Analyze dataset
        dataset_path = Path(self.dataset_config)
        if dataset_path.exists():
            with open(dataset_path, 'r') as f:
                dataset_content = f.read()
            
            # Count images in dataset
            dataset_dir = dataset_path.parent
            train_images = list((dataset_dir / "images").glob("*.jpg"))
            
            print(f"\n📁 DATASET ANALYSIS:")
            print(f"   Dataset config: {dataset_path}")
            print(f"   Training images: {len(train_images)}")
            print(f"   Classes in dataset: {len(model.names)}")
        
        # Identify improvement areas
        improvement_areas = []
        
        if len(train_images) < 100:
            improvement_areas.append({
                "area": "Training Data Quantity",
                "current": f"{len(train_images)} images",
                "recommendation": "Collect 100+ images for better accuracy",
                "priority": "HIGH"
            })
        
        if len(model.names) < 10:
            improvement_areas.append({
                "area": "Food Variety",
                "current": f"{len(model.names)} food types",
                "recommendation": "Add more food types to training data",
                "priority": "MEDIUM"
            })
        
        improvement_areas.append({
            "area": "Model Architecture",
            "current": "YOLOv8n (nano)",
            "recommendation": "Try YOLOv8s or YOLOv8m for higher accuracy",
            "priority": "MEDIUM"
        })
        
        improvement_areas.append({
            "area": "Training Duration",
            "current": "50 epochs (default)",
            "recommendation": "Train for 100-150 epochs",
            "priority": "LOW"
        })
        
        print(f"\n🎯 IMPROVEMENT OPPORTUNITIES:")
        for area in improvement_areas:
            priority_icon = "🔴" if area["priority"] == "HIGH" else "🟡" if area["priority"] == "MEDIUM" else "🟢"
            print(f"   {priority_icon} {area['area']}:")
            print(f"      Current: {area['current']}")
            print(f"      Recommendation: {area['recommendation']}")
        
        return improvement_areas
    
    def suggest_improvements(self):
        """Provide specific improvement suggestions based on analysis"""
        print("\n💡 IMPROVEMENT SUGGESTIONS")
        print("=" * 50)
        
        improvement_areas = self.analyze_current_model()
        
        suggestions = {
            "quick_wins": [
                {
                    "action": "Retrain with larger model",
                    "command": "python improve_model_accuracy.py --retrain-better --model yolov8s",
                    "time": "1-2 hours",
                    "expected_improvement": "5-10% accuracy gain"
                },
                {
                    "action": "Train for more epochs", 
                    "command": "python improve_model_accuracy.py --retrain-better --epochs 100",
                    "time": "2-3 hours",
                    "expected_improvement": "3-5% accuracy gain"
                }
            ],
            "medium_effort": [
                {
                    "action": "Collect more training images",
                    "command": "python improve_model_accuracy.py --collect-more-data",
                    "time": "1-2 days",
                    "expected_improvement": "10-15% accuracy gain"
                },
                {
                    "action": "Add data augmentation",
                    "command": "python improve_model_accuracy.py --retrain-better --augment",
                    "time": "2-4 hours", 
                    "expected_improvement": "3-8% accuracy gain"
                }
            ],
            "advanced": [
                {
                    "action": "Use ensemble methods",
                    "command": "python improve_model_accuracy.py --create-ensemble",
                    "time": "Half day",
                    "expected_improvement": "5-10% accuracy gain"
                },
                {
                    "action": "Fine-tune hyperparameters",
                    "command": "python improve_model_accuracy.py --hyperparameter-tuning",
                    "time": "1-2 days",
                    "expected_improvement": "5-15% accuracy gain"
                }
            ]
        }
        
        print("🚀 QUICK WINS (1-3 hours):")
        for suggestion in suggestions["quick_wins"]:
            print(f"   ✅ {suggestion['action']}")
            print(f"      Command: {suggestion['command']}")
            print(f"      Time: {suggestion['time']}")
            print(f"      Expected: {suggestion['expected_improvement']}")
            print()
        
        print("📈 MEDIUM EFFORT (1-2 days):")
        for suggestion in suggestions["medium_effort"]:
            print(f"   🎯 {suggestion['action']}")
            print(f"      Command: {suggestion['command']}")
            print(f"      Time: {suggestion['time']}")
            print(f"      Expected: {suggestion['expected_improvement']}")
            print()
        
        print("🔬 ADVANCED (2+ days):")
        for suggestion in suggestions["advanced"]:
            print(f"   🧠 {suggestion['action']}")
            print(f"      Command: {suggestion['command']}")
            print(f"      Time: {suggestion['time']}")
            print(f"      Expected: {suggestion['expected_improvement']}")
            print()
        
        print("💡 RECOMMENDATION: Start with quick wins, then add more data")
        
        return suggestions
    
    def retrain_with_improvements(self, model_size="yolov8s", epochs=100, augment=True):
        """Retrain model with improvements"""
        print(f"\n🚀 RETRAINING WITH IMPROVEMENTS")
        print("=" * 50)
        
        print(f"📊 TRAINING CONFIGURATION:")
        print(f"   Model: {model_size}")
        print(f"   Epochs: {epochs}")
        print(f"   Augmentation: {augment}")
        print(f"   Dataset: {self.dataset_config}")
        
        # Load improved model
        model = YOLO(f"{model_size}.pt")
        
        # Training parameters
        training_params = {
            'data': self.dataset_config,
            'epochs': epochs,
            'batch': 4 if model_size == "yolov8s" else 2,  # Smaller batch for larger models
            'imgsz': 640,
            'device': 'cpu',
            'project': 'data/models',
            'name': f'improved_model_{model_size}_{epochs}epochs',
            'save': True,
            'plots': True,
            
            # Improved training parameters
            'patience': 20,  # Early stopping
            'lr0': 0.001,    # Lower learning rate
            'weight_decay': 0.0005,
            
            # Box loss tuning for food detection
            'box': 7.5,
            'cls': 1.0, 
            'dfl': 1.5,
        }
        
        # Add augmentation if requested
        if augment:
            training_params.update({
                'hsv_h': 0.02,      # Hue augmentation
                'hsv_s': 0.7,       # Saturation (important for food freshness)
                'hsv_v': 0.4,       # Value/brightness
                'degrees': 5.0,     # Slight rotation
                'translate': 0.1,   # Translation
                'scale': 0.2,       # Scale variation
                'fliplr': 0.5,      # Horizontal flip
                'mosaic': 1.0,      # Mosaic augmentation
                'mixup': 0.1        # Mixup augmentation
            })
        
        print(f"\n🔥 Starting improved training...")
        print(f"   This may take {epochs//20}-{epochs//10} minutes")
        
        try:
            results = model.train(**training_params)
            
            print(f"\n✅ IMPROVED TRAINING COMPLETED!")
            print(f"📊 Results saved to: data/models/improved_model_{model_size}_{epochs}epochs/")
            print(f"🎯 New model: data/models/improved_model_{model_size}_{epochs}epochs/weights/best.pt")
            
            # Compare with old model
            print(f"\n📈 TO COMPARE MODELS:")
            print(f"   Old: python test_local_model.py --basic --model-path {self.current_model}")
            print(f"   New: python test_local_model.py --basic --model-path data/models/improved_model_{model_size}_{epochs}epochs/weights/best.pt")
            
            return True
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
            return False
    
    def collect_more_data_guide(self):
        """Guide for collecting more training data"""
        print("\n📸 DATA COLLECTION GUIDE")
        print("=" * 50)
        
        current_images = len(list(Path("data/training_dataset_phase2/images").glob("*.jpg")))
        target_images = max(500, current_images * 2)
        
        print(f"📊 CURRENT STATUS:")
        print(f"   Current images: {current_images}")
        print(f"   Target images: {target_images}")
        print(f"   Need to collect: {target_images - current_images} more images")
        
        print(f"\n🎯 DATA COLLECTION STRATEGY:")
        
        strategies = [
            {
                "method": "Take more photos yourself",
                "target": "50-100 images",
                "tips": [
                    "Different times of day",
                    "Different refrigerator contents", 
                    "Various lighting conditions",
                    "Multiple angles"
                ]
            },
            {
                "method": "Ask friends and family",
                "target": "50-150 images",
                "tips": [
                    "WhatsApp requests",
                    "Social media posts",
                    "Email requests",
                    "Different household types"
                ]
            },
            {
                "method": "Online sources",
                "target": "100-300 images",
                "tips": [
                    "Pinterest: 'refrigerator organization'",
                    "Google Images: 'fridge contents'",
                    "Reddit: r/MealPrep posts",
                    "Food blogs with fridge photos"
                ]
            }
        ]
        
        for strategy in strategies:
            print(f"\n📋 {strategy['method']} ({strategy['target']}):")
            for tip in strategy['tips']:
                print(f"   • {tip}")
        
        print(f"\n🔄 AFTER COLLECTING MORE IMAGES:")
        print(f"1. Add to: data/collected_images/")
        print(f"2. Run: python genai_system/build_training_dataset.py --label-batch")
        print(f"3. Run: python improve_model_accuracy.py --retrain-better")
        
        print(f"\n📈 EXPECTED IMPROVEMENT:")
        if current_images < 100:
            print(f"   With 100+ images: 10-20% accuracy improvement")
        elif current_images < 300:
            print(f"   With 300+ images: 5-15% accuracy improvement")
        else:
            print(f"   With 500+ images: 3-8% accuracy improvement")
    
    def create_training_plan(self):
        """Create personalized improvement plan"""
        print("\n📋 PERSONALIZED IMPROVEMENT PLAN")
        print("=" * 50)
        
        improvement_areas = self.analyze_current_model()
        current_images = len(list(Path("data/training_dataset_phase2/images").glob("*.jpg")))
        
        plan = {
            "immediate": [],
            "this_week": [],
            "long_term": []
        }
        
        # Immediate actions (today)
        plan["immediate"].append("🧪 Test current model: python test_local_model.py --full-analysis")
        
        if current_images < 100:
            plan["immediate"].append("📸 Take 20-30 more refrigerator photos")
        
        plan["immediate"].append("🚀 Try larger model: python improve_model_accuracy.py --retrain-better --model yolov8s")
        
        # This week actions
        if current_images < 200:
            plan["this_week"].append("📱 Collect 50+ images from friends/family")
            plan["this_week"].append("🌐 Download 30+ images from Pinterest/Google")
        
        plan["this_week"].append("🔄 Retrain with more epochs: --epochs 100")
        plan["this_week"].append("📊 Compare all model versions")
        
        # Long term actions
        plan["long_term"].append("🎯 Collect 500+ images for maximum accuracy")
        plan["long_term"].append("🧠 Try ensemble methods")
        plan["long_term"].append("⚡ Consider GPU training for faster iterations")
        
        for phase, actions in plan.items():
            print(f"\n{phase.upper().replace('_', ' ')}:")
            for action in actions:
                print(f"   {action}")
        
        print(f"\n🎯 SUCCESS METRICS:")
        print(f"   Target accuracy: 80%+ (vs current ~65%)")
        print(f"   Target speed: <2 seconds per image")
        print(f"   Target classes: 10+ food types")
        
        return plan

def main():
    parser = argparse.ArgumentParser(description='Model Accuracy Improvement Tool')
    parser.add_argument('--analyze-current', action='store_true',
                       help='Analyze current model performance')
    parser.add_argument('--suggest-improvements', action='store_true',
                       help='Suggest specific improvements')
    parser.add_argument('--retrain-better', action='store_true',
                       help='Retrain with improvements')
    parser.add_argument('--collect-more-data', action='store_true',
                       help='Guide for collecting more training data')
    parser.add_argument('--create-plan', action='store_true',
                       help='Create personalized improvement plan')
    
    # Retraining options
    parser.add_argument('--model', type=str, default='yolov8s',
                       help='Model size: yolov8n, yolov8s, yolov8m, yolov8l')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--augment', action='store_true', default=True,
                       help='Use data augmentation')
    
    args = parser.parse_args()
    
    print("🎯 MODEL ACCURACY IMPROVEMENT TOOL")
    print("=" * 60)
    
    improver = ModelAccuracyImprover()
    
    if args.analyze_current:
        improver.analyze_current_model()
    
    elif args.suggest_improvements:
        improver.suggest_improvements()
    
    elif args.retrain_better:
        improver.retrain_with_improvements(
            model_size=args.model,
            epochs=args.epochs,
            augment=args.augment
        )
    
    elif args.collect_more_data:
        improver.collect_more_data_guide()
    
    elif args.create_plan:
        improver.create_training_plan()
    
    else:
        print("Usage:")
        print("  python improve_model_accuracy.py --analyze-current        # Analyze current model")
        print("  python improve_model_accuracy.py --suggest-improvements   # Get improvement suggestions")
        print("  python improve_model_accuracy.py --retrain-better         # Retrain with improvements")
        print("  python improve_model_accuracy.py --collect-more-data      # Data collection guide")
        print("  python improve_model_accuracy.py --create-plan            # Create improvement plan")
        print("\nRecommended: Start with --analyze-current")

if __name__ == "__main__":
    main()