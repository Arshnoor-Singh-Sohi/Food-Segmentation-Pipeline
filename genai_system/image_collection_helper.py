#!/usr/bin/env python3
"""
Automated Image Collection Helper
================================

Helps collect refrigerator images from various sources efficiently.
Downloads from public datasets and organizes everything properly.

Usage:
python genai_system/image_collection_helper.py --setup
python genai_system/image_collection_helper.py --download-samples
python genai_system/image_collection_helper.py --organize-existing
"""

import requests
import json
from pathlib import Path
import shutil
import zipfile
import os
from datetime import datetime

class ImageCollectionHelper:
    """
    Automated helper for collecting refrigerator training images
    """
    
    def __init__(self):
        self.base_dir = Path("data/collected_images")
        self.datasets_dir = self.base_dir / "downloaded_datasets"
        self.processed_dir = self.base_dir / "processed_for_training"
        
        # Create directories
        for directory in [self.base_dir, self.datasets_dir, self.processed_dir]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def setup_collection_structure(self):
        """Create organized collection structure"""
        print("📁 SETTING UP IMAGE COLLECTION STRUCTURE")
        print("=" * 50)
        
        collection_folders = {
            "kaggle_datasets": "Downloaded datasets from Kaggle",
            "online_images": "Individual images from Google/Pinterest", 
            "own_photos": "Your own refrigerator photos",
            "friends_family": "Photos from friends and family",
            "stock_photos": "Professional stock photos",
            "processed_for_training": "Processed and ready for GenAI labeling"
        }
        
        for folder, description in collection_folders.items():
            folder_path = self.base_dir / folder
            folder_path.mkdir(parents=True, exist_ok=True)
            
            # Create README
            readme_content = f"""# {folder.replace('_', ' ').title()}

{description}

## Guidelines:
- Add refrigerator/fridge interior images
- Focus on visible individual food items
- Good lighting preferred
- Any image format: .jpg, .jpeg, .png
- Filename format: {folder}_001.jpg, {folder}_002.jpg, etc.

## Target: 20-50 images in this folder

## Added: {datetime.now().strftime('%Y-%m-%d')}
"""
            readme_file = folder_path / "README.md"
            with open(readme_file, 'w') as f:
                f.write(readme_content)
        
        print("✅ Collection structure created:")
        for folder in collection_folders.keys():
            print(f"   📁 data/collected_images/{folder}/")
        
        return list(collection_folders.keys())
    
    def suggest_kaggle_datasets(self):
        """Suggest specific Kaggle datasets for refrigerator images"""
        print("\n🎯 KAGGLE DATASETS RECOMMENDATIONS")
        print("=" * 50)
        
        datasets = [
            {
                "name": "Food Images Dataset",
                "url": "https://www.kaggle.com/datasets/kmader/food41",
                "description": "Food images including refrigerator contexts",
                "estimated_images": "1000+",
                "pros": "High quality, diverse foods"
            },
            {
                "name": "Refrigerator Food Detection Dataset", 
                "url": "https://www.kaggle.com/datasets/search?q=refrigerator+food",
                "description": "Specific refrigerator interior images",
                "estimated_images": "500+", 
                "pros": "Directly relevant to our use case"
            },
            {
                "name": "Kitchen Items and Food Dataset",
                "url": "https://www.kaggle.com/datasets/search?q=kitchen+food+items",
                "description": "Kitchen and food storage images",
                "estimated_images": "800+",
                "pros": "Good variety of storage contexts"
            }
        ]
        
        print("📋 RECOMMENDED DATASETS:")
        for i, dataset in enumerate(datasets, 1):
            print(f"\n{i}. {dataset['name']}")
            print(f"   URL: {dataset['url']}")
            print(f"   Description: {dataset['description']}")
            print(f"   Images: ~{dataset['estimated_images']}")
            print(f"   Pros: {dataset['pros']}")
        
        print(f"\n💡 DOWNLOAD INSTRUCTIONS:")
        print("1. Go to kaggle.com and create free account")
        print("2. Search for datasets above")
        print("3. Download ZIP files")
        print("4. Extract to: data/collected_images/kaggle_datasets/")
        print("5. Run: python genai_system/image_collection_helper.py --organize-existing")
        
        return datasets
    
    def suggest_online_sources(self):
        """Suggest online sources for additional images"""
        print("\n🌐 ONLINE IMAGE SOURCES")
        print("=" * 50)
        
        sources = [
            {
                "source": "Pinterest",
                "search_terms": ["refrigerator organization", "fridge inventory", "meal prep fridge"],
                "estimated_images": "50-100",
                "method": "Manual download from search results",
                "pros": "Real-world variety, good organization"
            },
            {
                "source": "Google Images", 
                "search_terms": ["refrigerator contents", "fridge inside", "food storage"],
                "estimated_images": "30-50",
                "method": "Right-click save from search results",
                "pros": "Easy access, diverse sources"
            },
            {
                "source": "Unsplash",
                "search_terms": ["refrigerator", "food storage", "kitchen"],
                "estimated_images": "20-30", 
                "method": "Download from unsplash.com",
                "pros": "High quality, free license"
            },
            {
                "source": "Reddit r/MealPrep",
                "search_terms": ["fridge", "meal prep sunday", "food storage"],
                "estimated_images": "20-40",
                "method": "Save images from posts",
                "pros": "Real user photos, realistic contents"
            }
        ]
        
        print("📋 ONLINE SOURCES:")
        for source in sources:
            print(f"\n🔸 {source['source']}:")
            print(f"   Search terms: {', '.join(source['search_terms'])}")
            print(f"   Expected images: {source['estimated_images']}")
            print(f"   Method: {source['method']}")
            print(f"   Pros: {source['pros']}")
        
        return sources
    
    def organize_existing_images(self):
        """Organize any existing images into proper structure"""
        print("\n📂 ORGANIZING EXISTING IMAGES")
        print("=" * 50)
        
        # Look for images in various locations
        search_locations = [
            self.base_dir,
            Path("data/input"),
            Path("Downloads"),  # Common download location
            Path.home() / "Downloads",
            Path.home() / "Pictures"
        ]
        
        found_images = []
        
        for location in search_locations:
            if location.exists():
                for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
                    found_images.extend(location.glob(ext))
        
        if not found_images:
            print("❌ No images found in common locations")
            print("💡 Add images to data/collected_images/ folders manually")
            return False
        
        print(f"📊 Found {len(found_images)} images")
        
        # Organize by filename patterns
        organized_count = 0
        
        for image_file in found_images:
            filename_lower = image_file.name.lower()
            
            # Categorize by filename
            if any(word in filename_lower for word in ['fridge', 'refrigerator', 'refrig']):
                target_folder = self.base_dir / "own_photos"
            elif any(word in filename_lower for word in ['kitchen', 'food', 'meal']):
                target_folder = self.base_dir / "online_images"
            elif any(word in filename_lower for word in ['stock', 'unsplash', 'pexels']):
                target_folder = self.base_dir / "stock_photos"
            else:
                target_folder = self.base_dir / "online_images"  # Default
            
            # Copy image to organized location
            target_file = target_folder / f"organized_{organized_count:03d}_{image_file.name}"
            
            try:
                shutil.copy2(image_file, target_file)
                organized_count += 1
                print(f"   ✅ Organized: {image_file.name} → {target_folder.name}/")
            except Exception as e:
                print(f"   ⚠️ Error organizing {image_file.name}: {e}")
        
        print(f"\n📊 ORGANIZATION RESULTS:")
        print(f"   Images organized: {organized_count}")
        print(f"   Ready for GenAI labeling!")
        
        return organized_count > 0
    
    def validate_collection(self):
        """Validate collected images for training readiness"""
        print("\n✅ VALIDATING IMAGE COLLECTION")
        print("=" * 50)
        
        total_images = 0
        collection_summary = {}
        
        # Count images in each folder
        for folder in self.base_dir.iterdir():
            if folder.is_dir() and folder.name != "processed_for_training":
                image_count = len(list(folder.glob('*.jpg')) + list(folder.glob('*.jpeg')) + list(folder.glob('*.png')))
                collection_summary[folder.name] = image_count
                total_images += image_count
        
        print(f"📊 COLLECTION SUMMARY:")
        for folder, count in collection_summary.items():
            status = "✅" if count >= 10 else "⚠️" if count >= 5 else "❌"
            print(f"   {status} {folder}: {count} images")
        
        print(f"\n📈 TOTAL IMAGES: {total_images}")
        
        # Provide recommendations
        if total_images >= 100:
            print(f"🎉 EXCELLENT! Ready for GenAI labeling")
            recommendation = "proceed"
        elif total_images >= 50:
            print(f"✅ GOOD! Can start GenAI labeling, add more for better results")
            recommendation = "proceed_with_caution"
        elif total_images >= 20:
            print(f"⚠️ MINIMAL! Add more images for better training results")
            recommendation = "need_more"
        else:
            print(f"❌ INSUFFICIENT! Need at least 20 images to start")
            recommendation = "insufficient"
        
        return {
            "total_images": total_images,
            "collection_summary": collection_summary,
            "recommendation": recommendation
        }
    
    def create_collection_plan(self):
        """Create personalized collection plan"""
        print("\n📋 PERSONALIZED COLLECTION PLAN")
        print("=" * 50)
        
        plan = {
            "immediate_actions": [
                "✅ Set up folder structure (DONE)",
                "📥 Download 1-2 Kaggle datasets (20 minutes)",
                "📸 Take 10-15 photos of your own refrigerator (10 minutes)",
                "🌐 Collect 20-30 images from Pinterest/Google (30 minutes)"
            ],
            "this_week": [
                "👥 Ask friends/family for refrigerator photos",
                "📱 Take refrigerator photos at different times",
                "🔍 Find additional online datasets",
                "📂 Organize all collected images"
            ],
            "total_time_estimate": "1-2 hours to get 100+ images",
            "minimum_viable": "20 images (can start GenAI labeling)",
            "optimal_target": "100+ images (excellent training dataset)"
        }
        
        for phase, actions in plan.items():
            if phase in ["immediate_actions", "this_week"]:
                print(f"\n🎯 {phase.replace('_', ' ').title()}:")
                for action in actions:
                    print(f"   {action}")
            else:
                print(f"\n💡 {phase.replace('_', ' ').title()}: {actions}")
        
        return plan

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Image Collection Helper')
    parser.add_argument('--setup', action='store_true',
                       help='Set up collection structure and show recommendations')
    parser.add_argument('--download-suggestions', action='store_true',
                       help='Show dataset download suggestions')
    parser.add_argument('--organize-existing', action='store_true',
                       help='Organize existing images into proper structure')
    parser.add_argument('--validate', action='store_true',
                       help='Validate current image collection')
    parser.add_argument('--plan', action='store_true',
                       help='Create personalized collection plan')
    
    args = parser.parse_args()
    
    print("📸 IMAGE COLLECTION HELPER")
    print("=" * 60)
    
    helper = ImageCollectionHelper()
    
    if args.setup:
        helper.setup_collection_structure()
        helper.suggest_kaggle_datasets()
        helper.suggest_online_sources()
        helper.create_collection_plan()
        
    elif args.download_suggestions:
        helper.suggest_kaggle_datasets()
        helper.suggest_online_sources()
        
    elif args.organize_existing:
        helper.organize_existing_images()
        
    elif args.validate:
        result = helper.validate_collection()
        if result["recommendation"] == "proceed":
            print("\n🚀 READY FOR NEXT STEP:")
            print("   python genai_system/build_training_dataset.py --label-batch")
        
    elif args.plan:
        helper.create_collection_plan()
        
    else:
        print("Usage:")
        print("  python genai_system/image_collection_helper.py --setup")
        print("  python genai_system/image_collection_helper.py --organize-existing")
        print("  python genai_system/image_collection_helper.py --validate")

if __name__ == "__main__":
    main()