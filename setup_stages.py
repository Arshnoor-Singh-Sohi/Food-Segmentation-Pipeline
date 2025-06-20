"""
Stage Setup Script
==================

Run this script to create the organized stages structure and install Stage 1A files.

Usage:
python setup_stages.py

This will:
1. Create the organized directory structure
2. Save the Stage 1A files in the correct locations
3. Provide clear instructions on how to run Stage 1A
"""

from pathlib import Path
import os

# Get the artifacts content (you'll need to copy these from the artifacts above)
STAGE_RUNNER_CONTENT = '''# Copy the content from run_stage.py artifact here'''

STAGE1A_RUNNER_CONTENT = '''# Copy the content from 1a_runner.py artifact here'''

STAGE1A_DETECTION_FIXER_CONTENT = '''# Copy the content from detection_fixer.py artifact here'''

def create_directory_structure():
    """Create the organized directory structure"""
    print("📁 Creating Stage-Based Directory Structure...")
    
    # Define all directories to create
    directories = [
        # Stages directories
        "stages/stage1a_detection_fixes",
        "stages/stage1b_display_fixes", 
        "stages/stage1c_enhanced_detection",
        "stages/stage2a_bottle_detection",
        "stages/stage2b_bottle_ocr",
        "stages/stage2c_bottle_classification",
        "stages/stage3a_receipt_ocr",
        "stages/stage3b_package_ocr",
        
        # Data directories
        "data/input",
        "data/models",
        
        # Output directories by stage
        "data/output/stage1a_results",
        "data/output/stage1b_results",
        "data/output/stage1c_results",
        "data/output/stage2a_results",
        "data/output/stage2b_results", 
        "data/output/stage2c_results",
        "data/output/stage3a_results",
        "data/output/stage3b_results",
        
        # Config directory
        "config"
    ]
    
    # Create all directories
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"   ✅ Created: {directory}")
    
    print("✅ Directory structure created successfully!")

def create_stage1a_config():
    """Create Stage 1A configuration file"""
    config_content = """# Stage 1A Configuration
# Detection Accuracy Fixes

# Enhanced confidence thresholds to reduce false positives
confidence_thresholds:
  bottle: 0.65      # Higher threshold - CEO feedback: too many bottles
  banana: 0.35      # Lower threshold - better individual banana detection  
  apple: 0.4
  food: 0.3
  pizza: 0.5
  default: 0.25

# Bottle validation parameters (main Stage 1A issue)
bottle_validation:
  min_area_pixels: 800
  max_area_pixels: 40000
  min_aspect_ratio: 0.15
  max_aspect_ratio: 1.0
  min_confidence: 0.65

# Stage 1A objectives
objectives:
  - Fix false positive bottle detection
  - Fix banana quantity counting (no more "whole on 3")
  - Improve item classification accuracy  
  - Fix portion vs complete dish display

# Success criteria
success_criteria:
  bottle_false_positives_reduced: true
  banana_counting_improved: true
  food_classification_working: true
  display_formatting_clear: true
"""
    
    config_file = Path("stages/stage1a_detection_fixes/config.yaml")
    with open(config_file, 'w') as f:
        f.write(config_content)
    
    print(f"✅ Created Stage 1A config: {config_file}")

def create_stage1a_readme():
    """Create Stage 1A README file"""
    readme_content = """# Stage 1A: Detection Accuracy Fixes

## Objectives
This stage addresses the immediate detection issues identified:

1. **False positive bottle detection** - CEO feedback: "too many bottles count"
2. **Banana quantity counting errors** - Showing "whole on 3" instead of individual count
3. **Item classification accuracy** - General improvements
4. **Portion vs Complete Dish display** - Clear classification display

## Files in This Stage

- `detection_fixer.py` - Main detection improvement system
- `1a_runner.py` - Stage-specific runner  
- `config.yaml` - Stage 1A configuration
- `README.md` - This documentation

## How to Run Stage 1A

### From Project Root (Recommended):
```bash
# Test refrigerator scenario (main focus)
python run_stage.py 1a --refrigerator

# Test specific image
python run_stage.py 1a --image data/input/refrigerator.jpg

# Run all Stage 1A tests
python run_stage.py 1a --test-all
```

### Direct Execution (Alternative):
```bash
# From project root
cd stages/stage1a_detection_fixes
python 1a_runner.py --refrigerator
```

## Expected Outputs

- **Visualization**: Before/after comparison showing raw vs enhanced detections
- **JSON Report**: Detailed analysis with item counts and classifications
- **Console Summary**: Clear feedback on fixes applied

## Success Criteria

✅ Bottle false positives reduced (fewer bottles detected)
✅ Banana counting shows individual bananas correctly
✅ Food type classification displays clearly (COMPLETE DISH vs INDIVIDUAL ITEMS)
✅ Overall detection accuracy improved

## Files Created

Results are saved in: `data/output/stage1a_results/`

- `{image_name}_stage1a_{timestamp}.json` - Detailed analysis
- `{image_name}_stage1a_fixes_{timestamp}.png` - Visual comparison

## Next Stage

Once Stage 1A is working satisfactorily, proceed to:
**Stage 1B: Display Formatting Fixes**
"""
    
    readme_file = Path("stages/stage1a_detection_fixes/README.md")
    with open(readme_file, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print(f"✅ Created Stage 1A README: {readme_file}")

def show_instructions():
    """Show clear instructions for what to do next"""
    print("\n" + "="*60)
    print("🎉 STAGE SETUP COMPLETE!")
    print("="*60)
    
    print("\n📂 Directory Structure Created:")
    print("   stages/stage1a_detection_fixes/  # Current stage files")
    print("   data/input/                      # Place test images here")
    print("   data/output/stage1a_results/     # Results will be saved here")
    
    # print("\n📋 Next Steps:")
    # print("1. Copy the 3 artifacts content into the stage files:")
    # print("   - Copy run_stage.py content → run_stage.py")
    # print("   - Copy 1a_runner.py content → stages/stage1a_detection_fixes/1a_runner.py")
    # print("   - Copy detection_fixer.py content → stages/stage1a_detection_fixes/detection_fixer.py")
    
    # print("\n2. Add a refrigerator test image:")
    # print("   - Save your refrigerator image as: data/input/refrigerator.jpg")
    
    # print("\n3. Run Stage 1A tests:")
    # print("   python run_stage.py 1a --refrigerator")
    
    # print("\n🎯 Stage 1A will fix:")
    # print("   ✅ False positive bottle detection")
    # print("   ✅ Banana quantity counting errors")
    # print("   ✅ Item classification accuracy")
    # print("   ✅ Portion vs Complete Dish display")
    
    # print("\n📊 You'll get:")
    # print("   - Visual before/after comparisons")
    # print("   - Detailed JSON analysis reports") 
    # print("   - Clear success/failure feedback")
    
    print("\n⚠️ Remember: Complete Stage 1A before moving to Stage 1B!")

def main():
    print("🚀 Setting Up Staged Food Segmentation Pipeline")
    print("=" * 50)
    
    # Create directory structure
    create_directory_structure()
    
    # Create Stage 1A specific files
    create_stage1a_config()
    create_stage1a_readme()
    
    # Show instructions
    show_instructions()

if __name__ == "__main__":
    main()