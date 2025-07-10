#!/usr/bin/env python3
"""
Ground Truth Validator - Fix GenAI Inconsistencies
=================================================

This script helps you validate GenAI accuracy and fix the inconsistency issues.

Usage:
python validate_genai_accuracy.py --create-template
python validate_genai_accuracy.py --validate
"""

import json
import sys
from pathlib import Path
from datetime import datetime

# Add genai_system to path
sys.path.append(str(Path(__file__).parent / "genai_system"))

def create_ground_truth_template():
    """Create template for manual validation"""
    print("📋 CREATING GROUND TRUTH TEMPLATE")
    print("=" * 50)
    
    # Create template based on your latest results
    template = {
        "image_name": "refrigerator.jpg",
        "manual_count_date": datetime.now().isoformat(),
        "counted_by": "Manual verification - Your Name",
        "instructions": "Look at data/input/refrigerator.jpg and count each item type manually",
        "ground_truth_inventory": {
            "banana_individual": {
                "quantity": 0,
                "notes": "Count each banana separately - look carefully at clusters",
                "location": "Which shelf/area are they on?"
            },
            "apple_individual": {
                "quantity": 0, 
                "notes": "Count each apple separately - red and green",
                "location": "Which shelf/area are they on?"
            },
            "bottle_individual": {
                "quantity": 0,
                "notes": "Count all bottles - milk, juice, water, etc.",
                "location": "Which shelf/area are they on?"
            },
            "container_individual": {
                "quantity": 0,
                "notes": "Count plastic containers, jars, packages",
                "location": "Which shelf/area are they on?"
            }
        },
        "total_items_manual": 0,
        "verification_notes": "Add any observations about difficult items to count",
        "lighting_quality": "Good/Fair/Poor",
        "image_clarity": "Clear/Slightly blurry/Very blurry"
    }
    
    # Create directory
    ground_truth_dir = Path("data/ground_truth")
    ground_truth_dir.mkdir(parents=True, exist_ok=True)
    
    # Save template
    template_file = ground_truth_dir / "refrigerator_ground_truth.json"
    with open(template_file, 'w', encoding='utf-8') as f:
        json.dump(template, f, indent=2)
    
    print(f"✅ Template created: {template_file}")
    print("\n📝 INSTRUCTIONS:")
    print("1. Open your refrigerator image: data/input/refrigerator.jpg")
    print("2. Count each item type manually and carefully")
    print(f"3. Edit the template file: {template_file}")
    print("4. Fill in the actual quantities you see")
    print("5. Run validation: python validate_genai_accuracy.py --validate")
    
    return template_file

def run_consistency_test():
    """Run GenAI multiple times to measure consistency"""
    print("\n🔄 RUNNING CONSISTENCY TEST")
    print("=" * 50)
    
    try:
        from genai_analyzer import GenAIAnalyzer
        
        analyzer = GenAIAnalyzer()
        image_path = "data/input/refrigerator.jpg"
        
        if not Path(image_path).exists():
            print(f"❌ Refrigerator image not found: {image_path}")
            return None
        
        print("Running GenAI analysis 3 times to measure consistency...")
        
        results = []
        for i in range(3):
            print(f"  Run {i+1}/3...")
            result = analyzer.analyze_refrigerator(image_path)
            results.append(result)
        
        # Analyze consistency
        total_items = [r['total_items'] for r in results]
        
        print(f"\n📊 CONSISTENCY RESULTS:")
        print(f"   Run 1: {total_items[0]} items")
        print(f"   Run 2: {total_items[1]} items") 
        print(f"   Run 3: {total_items[2]} items")
        print(f"   Average: {sum(total_items)/3:.1f} items")
        print(f"   Range: {min(total_items)}-{max(total_items)} items")
        print(f"   Variation: ±{(max(total_items)-min(total_items))/2:.1f} items")
        
        # Check if variation is acceptable
        variation = max(total_items) - min(total_items)
        if variation <= 3:
            print(f"   ✅ Variation acceptable (±{variation/2:.1f} items)")
        elif variation <= 6:
            print(f"   ⚠️ Moderate variation (±{variation/2:.1f} items)")
        else:
            print(f"   ❌ High variation (±{variation/2:.1f} items) - needs improvement")
        
        return results
        
    except ImportError:
        print("❌ GenAI analyzer not found. Make sure genai_system/genai_analyzer.py exists")
        return None

def validate_accuracy():
    """Validate GenAI accuracy against ground truth"""
    print("\n🎯 VALIDATING GENAI ACCURACY")
    print("=" * 50)
    
    # Look for ground truth file
    ground_truth_files = list(Path("data/ground_truth").glob("*.json"))
    if not ground_truth_files:
        print("❌ No ground truth file found.")
        print("   Run: python validate_genai_accuracy.py --create-template")
        print("   Then manually fill it out and run validation again.")
        return False
    
    # Load ground truth
    ground_truth_file = ground_truth_files[0]
    with open(ground_truth_file, 'r', encoding='utf-8') as f:
        ground_truth = json.load(f)
    
    # Check if template was filled out
    total_manual = ground_truth.get('total_items_manual', 0)
    if total_manual == 0:
        print(f"❌ Ground truth template not filled out: {ground_truth_file}")
        print("   Please edit the file and add actual manual counts")
        return False
    
    # Get latest GenAI results
    genai_results_files = list(Path("data/genai_results").glob("*.json"))
    if not genai_results_files:
        print("❌ No GenAI results found. Run GenAI analysis first:")
        print("   python run_genai.py --analyze")
        return False
    
    latest_genai_file = max(genai_results_files, key=lambda x: x.stat().st_mtime)
    with open(latest_genai_file, 'r', encoding='utf-8') as f:
        genai_results = json.load(f)
    
    # Compare results
    print(f"📊 ACCURACY COMPARISON:")
    print(f"   Ground Truth File: {ground_truth_file.name}")
    print(f"   GenAI Results File: {latest_genai_file.name}")
    
    genai_total = genai_results.get('total_items', 0)
    manual_total = ground_truth.get('total_items_manual', 0)
    
    print(f"\n📈 TOTAL ITEMS:")
    print(f"   Manual Count: {manual_total}")
    print(f"   GenAI Count: {genai_total}")
    print(f"   Difference: {abs(genai_total - manual_total)}")
    
    # Item by item comparison
    print(f"\n📋 ITEM-BY-ITEM COMPARISON:")
    genai_inventory = {item['item_type']: item['quantity'] 
                      for item in genai_results.get('inventory', [])}
    ground_truth_inventory = ground_truth.get('ground_truth_inventory', {})
    
    total_accuracy_score = 0
    comparison_count = 0
    
    for item_type, truth_data in ground_truth_inventory.items():
        truth_qty = truth_data.get('quantity', 0)
        genai_qty = genai_inventory.get(item_type, 0)
        
        if truth_qty > 0 or genai_qty > 0:  # Only compare items that exist
            accuracy = min(truth_qty, genai_qty) / max(truth_qty, genai_qty) * 100 if max(truth_qty, genai_qty) > 0 else 0
            total_accuracy_score += accuracy
            comparison_count += 1
            
            status = "✅" if accuracy >= 80 else "⚠️" if accuracy >= 60 else "❌"
            print(f"   {status} {item_type.replace('_', ' ').title()}:")
            print(f"      Manual: {truth_qty} | GenAI: {genai_qty} | Accuracy: {accuracy:.1f}%")
    
    # Overall accuracy
    overall_accuracy = total_accuracy_score / comparison_count if comparison_count > 0 else 0
    print(f"\n🎯 OVERALL ACCURACY: {overall_accuracy:.1f}%")
    
    if overall_accuracy >= 85:
        print("   ✅ Excellent accuracy - ready for data collection phase")
    elif overall_accuracy >= 70:
        print("   ⚠️ Good accuracy - minor improvements possible")
    else:
        print("   ❌ Accuracy needs improvement before proceeding")
    
    return overall_accuracy >= 70

def clean_output_formatting():
    """Fix the messy output formatting issue"""
    print("\n🧹 FIXING OUTPUT FORMATTING")
    print("=" * 50)
    
    # Create cleaned version of genai_analyzer.py
    genai_file = Path("genai_system/genai_analyzer.py")
    if not genai_file.exists():
        print(f"❌ GenAI analyzer not found: {genai_file}")
        return
    
    # Read current file
    with open(genai_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Create backup
    backup_file = genai_file.with_suffix('.py.backup')
    with open(backup_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✅ Created backup: {backup_file}")
    
    # Suggest fixes for output formatting
    print("\n📝 FORMATTING FIXES NEEDED:")
    print("1. Add flush=True to all print statements")
    print("2. Use \\n consistently for line breaks") 
    print("3. Clear console before each major output section")
    print("4. Use single summary print instead of multiple overlapping prints")
    
    print("\n💡 Quick fix - run this command to see cleaner output:")
    print("   python run_genai.py --analyze > genai_output.txt")
    print("   cat genai_output.txt")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='GenAI Accuracy Validator')
    parser.add_argument('--create-template', action='store_true',
                       help='Create ground truth template for manual validation')
    parser.add_argument('--validate', action='store_true',
                       help='Validate GenAI accuracy against ground truth')
    parser.add_argument('--consistency', action='store_true',
                       help='Test GenAI consistency across multiple runs')
    parser.add_argument('--fix-formatting', action='store_true',
                       help='Fix output formatting issues')
    parser.add_argument('--all', action='store_true',
                       help='Run complete validation workflow')
    
    args = parser.parse_args()
    
    print("🎯 GENAI ACCURACY VALIDATOR")
    print("=" * 60)
    
    if args.all:
        # Complete workflow
        print("Running complete validation workflow...")
        create_ground_truth_template()
        print("\n⏸️ PAUSE: Please manually fill out the ground truth template")
        print("Then run: python validate_genai_accuracy.py --validate")
        
    elif args.create_template:
        create_ground_truth_template()
        
    elif args.consistency:
        run_consistency_test()
        
    elif args.fix_formatting:
        clean_output_formatting()
        
    elif args.validate:
        if validate_accuracy():
            print("\n🚀 READY FOR NEXT PHASE!")
            print("GenAI accuracy validated. Proceed to data collection phase.")
        else:
            print("\n🔧 NEEDS IMPROVEMENT")
            print("Fix accuracy issues before proceeding to data collection.")
    
    else:
        print("Usage:")
        print("  python validate_genai_accuracy.py --create-template")
        print("  python validate_genai_accuracy.py --consistency") 
        print("  python validate_genai_accuracy.py --validate")
        print("  python validate_genai_accuracy.py --all")

if __name__ == "__main__":
    main()