#!/usr/bin/env python3
"""
Main GenAI Runner - Dr. Niaki's Strategy
=======================================
"""

import sys
import os
from pathlib import Path

# Add genai_system to path
sys.path.append(str(Path(__file__).parent / "genai_system"))

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='GenAI Food Detection System')
    parser.add_argument('--demo', action='store_true', help='Run CEO demonstration')
    parser.add_argument('--analyze', action='store_true', help='Analyze refrigerator image')
    parser.add_argument('--accuracy-check', action='store_true', help='Run accuracy validation')
    parser.add_argument('--image', type=str, default='data/input/refrigerator.jpg', help='Path to image')
    
    args = parser.parse_args()
    
    try:
        if args.demo:
            from ceo_demo import CEODemo
            demo = CEODemo()
            demo.run_live_demo(args.image)
        
        elif args.analyze:
            from genai_analyzer import GenAIAnalyzer
            analyzer = GenAIAnalyzer()
            results = analyzer.analyze_refrigerator(args.image)
            analyzer.print_simple_summary(results)
            analyzer.save_results(args.image, results)
        
        elif args.accuracy_check:
            from accuracy_calculator import AccuracyCalculator
            calculator = AccuracyCalculator()
            print("Run: python run_genai.py --accuracy-check")
        
        else:
            print("GenAI Food Detection System - Dr. Niaki's Strategy")
            print("Usage:")
            print("  python run_genai.py --demo              # CEO demonstration")
            print("  python run_genai.py --analyze           # Analyze image")
            print("  python run_genai.py --accuracy-check    # Validate accuracy")
    
    except ImportError as e:
        print(f"❌ Module import error: {e}")
        print("💡 Make sure all files are created in genai_system/ directory")
        print("Files needed:")
        print("  - genai_system/genai_analyzer.py")
        print("  - genai_system/accuracy_calculator.py") 
        print("  - genai_system/ceo_demo.py")

if __name__ == "__main__":
    main()