#!/usr/bin/env python3
"""
Compare results from custom model vs default YOLOv8 model
Provides detailed analysis and visualizations
"""

import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from datetime import datetime
import pandas as pd
from typing import Dict, List, Tuple

class ModelResultsComparator:
    """Compare results from different models"""
    
    def __init__(self, custom_json_path: str, default_json_path: str):
        self.custom_path = Path(custom_json_path)
        self.default_path = Path(default_json_path)
        
        # Load JSON data
        with open(self.custom_path, 'r') as f:
            self.custom_data = json.load(f)
        
        with open(self.default_path, 'r') as f:
            self.default_data = json.load(f)
        
        self.comparison_results = {}
        
    def analyze_all_metrics(self) -> Dict:
        """Perform comprehensive comparison analysis"""
        print("🔍 Analyzing Model Performance...")
        print("="*60)
        
        # 1. Detection & Localization Analysis
        detection_metrics = self.analyze_detection_metrics()
        
        # 2. Classification Analysis
        classification_metrics = self.analyze_classification_metrics()
        
        # 3. Portion Estimation Analysis
        portion_metrics = self.analyze_portion_metrics()
        
        # 4. Nutrition Analysis
        nutrition_metrics = self.analyze_nutrition_metrics()
        
        # 5. Overall Performance Score
        overall_score = self.calculate_overall_score(
            detection_metrics, classification_metrics, 
            portion_metrics, nutrition_metrics
        )
        
        self.comparison_results = {
            'detection': detection_metrics,
            'classification': classification_metrics,
            'portion': portion_metrics,
            'nutrition': nutrition_metrics,
            'overall': overall_score
        }
        
        return self.comparison_results
    
    def analyze_detection_metrics(self) -> Dict:
        """Analyze detection and localization performance"""
        print("\n📍 Detection & Localization Analysis:")
        
        custom_items = self.custom_data.get('enriched_items', [])
        default_items = self.default_data.get('enriched_items', [])
        
        metrics = {
            'custom': {
                'num_detections': len(custom_items),
                'avg_confidence': 0,
                'bbox_coverage': [],
                'has_masks': False
            },
            'default': {
                'num_detections': len(default_items),
                'avg_confidence': 0,
                'bbox_coverage': [],
                'has_masks': False
            }
        }
        
        # Analyze custom model
        if custom_items:
            confidences = [item.get('confidence', 0) for item in custom_items]
            metrics['custom']['avg_confidence'] = np.mean(confidences)
            
            for item in custom_items:
                bbox = item.get('bbox', {})
                coverage = self.calculate_bbox_coverage(bbox)
                metrics['custom']['bbox_coverage'].append(coverage)
                
                if item.get('has_mask', False):
                    metrics['custom']['has_masks'] = True
        
        # Analyze default model
        if default_items:
            confidences = [item.get('confidence', 0) for item in default_items]
            metrics['default']['avg_confidence'] = np.mean(confidences)
            
            for item in default_items:
                bbox = item.get('bbox', {})
                coverage = self.calculate_bbox_coverage(bbox)
                metrics['default']['bbox_coverage'].append(coverage)
                
                if item.get('mask_info'):
                    metrics['default']['has_masks'] = True
                    metrics['default']['mask_coverage'] = item['mask_info'].get('area_percentage', 0)
        
        # Print comparison
        print(f"  Custom Model: {metrics['custom']['num_detections']} detections, "
              f"avg confidence: {metrics['custom']['avg_confidence']:.3f}")
        print(f"  Default Model: {metrics['default']['num_detections']} detections, "
              f"avg confidence: {metrics['default']['avg_confidence']:.3f}")
        
        return metrics
    
    def analyze_classification_metrics(self) -> Dict:
        """Analyze food classification accuracy"""
        print("\n🏷️ Classification Analysis:")
        
        metrics = {
            'custom': {'classifications': [], 'cuisines': [], 'confidence': []},
            'default': {'classifications': [], 'cuisines': [], 'confidence': []}
        }
        
        # Custom model classifications
        for item in self.custom_data.get('enriched_items', []):
            metrics['custom']['classifications'].append(
                item.get('detailed_food_type', 'unknown')
            )
            metrics['custom']['cuisines'].append(
                item.get('cuisine', 'unknown')
            )
            metrics['custom']['confidence'].append(
                item.get('classification_confidence', 0)
            )
        
        # Default model classifications
        for item in self.default_data.get('enriched_items', []):
            metrics['default']['classifications'].append(
                item.get('detailed_food_type', 'unknown')
            )
            metrics['default']['cuisines'].append(
                item.get('cuisine', 'unknown')
            )
            metrics['default']['confidence'].append(
                item.get('classification_confidence', 0)
            )
        
        print(f"  Custom: {metrics['custom']['classifications']}")
        print(f"  Default: {metrics['default']['classifications']}")
        
        return metrics
    
    def analyze_portion_metrics(self) -> Dict:
        """Analyze portion estimation accuracy"""
        print("\n🍕 Portion Estimation Analysis:")
        
        metrics = {'custom': {}, 'default': {}}
        
        # Custom model portions
        custom_portions = []
        for item in self.custom_data.get('enriched_items', []):
            portion = item.get('portion', {})
            custom_portions.append({
                'weight': portion.get('estimated_weight_g', 0),
                'description': portion.get('serving_description', ''),
                'confidence': portion.get('confidence', 'low')
            })
        
        # Default model portions
        default_portions = []
        for item in self.default_data.get('enriched_items', []):
            portion = item.get('portion', {})
            default_portions.append({
                'weight': portion.get('estimated_weight_g', 0),
                'description': portion.get('serving_description', ''),
                'confidence': portion.get('confidence', 'low')
            })
        
        metrics['custom']['portions'] = custom_portions
        metrics['default']['portions'] = default_portions
        
        # Calculate average weights
        if custom_portions:
            avg_weight_custom = np.mean([p['weight'] for p in custom_portions])
            print(f"  Custom avg weight: {avg_weight_custom:.1f}g")
        
        if default_portions:
            avg_weight_default = np.mean([p['weight'] for p in default_portions])
            print(f"  Default avg weight: {avg_weight_default:.1f}g")
        
        return metrics
    
    def analyze_nutrition_metrics(self) -> Dict:
        """Analyze nutrition estimation accuracy"""
        print("\n🥗 Nutrition Analysis:")
        
        custom_nutrition = self.custom_data.get('total_nutrition', {})
        default_nutrition = self.default_data.get('total_nutrition', {})
        
        metrics = {
            'custom': custom_nutrition,
            'default': default_nutrition,
            'difference': {}
        }
        
        # Calculate differences
        for nutrient in ['calories', 'protein_g', 'carbs_g', 'fat_g']:
            custom_val = custom_nutrition.get(nutrient, 0)
            default_val = default_nutrition.get(nutrient, 0)
            
            if custom_val > 0:
                diff_percent = ((default_val - custom_val) / custom_val) * 100
                metrics['difference'][nutrient] = diff_percent
            
            print(f"  {nutrient}: Custom={custom_val:.1f}, Default={default_val:.1f}")
        
        return metrics
    
    def calculate_bbox_coverage(self, bbox: Dict) -> float:
        """Calculate bounding box coverage percentage"""
        if not bbox:
            return 0
        
        width = bbox.get('x2', 0) - bbox.get('x1', 0)
        height = bbox.get('y2', 0) - bbox.get('y1', 0)
        
        # Assuming typical image size for percentage calculation
        # You might want to use actual image dimensions from the data
        typical_width = 1000
        typical_height = 800
        
        coverage = (width * height) / (typical_width * typical_height) * 100
        return min(coverage, 100)  # Cap at 100%
    
    def calculate_overall_score(self, detection, classification, portion, nutrition) -> Dict:
        """Calculate overall performance scores"""
        scores = {'custom': 0, 'default': 0}
        
        # Detection score (0-25 points)
        if detection['custom']['avg_confidence'] > detection['default']['avg_confidence']:
            scores['custom'] += 25
        else:
            scores['default'] += 25
        
        # Classification score (0-25 points)
        custom_class_conf = np.mean(classification['custom']['confidence']) if classification['custom']['confidence'] else 0
        default_class_conf = np.mean(classification['default']['confidence']) if classification['default']['confidence'] else 0
        
        if custom_class_conf > default_class_conf:
            scores['custom'] += 25
        else:
            scores['default'] += 25
        
        # Portion realism score (0-25 points)
        # Check if portions are realistic (between 50-500g for most foods)
        custom_realistic = all(50 <= p['weight'] <= 500 for p in portion['custom']['portions'])
        default_realistic = all(50 <= p['weight'] <= 500 for p in portion['default']['portions'])
        
        if custom_realistic and not default_realistic:
            scores['custom'] += 25
        elif default_realistic and not custom_realistic:
            scores['default'] += 25
        else:
            scores['custom'] += 12.5
            scores['default'] += 12.5
        
        # Nutrition realism score (0-25 points)
        # Check if calories are realistic (100-1000 per item)
        custom_cal = nutrition['custom'].get('calories', 0)
        default_cal = nutrition['default'].get('calories', 0)
        
        custom_realistic_cal = 100 <= custom_cal <= 1000
        default_realistic_cal = 100 <= default_cal <= 1000
        
        if custom_realistic_cal and not default_realistic_cal:
            scores['custom'] += 25
        elif default_realistic_cal and not custom_realistic_cal:
            scores['default'] += 25
        else:
            scores['custom'] += 12.5
            scores['default'] += 12.5
        
        return scores
    
    def create_comparison_visualizations(self, output_dir: str = "data/output/comparisons"):
        """Create comprehensive comparison visualizations"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create main comparison figure
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Detection Metrics Comparison
        self._plot_detection_comparison(plt.subplot(3, 3, 1))
        
        # 2. Confidence Scores
        self._plot_confidence_comparison(plt.subplot(3, 3, 2))
        
        # 3. Nutrition Comparison
        self._plot_nutrition_comparison(plt.subplot(3, 3, 3))
        
        # 4. Portion Size Comparison
        self._plot_portion_comparison(plt.subplot(3, 3, 4))
        
        # 5. Overall Score
        self._plot_overall_score(plt.subplot(3, 3, 5))
        
        # 6. Detailed Metrics Table
        self._plot_metrics_table(plt.subplot(3, 3, 6))
        
        # 7. Winner Summary
        self._plot_winner_summary(plt.subplot(3, 1, 3))
        
        # Main title
        plt.suptitle("Model Performance Comparison: Custom vs Default YOLOv8", 
                    fontsize=20, fontweight='bold')
        
        # Save figure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        viz_path = output_path / f"model_comparison_{timestamp}.png"
        plt.tight_layout()
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\n✅ Saved comparison visualization: {viz_path}")
        
        # Also save detailed report
        self._save_detailed_report(output_path, timestamp)
    
    def _plot_detection_comparison(self, ax):
        """Plot detection metrics comparison"""
        detection = self.comparison_results['detection']
        
        metrics = ['Avg Confidence', 'Num Detections', 'Has Masks']
        custom_values = [
            detection['custom']['avg_confidence'],
            detection['custom']['num_detections'],
            1 if detection['custom']['has_masks'] else 0
        ]
        default_values = [
            detection['default']['avg_confidence'],
            detection['default']['num_detections'],
            1 if detection['default']['has_masks'] else 0
        ]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, custom_values, width, label='Custom Model', color='#2ecc71')
        bars2 = ax.bar(x + width/2, default_values, width, label='Default YOLOv8', color='#3498db')
        
        ax.set_xlabel('Metrics')
        ax.set_title('Detection Performance', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend()
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.2f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom')
    
    def _plot_confidence_comparison(self, ax):
        """Plot confidence score distributions"""
        custom_conf = self.comparison_results['classification']['custom']['confidence']
        default_conf = self.comparison_results['classification']['default']['confidence']
        
        if custom_conf and default_conf:
            ax.boxplot([custom_conf, default_conf], labels=['Custom', 'Default'])
            ax.set_title('Classification Confidence Distribution', fontweight='bold')
            ax.set_ylabel('Confidence Score')
            ax.grid(True, alpha=0.3)
    
    def _plot_nutrition_comparison(self, ax):
        """Plot nutrition comparison"""
        nutrition = self.comparison_results['nutrition']
        
        nutrients = ['Calories', 'Protein', 'Carbs', 'Fat']
        custom_values = [
            nutrition['custom'].get('calories', 0),
            nutrition['custom'].get('protein_g', 0),
            nutrition['custom'].get('carbs_g', 0),
            nutrition['custom'].get('fat_g', 0)
        ]
        default_values = [
            nutrition['default'].get('calories', 0),
            nutrition['default'].get('protein_g', 0),
            nutrition['default'].get('carbs_g', 0),
            nutrition['default'].get('fat_g', 0)
        ]
        
        x = np.arange(len(nutrients))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, custom_values, width, label='Custom Model', color='#e74c3c')
        bars2 = ax.bar(x + width/2, default_values, width, label='Default YOLOv8', color='#f39c12')
        
        ax.set_xlabel('Nutrients')
        ax.set_ylabel('Amount')
        ax.set_title('Nutrition Estimation Comparison', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(nutrients)
        ax.legend()
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.0f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=8)
    
    def _plot_portion_comparison(self, ax):
        """Plot portion size comparison"""
        portion = self.comparison_results['portion']
        
        custom_weights = [p['weight'] for p in portion['custom']['portions']]
        default_weights = [p['weight'] for p in portion['default']['portions']]
        
        if custom_weights and default_weights:
            data = [custom_weights, default_weights]
            ax.violinplot(data, positions=[1, 2], showmeans=True)
            ax.set_xticks([1, 2])
            ax.set_xticklabels(['Custom', 'Default'])
            ax.set_ylabel('Weight (g)')
            ax.set_title('Portion Size Distribution', fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # Add mean values
            ax.text(1, np.mean(custom_weights), f'{np.mean(custom_weights):.1f}g', 
                   ha='center', va='bottom')
            ax.text(2, np.mean(default_weights), f'{np.mean(default_weights):.1f}g', 
                   ha='center', va='bottom')
    
    def _plot_overall_score(self, ax):
        """Plot overall performance scores"""
        scores = self.comparison_results['overall']
        
        # Create pie chart
        sizes = [scores['custom'], scores['default']]
        labels = ['Custom Model', 'Default YOLOv8']
        colors = ['#2ecc71', '#3498db']
        explode = (0.1, 0) if scores['custom'] > scores['default'] else (0, 0.1)
        
        ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
               shadow=True, startangle=90)
        ax.set_title('Overall Performance Score', fontweight='bold', fontsize=14)
    
    def _plot_metrics_table(self, ax):
        """Create detailed metrics comparison table"""
        ax.axis('tight')
        ax.axis('off')
        
        # Create comparison data
        table_data = [
            ['Metric', 'Custom Model', 'Default YOLOv8', 'Winner'],
            ['Avg Confidence', 
             f"{self.comparison_results['detection']['custom']['avg_confidence']:.3f}",
             f"{self.comparison_results['detection']['default']['avg_confidence']:.3f}",
             '✓ Custom' if self.comparison_results['detection']['custom']['avg_confidence'] > 
             self.comparison_results['detection']['default']['avg_confidence'] else '✓ Default'],
            ['Calories Estimate',
             f"{self.comparison_results['nutrition']['custom'].get('calories', 0):.0f}",
             f"{self.comparison_results['nutrition']['default'].get('calories', 0):.0f}",
             self._determine_nutrition_winner()],
            ['Portion Realism',
             self._get_portion_summary('custom'),
             self._get_portion_summary('default'),
             self._determine_portion_winner()]
        ]
        
        table = ax.table(cellText=table_data, cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 2)
        
        # Style header row
        for i in range(4):
            table[(0, i)].set_facecolor('#34495e')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        ax.set_title('Detailed Metrics Comparison', fontweight='bold', pad=20)
    
    def _plot_winner_summary(self, ax):
        """Plot final winner summary"""
        ax.axis('off')
        
        scores = self.comparison_results['overall']
        winner = 'Custom Model' if scores['custom'] > scores['default'] else 'Default YOLOv8'
        winner_score = max(scores['custom'], scores['default'])
        
        # Create summary text
        summary_text = f"""
        🏆 WINNER: {winner.upper()} 🏆
        
        Overall Performance Score: {winner_score}/100
        
        KEY FINDINGS:
        """
        
        # Add key findings based on analysis
        findings = []
        
        # Detection finding
        if self.comparison_results['detection']['custom']['avg_confidence'] > \
           self.comparison_results['detection']['default']['avg_confidence']:
            findings.append("✅ Custom model has higher detection confidence")
        else:
            findings.append("✅ Default model has higher detection confidence")
        
        # Nutrition finding
        custom_cal = self.comparison_results['nutrition']['custom'].get('calories', 0)
        default_cal = self.comparison_results['nutrition']['default'].get('calories', 0)
        
        if 100 <= custom_cal <= 1000 and not (100 <= default_cal <= 1000):
            findings.append("✅ Custom model provides more realistic calorie estimates")
        elif 100 <= default_cal <= 1000 and not (100 <= custom_cal <= 1000):
            findings.append("✅ Default model provides more realistic calorie estimates")
        
        # Add findings to summary
        for finding in findings:
            summary_text += f"\n        {finding}"
        
        # Recommendation
        summary_text += f"""
        
        RECOMMENDATION:
        Based on the comprehensive analysis, the {winner} demonstrates superior 
        performance in food detection and metadata extraction tasks.
        """
        
        ax.text(0.5, 0.5, summary_text, transform=ax.transAxes,
                fontsize=14, ha='center', va='center',
                bbox=dict(boxstyle="round,pad=1", facecolor='lightgreen' if winner == 'Custom Model' 
                         else 'lightblue', alpha=0.8))
    
    def _determine_nutrition_winner(self) -> str:
        """Determine which model has more realistic nutrition estimates"""
        custom_cal = self.comparison_results['nutrition']['custom'].get('calories', 0)
        default_cal = self.comparison_results['nutrition']['default'].get('calories', 0)
        
        # Check if calories are in realistic range (100-1000 for a meal)
        custom_realistic = 100 <= custom_cal <= 1000
        default_realistic = 100 <= default_cal <= 1000
        
        if custom_realistic and not default_realistic:
            return '✓ Custom'
        elif default_realistic and not custom_realistic:
            return '✓ Default'
        else:
            return 'Tie'
    
    def _determine_portion_winner(self) -> str:
        """Determine which model has more realistic portion estimates"""
        custom_portions = self.comparison_results['portion']['custom']['portions']
        default_portions = self.comparison_results['portion']['default']['portions']
        
        if not custom_portions or not default_portions:
            return 'N/A'
        
        # Check if portions are realistic (50-500g for most foods)
        custom_realistic = all(50 <= p['weight'] <= 500 for p in custom_portions)
        default_realistic = all(50 <= p['weight'] <= 500 for p in default_portions)
        
        if custom_realistic and not default_realistic:
            return '✓ Custom'
        elif default_realistic and not custom_realistic:
            return '✓ Default'
        else:
            return 'Tie'
    
    def _get_portion_summary(self, model: str) -> str:
        """Get portion summary for a model"""
        portions = self.comparison_results['portion'][model]['portions']
        if not portions:
            return 'N/A'
        
        weights = [p['weight'] for p in portions]
        return f"{np.mean(weights):.1f}g avg"
    
    def _save_detailed_report(self, output_path: Path, timestamp: str):
        """Save detailed text report"""
        report_path = output_path / f"model_comparison_report_{timestamp}.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("MODEL PERFORMANCE COMPARISON REPORT\n")
            f.write("="*60 + "\n\n")
            
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Custom Model: {self.custom_path.name}\n")
            f.write(f"Default Model: {self.default_path.name}\n\n")
            
            # Detection Analysis
            f.write("1. DETECTION & LOCALIZATION\n")
            f.write("-"*40 + "\n")
            detection = self.comparison_results['detection']
            f.write(f"Custom Model:\n")
            f.write(f"  - Detections: {detection['custom']['num_detections']}\n")
            f.write(f"  - Avg Confidence: {detection['custom']['avg_confidence']:.3f}\n")
            f.write(f"  - Has Masks: {detection['custom']['has_masks']}\n")
            f.write(f"Default Model:\n")
            f.write(f"  - Detections: {detection['default']['num_detections']}\n")
            f.write(f"  - Avg Confidence: {detection['default']['avg_confidence']:.3f}\n")
            f.write(f"  - Has Masks: {detection['default']['has_masks']}\n\n")
            
            # Classification Analysis
            f.write("2. CLASSIFICATION\n")
            f.write("-"*40 + "\n")
            classification = self.comparison_results['classification']
            f.write(f"Custom Model: {classification['custom']['classifications']}\n")
            f.write(f"Default Model: {classification['default']['classifications']}\n\n")
            
            # Nutrition Analysis
            f.write("3. NUTRITION ESTIMATION\n")
            f.write("-"*40 + "\n")
            nutrition = self.comparison_results['nutrition']
            f.write(f"Custom Model:\n")
            f.write(f"  - Calories: {nutrition['custom'].get('calories', 0):.0f}\n")
            f.write(f"  - Protein: {nutrition['custom'].get('protein_g', 0):.1f}g\n")
            f.write(f"  - Carbs: {nutrition['custom'].get('carbs_g', 0):.1f}g\n")
            f.write(f"  - Fat: {nutrition['custom'].get('fat_g', 0):.1f}g\n")
            f.write(f"Default Model:\n")
            f.write(f"  - Calories: {nutrition['default'].get('calories', 0):.0f}\n")
            f.write(f"  - Protein: {nutrition['default'].get('protein_g', 0):.1f}g\n")
            f.write(f"  - Carbs: {nutrition['default'].get('carbs_g', 0):.1f}g\n")
            f.write(f"  - Fat: {nutrition['default'].get('fat_g', 0):.1f}g\n\n")
            
            # Overall Score
            f.write("4. OVERALL PERFORMANCE\n")
            f.write("-"*40 + "\n")
            scores = self.comparison_results['overall']
            f.write(f"Custom Model Score: {scores['custom']}/100\n")
            f.write(f"Default Model Score: {scores['default']}/100\n\n")
            
            # Winner
            winner = 'Custom Model' if scores['custom'] > scores['default'] else 'Default YOLOv8'
            f.write(f"🏆 WINNER: {winner}\n")
        
        print(f"📄 Saved detailed report: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare custom vs default model results")
    parser.add_argument('--custom', type=str, required=True, 
                        help='Path to custom model JSON results')
    parser.add_argument('--default', type=str, required=True, 
                        help='Path to default model JSON results')
    parser.add_argument('--output-dir', type=str, default='data/output/comparisons',
                        help='Output directory for comparison results')
    
    args = parser.parse_args()
    
    # Verify files exist
    if not Path(args.custom).exists():
        print(f"❌ Custom model results not found: {args.custom}")
        return
    
    if not Path(args.default).exists():
        print(f"❌ Default model results not found: {args.default}")
        return
    
    # Create comparator
    comparator = ModelResultsComparator(args.custom, args.default)
    
    # Analyze all metrics
    results = comparator.analyze_all_metrics()
    
    # Create visualizations
    comparator.create_comparison_visualizations(args.output_dir)
    
    # Print summary
    print("\n" + "="*60)
    print("📊 COMPARISON SUMMARY")
    print("="*60)
    
    scores = results['overall']
    winner = 'Custom Model' if scores['custom'] > scores['default'] else 'Default YOLOv8'
    print(f"\n🏆 Winner: {winner}")
    print(f"Custom Model Score: {scores['custom']}/100")
    print(f"Default Model Score: {scores['default']}/100")
    
    print("\n✅ Comparison complete! Check the output directory for detailed results.")


if __name__ == "__main__":
    main()