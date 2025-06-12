"""Generate HTML viewer for single image results."""

import json
from pathlib import Path
import webbrowser

def create_html_viewer(json_file_path):
    """Create HTML viewer for single image results."""
    
    # Load results
    with open(json_file_path, 'r') as f:
        results = json.load(f)
    
    if 'error' in results:
        print(f"Error in results: {results['error']}")
        return
    
    # Create HTML
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Food Analysis Results</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
            .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; }}
            .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; }}
            .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }}
            .stat-card {{ background: #e3f2fd; padding: 15px; border-radius: 8px; text-align: center; border-left: 4px solid #2196f3; }}
            .stat-number {{ font-size: 2em; font-weight: bold; color: #1976d2; }}
            .stat-label {{ color: #666; margin-top: 5px; }}
            .items-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 15px; }}
            .item-card {{ border: 1px solid #ddd; border-radius: 8px; padding: 15px; background: white; }}
            .item-card.food {{ border-left: 4px solid #4caf50; }}
            .item-card.non-food {{ border-left: 4px solid #ff9800; }}
            .confidence-bar {{ background: #f0f0f0; border-radius: 10px; height: 20px; margin: 10px 0; }}
            .confidence-fill {{ background: #4caf50; height: 100%; border-radius: 10px; text-align: center; line-height: 20px; color: white; font-size: 12px; }}
            .nutrition {{ background: #f8f9fa; padding: 10px; border-radius: 5px; margin-top: 10px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🍽️ Food Analysis Results</h1>
                <p><strong>Image:</strong> {results['image_info']['filename']}</p>
                <p><strong>Processing Time:</strong> {results['image_info']['processing_time_seconds']}s</p>
            </div>
            
            <div class="stats">
                <div class="stat-card">
                    <div class="stat-number">{results['analysis_summary']['total_items_detected']}</div>
                    <div class="stat-label">Total Items</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{results['analysis_summary']['food_items_count']}</div>
                    <div class="stat-label">Food Items</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{results['nutrition_totals']['calories']:.1f}</div>
                    <div class="stat-label">Total Calories</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{results['analysis_summary']['avg_confidence']:.2f}</div>
                    <div class="stat-label">Avg Confidence</div>
                </div>
            </div>
            
            <h2>[INFO] Detected Items</h2>
            <div class="items-grid">
    """
    
    for item in results['food_items']:
        item_class = "food" if item['is_food'] else "non-food"
        confidence_percent = item['confidence'] * 100
        
        nutrition_html = ""
        if item['is_food']:
            nutrition = item['nutrition']
            nutrition_html = f"""
            <div class="nutrition">
                <strong>Nutrition (estimated):</strong><br>
                [FIRE] {nutrition['calories']:.1f} calories<br>
                🥩 {nutrition['protein_g']:.1f}g protein<br>
                🍞 {nutrition['carbs_g']:.1f}g carbs<br>
                🧈 {nutrition['fat_g']:.1f}g fat<br>
                ⚖️ {nutrition['portion_grams']}g portion
            </div>
            """
        
        html_content += f"""
        <div class="item-card {item_class}">
            <h3>{'🍎' if item['is_food'] else '🍽️'} {item['name'].title()}</h3>
            <div class="confidence-bar">
                <div class="confidence-fill" style="width: {confidence_percent}%">
                    {confidence_percent:.1f}% confidence
                </div>
            </div>
            <p><strong>Type:</strong> {'Food Item' if item['is_food'] else 'Non-Food Item'}</p>
            <p><strong>Area:</strong> {item['mask_info']['area_percentage']:.1f}% of image</p>
            {nutrition_html}
        </div>
        """
    
    html_content += """
            </div>
            
            <div style="margin-top: 30px; padding: 20px; background: #f8f9fa; border-radius: 8px;">
                <h3>[STATS] Summary</h3>
                <p><strong>Total Nutrition:</strong></p>
                <ul>
    """
    
    nutrition = results['nutrition_totals']
    html_content += f"""
                    <li>[FIRE] <strong>Calories:</strong> {nutrition['calories']:.1f}</li>
                    <li>🥩 <strong>Protein:</strong> {nutrition['protein_g']:.1f}g</li>
                    <li>🍞 <strong>Carbohydrates:</strong> {nutrition['carbs_g']:.1f}g</li>
                    <li>🧈 <strong>Fat:</strong> {nutrition['fat_g']:.1f}g</li>
    """
    
    html_content += """
                </ul>
            </div>
        </div>
    </body>
    </html>
    """
    
    # Save HTML file
    html_file = Path(json_file_path).parent / f"{Path(json_file_path).stem}_viewer.html"
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"HTML viewer created: {html_file}")
    
    # Open in browser
    webbrowser.open(f'file://{html_file.absolute()}')
    
    return str(html_file)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        json_file = sys.argv[1]
        create_html_viewer(json_file)
    else:
        # Find the most recent results file
        results_dir = Path("data/output/yolo_results")
        json_files = list(results_dir.glob("*_results.json"))
        if json_files:
            latest_file = max(json_files, key=lambda p: p.stat().st_mtime)
            print(f"Using latest results: {latest_file}")
            create_html_viewer(latest_file)
        else:
            print("No results files found. Process an image first!")