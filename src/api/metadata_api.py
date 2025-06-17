"""
Flask API server for metadata extraction
Deploy this on RunPod for web-based access
"""

from flask import Flask, request, jsonify, send_file
import os
import tempfile
from pathlib import Path
import json

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.process_with_metadata import process_image_with_metadata

app = Flask(__name__)

@app.route('/')
def home():
    return jsonify({
        "service": "Food Metadata Extraction API",
        "version": "1.0",
        "endpoints": {
            "/analyze": "POST - Analyze food image",
            "/health": "GET - Health check"
        }
    })

@app.route('/health')
def health():
    return jsonify({"status": "healthy"})

@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyze uploaded food image"""
    try:
        # Check if image was uploaded
        if 'image' not in request.files:
            return jsonify({"error": "No image uploaded"}), 400
        
        file = request.files['image']
        
        # Save temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            file.save(tmp_file.name)
            
            # Process image
            results = process_image_with_metadata(tmp_file.name)
            
            # Clean up
            os.unlink(tmp_file.name)
            
            if results:
                return jsonify(results)
            else:
                return jsonify({"error": "Processing failed"}), 500
                
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)