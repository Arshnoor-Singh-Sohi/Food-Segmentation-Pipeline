#!/usr/bin/env python3
"""
Diagnose what each model is actually detecting
"""

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import matplotlib.pyplot as plt
import json

def diagnose_model(image_path: str, model_path: str = None):
    """Diagnose what a model detects"""
    
    # Load model
    if model_path and Path(model_path).exists():
        print(f"\n🔍 Analyzing CUSTOM model: {model_path}")
        model = YOLO(model_path)
        model_type = "custom"
    else:
        print(f"\n🔍 Analyzing DEFAULT YOLOv8 model")
        model = YOLO('yolov8n.pt')
        model_type = "default"
    
    # Run detection
    results = model.predict(image_path, conf=0.25, device='cpu', verbose=False)
    
    if not results or len(results) == 0:
        print("No detections")
        return
    
    result = results[0]
    
    # Print model classes
    print(f"\n📋 Model Classes Available:")
    if hasattr(model, 'names'):
        for idx, name in model.names.items():
            print(f"  Class {idx}: {name}")
    
    # Analyze detections
    print(f"\n🎯 Detections in image:")
    if result.boxes is not None:
        for i, box in enumerate(result.boxes):
            class_id = int(box.cls)
            confidence = float(box.conf)
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            area = (x2 - x1) * (y2 - y1)
            
            if hasattr(result, 'names'):
                class_name = result.names.get(class_id, f"class_{class_id}")
            else:
                class_name = f"class_{class_id}"
            
            print(f"  Detection {i+1}:")
            print(f"    - Class: {class_name} (ID: {class_id})")
            print(f"    - Confidence: {confidence:.3f}")
            print(f"    - BBox area: {area:.0f} pixels")
            print(f"    - Position: ({x1:.0f}, {y1:.0f}) to ({x2:.0f}, {y2:.0f})")
    
    # Visualize
    visualize_detections(image_path, result, model_type)
    
    return result

def visualize_detections(image_path, result, model_type):
    """Visualize what model detected"""
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(image_rgb)
    plt.title(f"{model_type.upper()} Model Detections", fontsize=16, fontweight='bold')
    
    if result.boxes is not None:
        for i, box in enumerate(result.boxes):
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            class_id = int(box.cls)
            conf = float(box.conf)
            
            # Draw box
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                               linewidth=3, edgecolor='red', facecolor='none')
            plt.gca().add_patch(rect)
            
            # Add label
            if hasattr(result, 'names'):
                class_name = result.names.get(class_id, f"class_{class_id}")
            else:
                class_name = f"class_{class_id}"
                
            label = f"{class_name} ({conf:.2f})"
            plt.text(x1, y1-10, label, color='white', fontsize=10, 
                    weight='bold', bbox=dict(boxstyle="round,pad=0.3", 
                    facecolor='red', alpha=0.7))
    
    plt.axis('off')
    plt.tight_layout()
    
    # Save
    output_path = f"data/output/diagnosis_{model_type}_detections.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n💾 Saved visualization: {output_path}")
    plt.close()

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Diagnose model detections")
    parser.add_argument('--image', type=str, required=True, help='Image path')
    parser.add_argument('--model', type=str, help='Custom model path')
    
    args = parser.parse_args()
    
    # Diagnose default model
    diagnose_model(args.image)
    
    # Diagnose custom model if provided
    if args.model:
        diagnose_model(args.image, args.model)

if __name__ == "__main__":
    main()