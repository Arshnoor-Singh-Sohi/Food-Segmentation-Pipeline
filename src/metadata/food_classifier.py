"""
Food classification module using pre-trained models
"""

import torch
import numpy as np
from transformers import AutoImageProcessor, AutoModelForImageClassification
from typing import Tuple, List, Dict

class FoodClassifier:
    """Classify food items using pre-trained models"""
    
    def __init__(self, model_name: str = "nateraw/food"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load Food-101 model
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModelForImageClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
    def classify(self, image: np.ndarray, top_k: int = 3) -> List[Dict[str, float]]:
        """
        Classify food image
        
        Args:
            image: RGB image array
            top_k: Return top k predictions
            
        Returns:
            List of {label, confidence} dicts
        """
        # Preprocess
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
        
        # Get top predictions
        top_probs, top_indices = torch.topk(probs, k=top_k, dim=-1)
        
        results = []
        for i in range(top_k):
            idx = top_indices[0, i].item()
            label = self.model.config.id2label[idx]
            confidence = top_probs[0, i].item()
            
            # Clean label (Food-101 uses underscores)
            label = label.replace('_', ' ')
            
            results.append({
                'label': label,
                'confidence': confidence
            })
        
        return results
    
    def get_food_category(self, food_label: str) -> str:
        """Map food label to general category"""
        categories = {
            'fruit': ['apple', 'banana', 'orange', 'berry'],
            'vegetable': ['salad', 'broccoli', 'carrot'],
            'protein': ['steak', 'chicken', 'fish', 'eggs'],
            'grain': ['bread', 'rice', 'pasta'],
            'dairy': ['cheese', 'yogurt', 'ice cream'],
            'dessert': ['cake', 'donut', 'chocolate'],
            'prepared': ['pizza', 'burger', 'sandwich', 'soup']
        }
        
        food_lower = food_label.lower()
        for category, keywords in categories.items():
            if any(keyword in food_lower for keyword in keywords):
                return category
        
        return 'other'