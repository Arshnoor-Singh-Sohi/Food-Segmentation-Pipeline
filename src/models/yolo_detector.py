"""Enhanced YOLO detector specifically optimized for food detection."""

import torch
import numpy as np
import cv2
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import logging
from ultralytics import YOLO
import supervision as sv

logger = logging.getLogger(__name__)

class FoodYOLODetector:
    """Enhanced YOLO detector optimized for food detection with better preprocessing."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize YOLO detector with food-specific optimizations."""
        self.config = config
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.confidence_threshold = config.get('confidence_threshold', 0.25)
        self.iou_threshold = config.get('iou_threshold', 0.45)
        self.model = None
        
        # Food-specific class mapping (if using custom model)
        self.food_classes = self._load_food_classes()
        
        logger.info(f"Initializing YOLO food detector on device: {self.device}")
        self._load_model()
    
    def _load_model(self):
        """Load YOLO model with fallback options."""
        model_path = self.config.get('model_path')
        
        try:
            # Try to load custom food model first
            if model_path and Path(model_path).exists():
                logger.info(f"Loading custom food model: {model_path}")
                self.model = YOLO(model_path)
            else:
                # Fallback to pre-trained YOLOv8 and enhance for food
                logger.info("Custom model not found, using YOLOv8n with food enhancements")
                self.model = YOLO('yolov8n.pt')
                
                # You can download a food-specific model here
                self._try_download_food_model()
            
            # Move model to specified device
            self.model.to(self.device)
            
            logger.info("YOLO model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise
    
    def _try_download_food_model(self):
        """Attempt to download a food-specific YOLO model."""
        try:
            # You can add URLs to food-specific models here
            food_model_urls = [
                # Example: "https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5s_food.pt"
            ]
            
            # For now, we'll enhance the general model for food detection
            logger.info("Using general YOLOv8 model with food-specific post-processing")
            
        except Exception as e:
            logger.warning(f"Could not download food-specific model: {e}")
    
    def _load_food_classes(self) -> Dict[str, int]:
        """Load food class mappings."""
        # Default COCO classes that are food-related
        coco_food_classes = {
            'apple': 47,
            'banana': 46,
            'orange': 49,
            'broccoli': 56,
            'carrot': 57,
            'hot dog': 58,
            'pizza': 59,
            'donut': 60,
            'cake': 61,
            'sandwich': 54,
        }
        
        return coco_food_classes
    
    def detect_food_items(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect food items in the image with enhanced preprocessing."""
        if image is None or image.size == 0:
            raise ValueError("Invalid image provided")
        
        # Preprocess image for better food detection
        processed_image = self._preprocess_for_food_detection(image)
        
        # Run detection
        results = self.model(
            processed_image,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False
        )
        
        # Process and filter results for food items
        detections = self._process_detections(results[0], image.shape)
        
        # Enhance detections with food-specific post-processing
        enhanced_detections = self._enhance_food_detections(detections, image)
        
        logger.info(f"Detected {len(enhanced_detections)} food items")
        return enhanced_detections
    
    def _preprocess_for_food_detection(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image specifically for food detection."""
        processed = image.copy()
        
        # Convert to RGB if needed
        if len(processed.shape) == 3 and processed.shape[2] == 3:
            # Enhance contrast and saturation for better food visibility
            
            # 1. CLAHE for better contrast
            lab = cv2.cvtColor(processed, cv2.COLOR_RGB2LAB)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            processed = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            
            # 2. Enhance saturation for food colors
            hsv = cv2.cvtColor(processed, cv2.COLOR_RGB2HSV)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.2, 0, 255)  # Increase saturation
            processed = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
            
            # 3. Slight sharpening for better edge detection
            kernel = np.array([[-1, -1, -1],
                             [-1,  9, -1],
                             [-1, -1, -1]])
            sharpened = cv2.filter2D(processed, -1, kernel)
            processed = cv2.addWeighted(processed, 0.7, sharpened, 0.3, 0)
        
        return processed
    
    def _process_detections(self, results, image_shape: Tuple[int, int, int]) -> List[Dict[str, Any]]:
        """Process raw YOLO detections into structured format."""
        detections = []
        
        if results.boxes is not None:
            boxes = results.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
            confidences = results.boxes.conf.cpu().numpy()
            classes = results.boxes.cls.cpu().numpy().astype(int)
            
            for box, conf, cls in zip(boxes, confidences, classes):
                # Get class name
                class_name = self.model.names[cls] if cls < len(self.model.names) else f"class_{cls}"
                
                # Filter for food-related classes or high-confidence detections
                if self._is_food_related(class_name, conf):
                    detection = {
                        'bbox': box.tolist(),  # [x1, y1, x2, y2]
                        'confidence': float(conf),
                        'class_id': int(cls),
                        'class_name': class_name,
                        'center': [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2],
                        'area': (box[2] - box[0]) * (box[3] - box[1]),
                        'aspect_ratio': (box[2] - box[0]) / (box[3] - box[1])
                    }
                    detections.append(detection)
        
        return detections
    
    def _is_food_related(self, class_name: str, confidence: float) -> bool:
        """Determine if a detection is food-related."""
        # Known food classes from COCO
        food_keywords = {
            'apple', 'banana', 'orange', 'broccoli', 'carrot', 'hot dog',
            'pizza', 'donut', 'cake', 'sandwich', 'bottle', 'cup', 'bowl',
            'spoon', 'knife', 'fork'
        }
        
        # Check if class name contains food keywords
        class_lower = class_name.lower()
        is_food = any(keyword in class_lower for keyword in food_keywords)
        
        # For non-food classes, require higher confidence
        if not is_food and confidence < 0.5:
            return False
        
        # For food classes, accept lower confidence
        if is_food and confidence < 0.2:
            return False
        
        return True
    
    def _enhance_food_detections(self, detections: List[Dict], image: np.ndarray) -> List[Dict]:
        """Enhance detections with food-specific analysis."""
        enhanced_detections = []
        
        for detection in detections:
            # Extract region for analysis
            bbox = detection['bbox']
            x1, y1, x2, y2 = map(int, bbox)
            
            # Ensure coordinates are within image bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(image.shape[1], x2)
            y2 = min(image.shape[0], y2)
            
            if x2 <= x1 or y2 <= y1:
                continue
            
            food_region = image[y1:y2, x1:x2]
            
            # Analyze food region characteristics
            food_analysis = self._analyze_food_region(food_region)
            
            # Enhance detection with analysis
            enhanced_detection = detection.copy()
            enhanced_detection.update({
                'food_analysis': food_analysis,
                'region_bbox': [x1, y1, x2, y2],  # Clipped bbox
                'is_likely_food': food_analysis['is_likely_food'],
                'color_profile': food_analysis['color_profile'],
                'texture_score': food_analysis['texture_score']
            })
            
            # Only keep detections that are likely food
            if food_analysis['is_likely_food']:
                enhanced_detections.append(enhanced_detection)
        
        return enhanced_detections
    
    def _analyze_food_region(self, region: np.ndarray) -> Dict[str, Any]:
        """Analyze a detected region to determine if it's likely food."""
        if region.size == 0:
            return {
                'is_likely_food': False,
                'color_profile': 'unknown',
                'texture_score': 0.0,
                'brightness': 0.0,
                'color_variance': 0.0
            }
        
        try:
            # Color analysis
            mean_color = np.mean(region, axis=(0, 1))
            color_std = np.std(region, axis=(0, 1))
            brightness = np.mean(mean_color)
            color_variance = np.mean(color_std)
            
            # Texture analysis
            gray_region = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
            texture_score = np.std(gray_region) / 255.0
            
            # Determine color profile
            color_profile = self._classify_color_profile(mean_color)
            
            # Food likelihood heuristics
            is_likely_food = (
                15 < brightness < 240 and      # Not too dark or bright
                color_variance > 5 and         # Some color variation
                texture_score > 0.1 and        # Some texture
                self._is_food_color_profile(color_profile)
            )
            
            return {
                'is_likely_food': is_likely_food,
                'color_profile': color_profile,
                'texture_score': float(texture_score),
                'brightness': float(brightness),
                'color_variance': float(color_variance)
            }
            
        except Exception as e:
            logger.warning(f"Error analyzing food region: {e}")
            return {
                'is_likely_food': True,  # Default to keeping detection
                'color_profile': 'unknown',
                'texture_score': 0.5,
                'brightness': 128.0,
                'color_variance': 10.0
            }
    
    def _classify_color_profile(self, mean_color: np.ndarray) -> str:
        """Classify the dominant color profile of a region."""
        r, g, b = mean_color
        
        # Simple color classification
        if r > g and r > b and r > 100:
            if r > 180 and g < 100 and b < 100:
                return 'red'
            elif r > 150 and g > 100 and b < 100:
                return 'orange'
            else:
                return 'reddish'
        elif g > r and g > b and g > 100:
            if g > 150 and r < 100 and b < 100:
                return 'green'
            else:
                return 'greenish'
        elif b > r and b > g and b > 100:
            return 'blue'
        elif r > 150 and g > 150 and b < 100:
            return 'yellow'
        elif r > 120 and g > 80 and b > 60:
            return 'brown'
        elif r > 200 and g > 200 and b > 200:
            return 'white'
        elif r < 50 and g < 50 and b < 50:
            return 'black'
        else:
            return 'mixed'
    
    def _is_food_color_profile(self, color_profile: str) -> bool:
        """Check if color profile is typical for food."""
        food_colors = {
            'red', 'orange', 'green', 'yellow', 'brown', 'reddish', 
            'greenish', 'mixed'
        }
        return color_profile in food_colors
    
    def get_detection_summary(self, detections: List[Dict]) -> Dict[str, Any]:
        """Get summary statistics of detections."""
        if not detections:
            return {
                'total_detections': 0,
                'avg_confidence': 0.0,
                'food_types': [],
                'detection_areas': []
            }
        
        confidences = [d['confidence'] for d in detections]
        food_types = list(set(d['class_name'] for d in detections))
        areas = [d['area'] for d in detections]
        
        return {
            'total_detections': len(detections),
            'avg_confidence': float(np.mean(confidences)),
            'max_confidence': float(max(confidences)),
            'min_confidence': float(min(confidences)),
            'food_types': food_types,
            'detection_areas': areas,
            'total_area': sum(areas)
        }