"""Food-specific image preprocessing."""

import cv2
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class FoodImagePreprocessor:
    """Enhanced preprocessing specifically for food images."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize food image preprocessor."""
        self.config = config
        self.max_size = config.get('max_image_size', 1024)
        self.enhance_contrast = config.get('enhance_contrast', True)
        self.enhance_saturation = config.get('enhance_saturation', True)
        self.sharpen = config.get('sharpen', True)
        
        # Initialize CLAHE
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        
        logger.info("Food image preprocessor initialized")
    
    def load_and_preprocess(self, image_path: str) -> np.ndarray:
        """Load and preprocess a food image."""
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize if too large
        image_rgb = self._resize_if_needed(image_rgb)
        
        # Apply food-specific enhancements
        if self.enhance_contrast:
            image_rgb = self._enhance_contrast(image_rgb)
        
        if self.enhance_saturation:
            image_rgb = self._enhance_saturation(image_rgb)
        
        if self.sharpen:
            image_rgb = self._apply_sharpening(image_rgb)
        
        return image_rgb
    
    def _resize_if_needed(self, image: np.ndarray) -> np.ndarray:
        """Resize image if it's too large."""
        h, w = image.shape[:2]
        
        if max(h, w) > self.max_size:
            if h > w:
                new_h = self.max_size
                new_w = int(w * self.max_size / h)
            else:
                new_w = self.max_size
                new_h = int(h * self.max_size / w)
            
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            logger.debug(f"Resized image from {w}x{h} to {new_w}x{new_h}")
        
        return image
    
    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """Enhance contrast using CLAHE."""
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        lab[:, :, 0] = self.clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    def _enhance_saturation(self, image: np.ndarray) -> np.ndarray:
        """Enhance saturation for better food color visibility."""
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.2, 0, 255)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    
    def _apply_sharpening(self, image: np.ndarray) -> np.ndarray:
        """Apply subtle sharpening for better edge detection."""
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]])
        sharpened = cv2.filter2D(image, -1, kernel)
        return cv2.addWeighted(image, 0.7, sharpened, 0.3, 0)