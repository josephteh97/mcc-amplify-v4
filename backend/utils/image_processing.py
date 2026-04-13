"""
Image Processing Utilities
"""

import cv2
import numpy as np
from typing import Tuple

def resize_image_aspect_ratio(image: np.ndarray, max_size: int) -> np.ndarray:
    """Resize image maintaining aspect ratio"""
    h, w = image.shape[:2]
    
    if max(h, w) <= max_size:
        return image
        
    scale = max_size / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert to grayscale if needed"""
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image
