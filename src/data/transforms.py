"""
Data augmentation transforms for PPE detection.
"""

import cv2
import numpy as np
import random
from typing import Tuple


class TrainTransform:
    """
    Training augmentation transform.
    
    Includes:
    - Random scaling
    - Random flip
    - Color jittering
    - Normalization
    """
    
    def __init__(
        self,
        input_size: Tuple[int, int] = (320, 320),
        scale_range: Tuple[float, float] = (0.6, 1.4),
        flip_prob: float = 0.5,
        brightness: float = 0.2,
        contrast: Tuple[float, float] = (0.6, 1.4),
        saturation: Tuple[float, float] = (0.5, 1.2),
        mean: Tuple[float, ...] = (103.53, 116.28, 123.675),
        std: Tuple[float, ...] = (57.375, 57.12, 58.395)
    ):
        self.input_size = input_size
        self.scale_range = scale_range
        self.flip_prob = flip_prob
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
    
    def __call__(
        self,
        image: np.ndarray,
        boxes: np.ndarray,
        labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply transforms."""
        h, w = image.shape[:2]
        
        # Random scale
        scale = random.uniform(*self.scale_range)
        new_h, new_w = int(h * scale), int(w * scale)
        image = cv2.resize(image, (new_w, new_h))
        
        if len(boxes) > 0:
            boxes = boxes * scale
        
        # Random flip
        if random.random() < self.flip_prob:
            image = cv2.flip(image, 1)
            if len(boxes) > 0:
                boxes[:, [0, 2]] = new_w - boxes[:, [2, 0]]
        
        # Color jittering
        image = self._color_jitter(image)
        
        # Resize to input size
        image, boxes = self._resize_with_boxes(image, boxes, self.input_size)
        
        # Clip boxes
        if len(boxes) > 0:
            boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, self.input_size[0])
            boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, self.input_size[1])
            
            # Remove invalid boxes
            valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
            boxes = boxes[valid]
            labels = labels[valid]
        
        # Normalize
        image = image.astype(np.float32)
        image = (image - self.mean) / self.std
        
        # To tensor format [C, H, W]
        import torch
        image = torch.from_numpy(image.transpose(2, 0, 1)).float()
        
        return image, boxes, labels
    
    def _color_jitter(self, image: np.ndarray) -> np.ndarray:
        """Apply color jittering."""
        # Brightness
        if self.brightness > 0:
            delta = random.uniform(-self.brightness, self.brightness) * 255
            image = np.clip(image + delta, 0, 255).astype(np.uint8)
        
        # Contrast
        alpha = random.uniform(*self.contrast)
        image = np.clip(image * alpha, 0, 255).astype(np.uint8)
        
        # Saturation
        if random.random() < 0.5:
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            s_scale = random.uniform(*self.saturation)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * s_scale, 0, 255).astype(np.uint8)
            image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        return image
    
    def _resize_with_boxes(
        self,
        image: np.ndarray,
        boxes: np.ndarray,
        target_size: Tuple[int, int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Resize image and adjust boxes."""
        h, w = image.shape[:2]
        tw, th = target_size
        
        scale_x = tw / w
        scale_y = th / h
        
        image = cv2.resize(image, target_size)
        
        if len(boxes) > 0:
            boxes[:, [0, 2]] *= scale_x
            boxes[:, [1, 3]] *= scale_y
        
        return image, boxes


class ValTransform:
    """
    Validation transform (no augmentation).
    
    Only resize and normalize.
    """
    
    def __init__(
        self,
        input_size: Tuple[int, int] = (320, 320),
        mean: Tuple[float, ...] = (103.53, 116.28, 123.675),
        std: Tuple[float, ...] = (57.375, 57.12, 58.395)
    ):
        self.input_size = input_size
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
    
    def __call__(
        self,
        image: np.ndarray,
        boxes: np.ndarray,
        labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply transforms."""
        h, w = image.shape[:2]
        tw, th = self.input_size
        
        # Scale factors
        scale_x = tw / w
        scale_y = th / h
        
        # Resize
        image = cv2.resize(image, self.input_size)
        
        # Adjust boxes
        if len(boxes) > 0:
            boxes[:, [0, 2]] *= scale_x
            boxes[:, [1, 3]] *= scale_y
        
        # Normalize
        image = image.astype(np.float32)
        image = (image - self.mean) / self.std
        
        # To tensor format [C, H, W]
        import torch
        image = torch.from_numpy(image.transpose(2, 0, 1)).float()
        
        return image, boxes, labels


class InferenceTransform:
    """Transform for inference (single image)."""
    
    def __init__(
        self,
        input_size: Tuple[int, int] = (320, 320),
        mean: Tuple[float, ...] = (103.53, 116.28, 123.675),
        std: Tuple[float, ...] = (57.375, 57.12, 58.395)
    ):
        self.input_size = input_size
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
    
    def __call__(self, image: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Transform image for inference.
        
        Returns:
            Transformed image and metadata for scaling boxes back
        """
        h, w = image.shape[:2]
        tw, th = self.input_size
        
        meta = {
            "original_size": (w, h),
            "input_size": self.input_size,
            "scale": (tw / w, th / h)
        }
        
        # Resize
        image = cv2.resize(image, self.input_size)
        
        # Normalize
        image = image.astype(np.float32)
        image = (image - self.mean) / self.std
        
        # To tensor format [C, H, W]
        image = image.transpose(2, 0, 1)
        
        return image, meta
