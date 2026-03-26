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
    - Mosaic (optional)
    - MixUp (optional)
    - Normalization
    """
    
    def __init__(
        self,
        input_size: Tuple[int, int] = (320, 320),
        scale_range: Tuple[float, float] = (0.5, 1.5),
        flip_prob: float = 0.5,
        brightness: float = 0.3,
        contrast: Tuple[float, float] = (0.5, 1.5),
        saturation: Tuple[float, float] = (0.4, 1.4),
        hue_delta: int = 18,
        mean: Tuple[float, ...] = (123.675, 116.28, 103.53),  # RGB order (ImageNet)
        std: Tuple[float, ...] = (58.395, 57.12, 57.375)      # RGB order (ImageNet)
    ):
        self.input_size = input_size
        self.scale_range = scale_range
        self.flip_prob = flip_prob
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue_delta = hue_delta
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
        
        # Clip boxes to image bounds - input_size is (width, height), boxes are (x1, y1, x2, y2)
        if len(boxes) > 0:
            boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, self.input_size[0] - 1)  # x to width
            boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, self.input_size[1] - 1)  # y to height
            
            # Remove invalid boxes (must have positive width and height)
            # Keep boxes with at least 2 pixels in each dimension for numerical stability
            widths = boxes[:, 2] - boxes[:, 0]
            heights = boxes[:, 3] - boxes[:, 1]
            valid = (widths >= 2) & (heights >= 2)
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
        """Apply color jittering with HSV augmentation."""
        # Convert to HSV for better color manipulation
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
        
        # Hue shift
        if self.hue_delta > 0 and random.random() < 0.5:
            hsv[:, :, 0] += random.uniform(-self.hue_delta, self.hue_delta)
            hsv[:, :, 0] = np.clip(hsv[:, :, 0], 0, 180)
        
        # Saturation
        if random.random() < 0.5:
            s_scale = random.uniform(*self.saturation)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * s_scale, 0, 255)
        
        # Value (brightness)
        if self.brightness > 0 and random.random() < 0.5:
            v_delta = random.uniform(1 - self.brightness, 1 + self.brightness)
            hsv[:, :, 2] = np.clip(hsv[:, :, 2] * v_delta, 0, 255)
        
        image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        
        # Contrast
        if random.random() < 0.5:
            alpha = random.uniform(*self.contrast)
            image = np.clip(image.astype(np.float32) * alpha, 0, 255).astype(np.uint8)
        
        # Random grayscale
        if random.random() < 0.1:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            image = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        
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
        mean: Tuple[float, ...] = (123.675, 116.28, 103.53),  # RGB order (ImageNet)
        std: Tuple[float, ...] = (58.395, 57.12, 57.375)      # RGB order (ImageNet)
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
        mean: Tuple[float, ...] = (123.675, 116.28, 103.53),  # RGB order (ImageNet)
        std: Tuple[float, ...] = (58.395, 57.12, 57.375)      # RGB order (ImageNet)
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
