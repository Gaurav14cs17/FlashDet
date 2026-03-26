"""
Visualization utilities for PPE detection.
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict

# Class names
CLASS_NAMES = [
    "Hardhat", "Mask", "NO-Hardhat", "NO-Mask", "NO-Safety Vest",
    "Person", "Safety Cone", "Safety Vest", "machinery", "vehicle"
]

# Colors for each class (BGR format)
COLORS = {
    "Hardhat": (0, 255, 0),        # Green - Safe
    "Mask": (0, 255, 0),            # Green - Safe
    "NO-Hardhat": (0, 0, 255),      # Red - Violation
    "NO-Mask": (0, 0, 255),         # Red - Violation
    "NO-Safety Vest": (0, 0, 255),  # Red - Violation
    "Person": (255, 255, 0),        # Cyan
    "Safety Cone": (0, 165, 255),   # Orange
    "Safety Vest": (0, 255, 0),     # Green - Safe
    "machinery": (128, 128, 0),     # Teal
    "vehicle": (128, 0, 128)        # Purple
}


def draw_boxes(
    image: np.ndarray,
    boxes: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray = None,
    class_names: List[str] = None,
    colors: Dict[str, Tuple[int, int, int]] = None,
    thickness: int = 2,
    font_scale: float = 0.5
) -> np.ndarray:
    """
    Draw bounding boxes on image.
    
    Args:
        image: Input image (BGR)
        boxes: Bounding boxes [N, 4] (x1, y1, x2, y2)
        labels: Class labels [N]
        scores: Confidence scores [N] (optional)
        class_names: List of class names
        colors: Dict mapping class names to colors
        thickness: Line thickness
        font_scale: Font scale for labels
        
    Returns:
        Image with drawn boxes
    """
    class_names = class_names or CLASS_NAMES
    colors = colors or COLORS
    
    output = image.copy()
    
    for i, (box, label) in enumerate(zip(boxes, labels)):
        x1, y1, x2, y2 = map(int, box)
        class_name = class_names[int(label)]
        color = colors.get(class_name, (255, 255, 255))
        
        # Draw box
        cv2.rectangle(output, (x1, y1), (x2, y2), color, thickness)
        
        # Create label
        if scores is not None:
            text = f"{class_name}: {scores[i]:.2f}"
        else:
            text = class_name
        
        # Draw label background
        (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        cv2.rectangle(output, (x1, y1 - h - 8), (x1 + w + 4, y1), color, -1)
        
        # Draw label text
        cv2.putText(
            output, text, (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1
        )
    
    return output


def draw_detections(
    image: np.ndarray,
    detections: List[Tuple],
    class_names: List[str] = None
) -> np.ndarray:
    """
    Draw detections on image.
    
    Args:
        image: Input image (BGR)
        detections: List of (class_name, score, x1, y1, x2, y2)
        class_names: List of class names
        
    Returns:
        Image with drawn detections
    """
    output = image.copy()
    
    for det in detections:
        if len(det) == 6:
            class_name, score, x1, y1, x2, y2 = det
        else:
            x1, y1, x2, y2, score, class_id = det
            class_names = class_names or CLASS_NAMES
            class_name = class_names[int(class_id)]
        
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        color = COLORS.get(class_name, (255, 255, 255))
        
        # Draw box
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        text = f"{class_name}: {score:.2f}"
        (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(output, (x1, y1 - h - 8), (x1 + w + 4, y1), color, -1)
        cv2.putText(
            output, text, (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
        )
    
    return output


def add_fps_overlay(
    image: np.ndarray,
    fps: float,
    position: Tuple[int, int] = (10, 30)
) -> np.ndarray:
    """Add FPS counter overlay to image."""
    cv2.putText(
        image, f"FPS: {fps:.1f}", position,
        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
    )
    return image


def add_violation_warning(
    image: np.ndarray,
    num_violations: int,
    position: Tuple[int, int] = (10, 70)
) -> np.ndarray:
    """Add violation warning overlay to image."""
    if num_violations > 0:
        cv2.putText(
            image, f"VIOLATIONS: {num_violations}", position,
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
        )
    return image


def count_violations(detections: List[Tuple]) -> Tuple[List, List]:
    """
    Count safety violations in detections.
    
    Returns:
        Tuple of (violations, safe_detections)
    """
    violations = []
    safe = []
    
    for det in detections:
        class_name = det[0] if isinstance(det[0], str) else CLASS_NAMES[int(det[5])]
        
        if class_name.startswith("NO-"):
            violations.append(det)
        elif class_name in ["Hardhat", "Mask", "Safety Vest"]:
            safe.append(det)
    
    return violations, safe
