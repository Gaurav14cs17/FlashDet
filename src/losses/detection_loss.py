"""
Combined Detection Loss for NanoDet-Plus-Lite.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple

from .focal_loss import QualityFocalLoss, DistributionFocalLoss
from .iou_loss import GIoULoss


class DetectionLoss(nn.Module):
    """
    Combined loss for object detection.
    
    Combines:
    - Quality Focal Loss for classification
    - Distribution Focal Loss for box regression
    - GIoU Loss for box regression
    
    Args:
        num_classes: Number of detection classes
        reg_max: Maximum value for distribution
        loss_weights: Dictionary of loss weights
    """
    
    def __init__(
        self,
        num_classes: int = 10,
        reg_max: int = 7,
        loss_weights: Dict[str, float] = None
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.reg_max = reg_max
        
        weights = loss_weights or {
            "cls": 1.0,
            "dfl": 0.25,
            "bbox": 2.0
        }
        
        self.cls_loss = QualityFocalLoss(beta=2.0, loss_weight=weights["cls"])
        self.dfl_loss = DistributionFocalLoss(loss_weight=weights["dfl"])
        self.bbox_loss = GIoULoss(loss_weight=weights["bbox"])
    
    def forward(
        self,
        cls_preds: List[torch.Tensor],
        bbox_preds: List[torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute detection loss.
        
        Args:
            cls_preds: Classification predictions per scale
            bbox_preds: Bounding box predictions per scale
            targets: Dictionary with 'cls_targets', 'bbox_targets', 'pos_mask'
            
        Returns:
            Dictionary with loss values
        """
        device = cls_preds[0].device
        
        cls_targets = targets["cls_targets"]
        bbox_targets = targets["bbox_targets"]
        pos_mask = targets["pos_mask"]
        
        num_pos = pos_mask.sum().clamp(min=1)
        
        # Flatten predictions
        cls_pred = torch.cat([p.permute(0, 2, 3, 1).reshape(-1, self.num_classes) 
                             for p in cls_preds], dim=0)
        bbox_pred = torch.cat([p.permute(0, 2, 3, 1).reshape(-1, 4 * (self.reg_max + 1)) 
                              for p in bbox_preds], dim=0)
        
        # Classification loss
        cls_loss = self.cls_loss(cls_pred, cls_targets) / num_pos
        
        # Box losses (only for positive samples)
        if pos_mask.sum() > 0:
            pos_bbox_pred = bbox_pred[pos_mask]
            pos_bbox_targets = bbox_targets[pos_mask]
            
            # Decode predictions
            pos_bbox_pred_decoded = self._decode_bbox(pos_bbox_pred)
            
            # GIoU loss
            bbox_loss = self.bbox_loss(pos_bbox_pred_decoded, pos_bbox_targets) / num_pos
            
            # DFL loss
            pos_bbox_pred_dfl = pos_bbox_pred.reshape(-1, self.reg_max + 1)
            pos_bbox_targets_dfl = pos_bbox_targets.reshape(-1)
            dfl_loss = self.dfl_loss(pos_bbox_pred_dfl, pos_bbox_targets_dfl) / num_pos
        else:
            bbox_loss = torch.tensor(0.0, device=device)
            dfl_loss = torch.tensor(0.0, device=device)
        
        total_loss = cls_loss + bbox_loss + dfl_loss
        
        return {
            "loss": total_loss,
            "cls_loss": cls_loss,
            "bbox_loss": bbox_loss,
            "dfl_loss": dfl_loss,
            "num_pos": num_pos
        }
    
    def _decode_bbox(self, bbox_pred: torch.Tensor) -> torch.Tensor:
        """Decode bbox predictions from distribution."""
        # Reshape to [N, 4, reg_max+1]
        bbox_pred = bbox_pred.reshape(-1, 4, self.reg_max + 1)
        
        # Softmax and integral
        bbox_pred = torch.softmax(bbox_pred, dim=-1)
        project = torch.linspace(0, self.reg_max, self.reg_max + 1, device=bbox_pred.device)
        bbox_pred = (bbox_pred * project).sum(dim=-1)
        
        return bbox_pred
