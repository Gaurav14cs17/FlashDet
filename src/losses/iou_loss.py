"""
IoU-based losses for bounding box regression.
"""

import torch
import torch.nn as nn


class IoULoss(nn.Module):
    """
    IoU Loss for bounding box regression.
    
    Official NanoDet uses -log(IoU) formulation, not 1-IoU.
    
    Args:
        loss_weight: Loss weight multiplier
        eps: Small value to avoid division by zero
    """
    
    def __init__(self, loss_weight: float = 1.0, eps: float = 1e-6):
        super().__init__()
        self.loss_weight = loss_weight
        self.eps = eps
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weight: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Compute IoU loss using -log(IoU) formulation (matches official NanoDet).
        
        Args:
            pred: Predicted boxes [N, 4] (x1, y1, x2, y2)
            target: Target boxes [N, 4] (x1, y1, x2, y2)
            weight: Optional sample weights [N]
            
        Returns:
            Loss value
        """
        # Intersection
        inter_x1 = torch.max(pred[:, 0], target[:, 0])
        inter_y1 = torch.max(pred[:, 1], target[:, 1])
        inter_x2 = torch.min(pred[:, 2], target[:, 2])
        inter_y2 = torch.min(pred[:, 3], target[:, 3])
        
        inter_w = (inter_x2 - inter_x1).clamp(min=0)
        inter_h = (inter_y2 - inter_y1).clamp(min=0)
        inter_area = inter_w * inter_h
        
        # Areas
        pred_area = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
        target_area = (target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1])
        
        # Union (use max for stability like official)
        union_area = pred_area + target_area - inter_area
        union_area = torch.max(union_area, torch.tensor(self.eps, device=union_area.device))
        
        # IoU
        iou = inter_area / union_area
        
        # Official NanoDet uses -log(IoU) loss
        loss = -torch.log(iou.clamp(min=self.eps))
        
        if weight is not None:
            loss = loss * weight
        
        return loss.sum() * self.loss_weight


class GIoULoss(nn.Module):
    """
    Generalized IoU Loss for bounding box regression.
    
    GIoU = IoU - (C - Union) / C
    where C is the smallest enclosing box.
    
    Args:
        loss_weight: Loss weight multiplier
        eps: Small value to avoid division by zero
    """
    
    def __init__(self, loss_weight: float = 2.0, eps: float = 1e-6):
        super().__init__()
        self.loss_weight = loss_weight
        self.eps = eps
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weight: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Compute GIoU loss.
        
        Args:
            pred: Predicted boxes [N, 4] (x1, y1, x2, y2)
            target: Target boxes [N, 4] (x1, y1, x2, y2)
            weight: Optional sample weights [N]
            
        Returns:
            Loss value
        """
        # Intersection
        inter_x1 = torch.max(pred[:, 0], target[:, 0])
        inter_y1 = torch.max(pred[:, 1], target[:, 1])
        inter_x2 = torch.min(pred[:, 2], target[:, 2])
        inter_y2 = torch.min(pred[:, 3], target[:, 3])
        
        inter_w = (inter_x2 - inter_x1).clamp(min=0)
        inter_h = (inter_y2 - inter_y1).clamp(min=0)
        inter_area = inter_w * inter_h
        
        # Areas
        pred_area = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
        target_area = (target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1])
        
        # Union
        union_area = pred_area + target_area - inter_area + self.eps
        
        # IoU
        iou = inter_area / union_area
        
        # Enclosing box
        enclose_x1 = torch.min(pred[:, 0], target[:, 0])
        enclose_y1 = torch.min(pred[:, 1], target[:, 1])
        enclose_x2 = torch.max(pred[:, 2], target[:, 2])
        enclose_y2 = torch.max(pred[:, 3], target[:, 3])
        
        enclose_area = (enclose_x2 - enclose_x1) * (enclose_y2 - enclose_y1) + self.eps
        
        # GIoU
        giou = iou - (enclose_area - union_area) / enclose_area
        
        loss = 1 - giou
        
        if weight is not None:
            loss = loss * weight
        
        return loss.sum() * self.loss_weight


class DIoULoss(nn.Module):
    """
    Distance IoU Loss for bounding box regression.
    
    DIoU = IoU - d^2 / c^2
    where d is center distance and c is diagonal of enclosing box.
    
    Args:
        loss_weight: Loss weight multiplier
        eps: Small value to avoid division by zero
    """
    
    def __init__(self, loss_weight: float = 2.0, eps: float = 1e-6):
        super().__init__()
        self.loss_weight = loss_weight
        self.eps = eps
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weight: torch.Tensor = None
    ) -> torch.Tensor:
        """Compute DIoU loss."""
        # Intersection
        inter_x1 = torch.max(pred[:, 0], target[:, 0])
        inter_y1 = torch.max(pred[:, 1], target[:, 1])
        inter_x2 = torch.min(pred[:, 2], target[:, 2])
        inter_y2 = torch.min(pred[:, 3], target[:, 3])
        
        inter_w = (inter_x2 - inter_x1).clamp(min=0)
        inter_h = (inter_y2 - inter_y1).clamp(min=0)
        inter_area = inter_w * inter_h
        
        # Areas
        pred_area = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
        target_area = (target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1])
        
        # Union
        union_area = pred_area + target_area - inter_area + self.eps
        iou = inter_area / union_area
        
        # Center distance
        pred_cx = (pred[:, 0] + pred[:, 2]) / 2
        pred_cy = (pred[:, 1] + pred[:, 3]) / 2
        target_cx = (target[:, 0] + target[:, 2]) / 2
        target_cy = (target[:, 1] + target[:, 3]) / 2
        
        center_dist = (pred_cx - target_cx) ** 2 + (pred_cy - target_cy) ** 2
        
        # Enclosing box diagonal
        enclose_x1 = torch.min(pred[:, 0], target[:, 0])
        enclose_y1 = torch.min(pred[:, 1], target[:, 1])
        enclose_x2 = torch.max(pred[:, 2], target[:, 2])
        enclose_y2 = torch.max(pred[:, 3], target[:, 3])
        
        enclose_diag = (enclose_x2 - enclose_x1) ** 2 + (enclose_y2 - enclose_y1) ** 2 + self.eps
        
        # DIoU
        diou = iou - center_dist / enclose_diag
        
        loss = 1 - diou
        
        if weight is not None:
            loss = loss * weight
        
        return loss.sum() * self.loss_weight
