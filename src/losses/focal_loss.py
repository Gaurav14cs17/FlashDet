"""
Focal Loss variants for object detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class QualityFocalLoss(nn.Module):
    """
    Quality Focal Loss for classification with soft labels.
    
    Combines focal loss with quality estimation for better localization.
    
    Args:
        beta: Focal loss parameter
        loss_weight: Loss weight multiplier
    """
    
    def __init__(self, beta: float = 2.0, loss_weight: float = 1.0):
        super().__init__()
        self.beta = beta
        self.loss_weight = loss_weight
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weight: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Compute quality focal loss.
        
        Args:
            pred: Predictions [N, C] (logits)
            target: Soft targets [N, C] (0-1 quality scores)
            weight: Optional sample weights [N]
            
        Returns:
            Loss value
        """
        pred_sigmoid = pred.sigmoid()
        
        # Scale factor: |target - pred|^beta
        scale_factor = (target - pred_sigmoid).abs().pow(self.beta)
        
        # Binary cross entropy
        bce = F.binary_cross_entropy_with_logits(
            pred, target, reduction="none"
        )
        
        loss = scale_factor * bce
        
        if weight is not None:
            loss = loss * weight.unsqueeze(-1)
        
        return loss.sum() * self.loss_weight


class DistributionFocalLoss(nn.Module):
    """
    Distribution Focal Loss for bounding box regression.
    
    Learns a discrete distribution over box offsets instead of 
    directly regressing values.
    
    Note: Official NanoDet does NOT clamp target indices here.
    The assigner already clamps targets to [0, reg_max - 0.1].
    
    Args:
        loss_weight: Loss weight multiplier
    """
    
    def __init__(self, loss_weight: float = 0.25):
        super().__init__()
        self.loss_weight = loss_weight
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weight: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Compute distribution focal loss.
        
        Args:
            pred: Predictions [N, reg_max+1]
            target: Target values [N] (continuous, assumed already clamped by assigner)
            weight: Optional sample weights [N]
            
        Returns:
            Loss value
        """
        # Discretize target (no clamping - official relies on assigner to clamp)
        target_left = target.long()
        target_right = target_left + 1
        
        # Interpolation weights for soft two-hot encoding
        weight_left = target_right.float() - target
        weight_right = target - target_left.float()
        
        # Cross entropy for left and right bins
        loss_left = F.cross_entropy(
            pred, target_left, reduction="none"
        ) * weight_left
        
        loss_right = F.cross_entropy(
            pred, target_right, reduction="none"
        ) * weight_right
        
        loss = loss_left + loss_right
        
        if weight is not None:
            loss = loss * weight
        
        return loss.sum() * self.loss_weight


class FocalLoss(nn.Module):
    """
    Standard Focal Loss for classification.
    
    Args:
        alpha: Weighting factor for rare classes
        gamma: Focusing parameter
        loss_weight: Loss weight multiplier
    """
    
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        loss_weight: float = 1.0
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.loss_weight = loss_weight
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weight: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Compute focal loss.
        
        Args:
            pred: Predictions [N, C] (logits)
            target: Target class indices [N]
            weight: Optional sample weights [N]
            
        Returns:
            Loss value
        """
        pred_sigmoid = pred.sigmoid()
        
        # Get probabilities for target class
        target_onehot = F.one_hot(target, pred.shape[-1]).float()
        pt = (pred_sigmoid * target_onehot).sum(dim=-1) + \
             ((1 - pred_sigmoid) * (1 - target_onehot)).sum(dim=-1)
        
        # Focal weight
        focal_weight = (1 - pt).pow(self.gamma)
        
        # Alpha weighting
        alpha_weight = self.alpha * target_onehot + (1 - self.alpha) * (1 - target_onehot)
        alpha_weight = alpha_weight.sum(dim=-1)
        
        # Binary cross entropy
        bce = F.binary_cross_entropy_with_logits(
            pred, target_onehot, reduction="none"
        ).sum(dim=-1)
        
        loss = focal_weight * alpha_weight * bce
        
        if weight is not None:
            loss = loss * weight
        
        return loss.sum() * self.loss_weight
