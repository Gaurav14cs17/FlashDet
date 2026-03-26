"""
Dynamic Soft Label Assigner for NanoDet-Plus.
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional

from ...utils.box_utils import bbox_overlaps


@dataclass
class AssignResult:
    """Assignment result container."""
    num_gts: int
    gt_inds: torch.Tensor
    max_overlaps: torch.Tensor
    labels: Optional[torch.Tensor] = None


class DynamicSoftLabelAssigner:
    """
    Dynamic Soft Label Assigner for target assignment.
    
    Uses OTA (Optimal Transport Assignment) style dynamic k matching.
    
    Args:
        topk (int): Top-k predictions to calculate dynamic k. Default: 13.
        iou_factor (float): IoU cost factor. Default: 3.0.
    """
    
    def __init__(self, topk: int = 13, iou_factor: float = 3.0):
        self.topk = topk
        self.iou_factor = iou_factor
    
    def assign(
        self,
        pred_scores: torch.Tensor,
        priors: torch.Tensor,
        decoded_bboxes: torch.Tensor,
        gt_bboxes: torch.Tensor,
        gt_labels: torch.Tensor,
    ) -> AssignResult:
        """
        Assign ground truths to predictions.
        
        Args:
            pred_scores: Classification scores [num_priors, num_classes].
            priors: Prior points [num_priors, 4] in [cx, cy, stride_w, stride_h].
            decoded_bboxes: Predicted boxes [num_priors, 4].
            gt_bboxes: Ground truth boxes [num_gts, 4].
            gt_labels: Ground truth labels [num_gts].
            
        Returns:
            AssignResult with assignment information.
        """
        INF = 100000000
        device = pred_scores.device
        num_gt = gt_bboxes.size(0)
        num_bboxes = decoded_bboxes.size(0)
        
        # Initialize assignments
        assigned_gt_inds = decoded_bboxes.new_full((num_bboxes,), 0, dtype=torch.long)
        
        if num_gt == 0 or num_bboxes == 0:
            max_overlaps = decoded_bboxes.new_zeros((num_bboxes,))
            assigned_labels = decoded_bboxes.new_full((num_bboxes,), -1, dtype=torch.long)
            return AssignResult(num_gt, assigned_gt_inds, max_overlaps, assigned_labels)
        
        # Check which priors are inside gt boxes (official NanoDet logic)
        prior_center = priors[:, :2]
        
        # Prior center must be inside GT box
        lt_ = prior_center[:, None] - gt_bboxes[:, :2]
        rb_ = gt_bboxes[:, 2:] - prior_center[:, None]
        deltas = torch.cat([lt_, rb_], dim=-1)
        is_in_gts = deltas.min(dim=-1).values > 0
        
        # Valid priors are those inside at least one GT
        valid_mask = is_in_gts.sum(dim=1) > 0
        
        valid_decoded_bbox = decoded_bboxes[valid_mask]
        valid_pred_scores = pred_scores[valid_mask]
        num_valid = valid_decoded_bbox.size(0)
        
        if num_valid == 0:
            max_overlaps = decoded_bboxes.new_zeros((num_bboxes,))
            assigned_labels = decoded_bboxes.new_full((num_bboxes,), -1, dtype=torch.long)
            return AssignResult(num_gt, assigned_gt_inds, max_overlaps, assigned_labels)
        
        # Calculate IoU cost
        pairwise_ious = bbox_overlaps(valid_decoded_bbox, gt_bboxes)
        iou_cost = -torch.log(pairwise_ious + 1e-7)
        
        # Calculate classification cost
        num_classes = pred_scores.shape[-1]
        gt_onehot_label = F.one_hot(gt_labels.long(), num_classes).float()
        gt_onehot_label = gt_onehot_label.unsqueeze(0).repeat(num_valid, 1, 1)
        
        valid_pred_scores_expanded = valid_pred_scores.unsqueeze(1).repeat(1, num_gt, 1)
        
        soft_label = gt_onehot_label * pairwise_ious[..., None]
        scale_factor = soft_label - valid_pred_scores_expanded.sigmoid()
        
        cls_cost = F.binary_cross_entropy_with_logits(
            valid_pred_scores_expanded, soft_label, reduction="none"
        ) * scale_factor.abs().pow(2.0)
        cls_cost = cls_cost.sum(dim=-1)
        
        # Total cost
        cost_matrix = cls_cost + iou_cost * self.iou_factor
        
        # Dynamic k matching
        matched_pred_ious, matched_gt_inds = self._dynamic_k_matching(
            cost_matrix, pairwise_ious, num_gt, valid_mask
        )
        
        # Convert to assignment result
        assigned_gt_inds[valid_mask] = matched_gt_inds + 1
        assigned_labels = assigned_gt_inds.new_full((num_bboxes,), -1)
        assigned_labels[valid_mask] = gt_labels[matched_gt_inds].long()
        
        max_overlaps = assigned_gt_inds.new_full((num_bboxes,), -INF, dtype=torch.float32)
        max_overlaps[valid_mask] = matched_pred_ious
        
        return AssignResult(num_gt, assigned_gt_inds, max_overlaps, assigned_labels)
    
    def _dynamic_k_matching(
        self,
        cost: torch.Tensor,
        pairwise_ious: torch.Tensor,
        num_gt: int,
        valid_mask: torch.Tensor
    ):
        """
        Dynamic k matching using OTA style.
        
        Args:
            cost: Cost matrix [num_valid, num_gt].
            pairwise_ious: IoU matrix [num_valid, num_gt].
            num_gt: Number of ground truths.
            valid_mask: Valid prior mask.
        """
        matching_matrix = torch.zeros_like(cost)
        
        # Select candidate topk ious for dynamic-k calculation
        candidate_topk = min(self.topk, pairwise_ious.size(0))
        topk_ious, _ = torch.topk(pairwise_ious, candidate_topk, dim=0)
        
        # Calculate dynamic k for each gt
        dynamic_ks = torch.clamp(topk_ious.sum(0).int(), min=1)
        
        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(
                cost[:, gt_idx], k=dynamic_ks[gt_idx].item(), largest=False
            )
            matching_matrix[:, gt_idx][pos_idx] = 1.0
        
        # Handle priors matched to multiple gts
        prior_match_gt_mask = matching_matrix.sum(1) > 1
        if prior_match_gt_mask.sum() > 0:
            _, cost_argmin = torch.min(cost[prior_match_gt_mask, :], dim=1)
            matching_matrix[prior_match_gt_mask, :] *= 0.0
            matching_matrix[prior_match_gt_mask, cost_argmin] = 1.0
        
        # Get foreground mask
        fg_mask_inboxes = matching_matrix.sum(1) > 0.0
        valid_mask[valid_mask.clone()] = fg_mask_inboxes
        
        matched_gt_inds = matching_matrix[fg_mask_inboxes, :].argmax(1)
        matched_pred_ious = (matching_matrix * pairwise_ious).sum(1)[fg_mask_inboxes]
        
        return matched_pred_ious, matched_gt_inds
