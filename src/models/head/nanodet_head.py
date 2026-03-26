"""
NanoDet-Plus Detection Head.
Matches official NanoDet implementation.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional

from ..assignment import DynamicSoftLabelAssigner
from ...utils.box_utils import distance2bbox, bbox2distance, multiclass_nms, bbox_overlaps


class DepthwiseConvModule(nn.Module):
    """Depthwise separable conv block."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        stride: int = 1,
        padding: int = None,
        activation: str = "LeakyReLU",
        **kwargs
    ):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size, stride, padding,
            groups=in_channels, bias=False
        )
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        if activation == "LeakyReLU":
            self.act = nn.LeakyReLU(0.1, inplace=True)
        else:
            self.act = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.act(self.bn1(self.depthwise(x)))
        x = self.act(self.bn2(self.pointwise(x)))
        return x


class Integral(nn.Module):
    """
    Distribution to integral for bounding box regression.
    Converts distribution predictions to distances.
    """
    
    def __init__(self, reg_max: int = 7):
        super().__init__()
        self.reg_max = reg_max
        self.register_buffer("project", torch.linspace(0, reg_max, reg_max + 1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Convert distribution to integral."""
        shape = x.shape
        x = F.softmax(x.reshape(-1, self.reg_max + 1), dim=1)
        x = F.linear(x, self.project.type_as(x))
        x = x.reshape(*shape[:-1], 4)
        return x


class NanoDetPlusHead(nn.Module):
    """
    NanoDet-Plus detection head.
    
    Features:
    - Quality Focal Loss for classification
    - Distribution Focal Loss for regression
    - GIoU Loss for bounding box
    - Dynamic Soft Label Assignment
    
    Args:
        num_classes: Number of detection classes.
        input_channel: Input feature channel.
        feat_channels: Feature channels in head.
        stacked_convs: Number of stacked conv layers.
        kernel_size: Convolution kernel size.
        strides: Feature map strides.
        reg_max: Max value for distribution focal loss.
        activation: Activation function name.
    """
    
    def __init__(
        self,
        num_classes: int = 10,
        input_channel: int = 96,
        feat_channels: int = 96,
        stacked_convs: int = 2,
        kernel_size: int = 5,
        strides: List[int] = [8, 16, 32, 64],
        reg_max: int = 7,
        activation: str = "LeakyReLU",
        loss_cfg: dict = None,
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.strides = strides
        self.reg_max = reg_max
        self.feat_channels = feat_channels
        self.use_sigmoid = True
        
        # Loss weights
        loss_cfg = loss_cfg or {}
        self.loss_qfl_weight = loss_cfg.get("qfl_weight", 1.0)
        self.loss_dfl_weight = loss_cfg.get("dfl_weight", 0.25)
        self.loss_bbox_weight = loss_cfg.get("bbox_weight", 2.0)
        
        # Target assigner
        self.assigner = DynamicSoftLabelAssigner(topk=13, iou_factor=3.0)
        
        # Distribution project
        self.distribution_project = Integral(reg_max)
        
        # Build head layers
        self._init_layers(input_channel, feat_channels, stacked_convs, kernel_size, activation)
        self._init_weights()
    
    def _init_layers(self, input_channel, feat_channels, stacked_convs, kernel_size, activation):
        """Initialize head layers.
        
        Official NanoDet-Plus uses SEPARATE conv layers for each scale (not shared).
        This is the _buid_not_shared_head pattern from the official implementation.
        """
        # Separate conv layers for each scale (matches official NanoDet-Plus)
        self.cls_convs = nn.ModuleList()
        for _ in self.strides:
            convs = nn.ModuleList()
            for i in range(stacked_convs):
                in_ch = input_channel if i == 0 else feat_channels
                convs.append(
                    DepthwiseConvModule(
                        in_ch, feat_channels, kernel_size, activation=activation
                    )
                )
            self.cls_convs.append(convs)
        
        # Separate output layer for each scale (classification + regression)
        self.gfl_cls = nn.ModuleList([
            nn.Conv2d(feat_channels, self.num_classes + 4 * (self.reg_max + 1), 1)
            for _ in self.strides
        ])
    
    def _init_weights(self):
        """Initialize weights."""
        for m in self.cls_convs.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        # Init cls head with confidence = 0.01
        bias_cls = -4.595  # -log((1-0.01)/0.01)
        for gfl_cls in self.gfl_cls:
            nn.init.normal_(gfl_cls.weight, std=0.01)
            nn.init.constant_(gfl_cls.bias, bias_cls)
    
    def forward(self, feats: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            feats: Multi-scale feature maps from FPN.
            
        Returns:
            Predictions tensor [B, num_points, num_classes + 4*(reg_max+1)].
        """
        outputs = []
        for feat, cls_convs, gfl_cls in zip(feats, self.cls_convs, self.gfl_cls):
            # Apply scale-specific conv layers
            for conv in cls_convs:
                feat = conv(feat)
            # Apply scale-specific output layer
            output = gfl_cls(feat)
            outputs.append(output.flatten(start_dim=2))
        
        outputs = torch.cat(outputs, dim=2).permute(0, 2, 1)
        return outputs
    
    def get_single_level_center_priors(
        self,
        batch_size: int,
        featmap_size: Tuple[int, int],
        stride: int,
        dtype: torch.dtype,
        device: torch.device
    ) -> torch.Tensor:
        """Generate center priors for a single feature level.
        
        Official NanoDet uses top-left corner of each grid cell (no +0.5 offset).
        For a grid cell at index (i, j), the point is at:
            x = j * stride
            y = i * stride
        """
        h, w = featmap_size
        # Official NanoDet: no +0.5 offset (top-left corner, not center)
        x_range = torch.arange(w, dtype=dtype, device=device) * stride
        y_range = torch.arange(h, dtype=dtype, device=device) * stride
        y, x = torch.meshgrid(y_range, x_range, indexing="ij")
        y = y.flatten()
        x = x.flatten()
        strides = x.new_full((x.shape[0],), stride)
        priors = torch.stack([x, y, strides, strides], dim=-1)
        return priors.unsqueeze(0).repeat(batch_size, 1, 1)
    
    def loss(
        self,
        preds: torch.Tensor,
        gt_meta: Dict,
        aux_preds: torch.Tensor = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute losses.
        
        Args:
            preds: Prediction tensor [B, num_points, C].
            gt_meta: Ground truth metadata with 'gt_bboxes', 'gt_labels', 'img'.
            aux_preds: Auxiliary head predictions (optional).
            
        Returns:
            Tuple of (total_loss, loss_states_dict).
        """
        device = preds.device
        batch_size = preds.shape[0]
        gt_bboxes_list = gt_meta["gt_bboxes"]
        gt_labels_list = gt_meta["gt_labels"]
        input_height, input_width = gt_meta["img"].shape[2:]
        
        # Generate center priors
        featmap_sizes = [
            (math.ceil(input_height / stride), math.ceil(input_width / stride))
            for stride in self.strides
        ]
        mlvl_center_priors = [
            self.get_single_level_center_priors(
                batch_size, featmap_sizes[i], stride,
                dtype=torch.float32, device=device
            )
            for i, stride in enumerate(self.strides)
        ]
        center_priors = torch.cat(mlvl_center_priors, dim=1)
        
        # Split predictions
        cls_preds, reg_preds = preds.split(
            [self.num_classes, 4 * (self.reg_max + 1)], dim=-1
        )
        
        # Decode boxes
        dis_preds = self.distribution_project(reg_preds) * center_priors[..., 2, None]
        decoded_bboxes = distance2bbox(center_priors[..., :2], dis_preds)
        
        # Target assignment
        batch_assign_res = []
        for i in range(batch_size):
            gt_bboxes = torch.from_numpy(gt_bboxes_list[i]).to(device).float()
            gt_labels = torch.from_numpy(gt_labels_list[i]).to(device).long()
            
            assign_result = self.assigner.assign(
                cls_preds[i].detach(),
                center_priors[i],
                decoded_bboxes[i].detach(),
                gt_bboxes,
                gt_labels
            )
            batch_assign_res.append(assign_result)
        
        # Compute losses
        loss, loss_states = self._compute_loss(
            cls_preds, reg_preds, decoded_bboxes, center_priors, batch_assign_res, gt_bboxes_list
        )
        
        return loss, loss_states
    
    def _compute_loss(
        self,
        cls_preds: torch.Tensor,
        reg_preds: torch.Tensor,
        decoded_bboxes: torch.Tensor,
        center_priors: torch.Tensor,
        assign_results: List,
        gt_bboxes_list: List
    ) -> Tuple[torch.Tensor, Dict]:
        """Compute losses from assignment results."""
        device = cls_preds.device
        batch_size = cls_preds.shape[0]
        
        # Gather targets
        all_labels = []
        all_label_scores = []
        all_bbox_targets = []
        all_dist_targets = []
        num_pos_list = []
        
        for i, assign_result in enumerate(assign_results):
            num_priors = center_priors.shape[1]
            gt_bboxes = torch.from_numpy(gt_bboxes_list[i]).to(device).float()
            
            # Labels
            labels = center_priors.new_full((num_priors,), self.num_classes, dtype=torch.long)
            label_scores = center_priors.new_zeros((num_priors,))
            bbox_targets = center_priors.new_zeros((num_priors, 4))
            dist_targets = center_priors.new_zeros((num_priors, 4))
            
            pos_inds = (assign_result.gt_inds > 0).nonzero(as_tuple=False).squeeze(-1)
            num_pos = pos_inds.numel()
            num_pos_list.append(num_pos)
            
            if num_pos > 0:
                pos_gt_inds = assign_result.gt_inds[pos_inds] - 1
                pos_gt_bboxes = gt_bboxes[pos_gt_inds]
                
                labels[pos_inds] = assign_result.labels[pos_inds]
                label_scores[pos_inds] = assign_result.max_overlaps[pos_inds]
                bbox_targets[pos_inds] = pos_gt_bboxes
                
                # Distance targets: compute distances in pixels, then normalize by stride
                # The max distance in stride units is reg_max, so max in pixels is reg_max * stride
                pos_strides = center_priors[i, pos_inds, 2:3]  # [N, 1]
                max_dis_pixels = self.reg_max * pos_strides.squeeze(-1).max().item()
                
                # Compute distances in pixels (with clipping to valid range)
                raw_distances = bbox2distance(
                    center_priors[i, pos_inds, :2], pos_gt_bboxes, max_dis=None
                )
                
                # Normalize by stride to get targets in [0, reg_max] range
                dist_targets[pos_inds] = raw_distances / pos_strides
                # Clamp to valid range for DFL targets (official uses reg_max - 0.1)
                dist_targets[pos_inds] = dist_targets[pos_inds].clamp(min=0, max=self.reg_max - 0.1)
            
            all_labels.append(labels)
            all_label_scores.append(label_scores)
            all_bbox_targets.append(bbox_targets)
            all_dist_targets.append(dist_targets)
        
        # Stack
        labels = torch.stack(all_labels, dim=0).reshape(-1)
        label_scores = torch.stack(all_label_scores, dim=0).reshape(-1)
        bbox_targets = torch.stack(all_bbox_targets, dim=0).reshape(-1, 4)
        dist_targets = torch.stack(all_dist_targets, dim=0).reshape(-1, 4)
        
        num_total_pos = max(sum(num_pos_list), 1)
        
        # Flatten predictions
        cls_preds = cls_preds.reshape(-1, self.num_classes)
        reg_preds = reg_preds.reshape(-1, 4 * (self.reg_max + 1))
        decoded_bboxes = decoded_bboxes.reshape(-1, 4)
        
        # Quality Focal Loss
        loss_qfl = self._quality_focal_loss(cls_preds, labels, label_scores, num_total_pos)
        
        # Box losses for positive samples
        pos_inds = ((labels >= 0) & (labels < self.num_classes)).nonzero(as_tuple=False).squeeze(-1)
        
        if pos_inds.numel() > 0:
            pos_decoded_bboxes = decoded_bboxes[pos_inds]
            pos_bbox_targets = bbox_targets[pos_inds]
            pos_reg_preds = reg_preds[pos_inds]
            pos_dist_targets = dist_targets[pos_inds]
            
            # Weight by classification score
            weight_targets = cls_preds[pos_inds].detach().sigmoid().max(dim=1)[0]
            bbox_avg_factor = max(weight_targets.sum().item(), 1.0)
            
            # GIoU Loss
            loss_bbox = self._giou_loss(pos_decoded_bboxes, pos_bbox_targets, weight_targets, bbox_avg_factor)
            
            # Distribution Focal Loss
            loss_dfl = self._distribution_focal_loss(
                pos_reg_preds.reshape(-1, self.reg_max + 1),
                pos_dist_targets.reshape(-1),
                weight_targets.unsqueeze(-1).expand(-1, 4).reshape(-1),
                4.0 * bbox_avg_factor
            )
        else:
            loss_bbox = reg_preds.sum() * 0
            loss_dfl = reg_preds.sum() * 0
        
        loss = loss_qfl + loss_bbox + loss_dfl
        
        loss_states = {
            "loss_qfl": loss_qfl.detach(),
            "loss_bbox": loss_bbox.detach(),
            "loss_dfl": loss_dfl.detach(),
            "num_pos": torch.tensor(num_total_pos, device=device)
        }
        
        return loss, loss_states
    
    def _quality_focal_loss(
        self,
        pred: torch.Tensor,
        labels: torch.Tensor,
        scores: torch.Tensor,
        num_total_pos: int
    ) -> torch.Tensor:
        """Quality Focal Loss - matches official implementation.
        
        Key insight: 
        - Negatives are supervised by 0 with scale factor = pred_sigmoid^beta
        - Positives are supervised by IoU score with scale factor = |IoU - pred|^beta
        """
        pred_sigmoid = pred.sigmoid()
        
        # Start with all zeros target (background supervision for all classes)
        zerolabel = pred_sigmoid.new_zeros(pred.shape)
        
        # Background loss with scale factor = pred_sigmoid^beta
        # This down-weights easy negatives (predictions close to 0)
        loss = F.binary_cross_entropy_with_logits(
            pred, zerolabel, reduction='none') * pred_sigmoid.pow(2.0)
        
        # Find positive samples
        pos_mask = (labels >= 0) & (labels < self.num_classes)
        pos_inds = pos_mask.nonzero(as_tuple=False).squeeze(-1)
        
        if pos_inds.numel() > 0:
            pos_labels = labels[pos_inds].long()
            pos_scores = scores[pos_inds]
            
            # For positive samples, compute loss with IoU as target
            # Scale factor = |IoU - pred|^beta for the positive class only
            pos_pred = pred[pos_inds, pos_labels]
            pos_pred_sigmoid = pred_sigmoid[pos_inds, pos_labels]
            scale_factor = (pos_scores - pos_pred_sigmoid).abs().pow(2.0)
            
            # Replace the loss for positive class entries
            loss[pos_inds, pos_labels] = F.binary_cross_entropy_with_logits(
                pos_pred, pos_scores, reduction='none') * scale_factor
        
        # Sum per sample, then sum all and normalize
        loss = loss.sum(dim=1).sum() / max(num_total_pos, 1)
        
        return loss * self.loss_qfl_weight
    
    def _giou_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weight: torch.Tensor,
        avg_factor: float
    ) -> torch.Tensor:
        """GIoU Loss."""
        gious = bbox_overlaps(pred, target, mode="giou", is_aligned=True)
        loss = (1 - gious) * weight
        return loss.sum() / avg_factor * self.loss_bbox_weight
    
    def _distribution_focal_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weight: torch.Tensor,
        avg_factor: float
    ) -> torch.Tensor:
        """Distribution Focal Loss."""
        target_left = target.long()
        target_right = target_left + 1
        
        target_left = target_left.clamp(0, self.reg_max)
        target_right = target_right.clamp(0, self.reg_max)
        
        weight_left = target_right.float() - target
        weight_right = target - target_left.float()
        
        loss_left = F.cross_entropy(pred, target_left, reduction="none") * weight_left
        loss_right = F.cross_entropy(pred, target_right, reduction="none") * weight_right
        
        loss = (loss_left + loss_right) * weight
        return loss.sum() / avg_factor * self.loss_dfl_weight
    
    def get_bboxes(
        self,
        preds: torch.Tensor,
        img_metas: Dict,
        score_thr: float = 0.05,  # Official NanoDet uses 0.05 for NMS
        nms_thr: float = 0.6
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Decode predictions to bboxes.
        
        Args:
            preds: Predictions [B, num_points, C].
            img_metas: Image metadata.
            score_thr: Score threshold.
            nms_thr: NMS threshold.
            
        Returns:
            List of (det_bboxes, det_labels) per image.
        """
        device = preds.device
        batch_size = preds.shape[0]
        input_height, input_width = img_metas["img"].shape[2:]
        input_shape = (input_height, input_width)
        
        # Generate center priors
        featmap_sizes = [
            (math.ceil(input_height / stride), math.ceil(input_width / stride))
            for stride in self.strides
        ]
        mlvl_center_priors = [
            self.get_single_level_center_priors(
                batch_size, featmap_sizes[i], stride,
                dtype=torch.float32, device=device
            )
            for i, stride in enumerate(self.strides)
        ]
        center_priors = torch.cat(mlvl_center_priors, dim=1)
        
        # Split predictions
        cls_preds, reg_preds = preds.split(
            [self.num_classes, 4 * (self.reg_max + 1)], dim=-1
        )
        
        # Decode boxes
        dis_preds = self.distribution_project(reg_preds) * center_priors[..., 2, None]
        bboxes = distance2bbox(center_priors[..., :2], dis_preds, max_shape=input_shape)
        scores = cls_preds.sigmoid()
        
        # NMS per image
        result_list = []
        for i in range(batch_size):
            # Add dummy background class
            score = scores[i]
            bbox = bboxes[i]
            padding = score.new_zeros(score.shape[0], 1)
            score = torch.cat([score, padding], dim=1)
            
            dets, labels = multiclass_nms(bbox, score, score_thr, nms_thr, max_num=100)
            result_list.append((dets, labels))
        
        return result_list
