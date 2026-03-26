"""
Box utility functions for NanoDet.
"""

import torch
import torch.nn.functional as F


def distance2bbox(points, distance, max_shape=None):
    """
    Decode distance prediction to bounding box.

    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom).
        max_shape (tuple): Shape of the image (height, width).

    Returns:
        Tensor: Decoded bboxes [x1, y1, x2, y2].
    """
    x1 = points[..., 0] - distance[..., 0]
    y1 = points[..., 1] - distance[..., 1]
    x2 = points[..., 0] + distance[..., 2]
    y2 = points[..., 1] + distance[..., 3]
    
    if max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape[1])
        y1 = y1.clamp(min=0, max=max_shape[0])
        x2 = x2.clamp(min=0, max=max_shape[1])
        y2 = y2.clamp(min=0, max=max_shape[0])
    
    bboxes = torch.stack([x1, y1, x2, y2], -1)
    return bboxes


def bbox2distance(points, bbox, max_dis=None, eps=0.1):
    """
    Encode bounding box to distances.

    Args:
        points (Tensor): Shape (n, 2), [x, y].
        bbox (Tensor): Shape (n, 4), "xyxy" format.
        max_dis (float): Upper bound of the distance.
        eps (float): Small value to ensure target < max_dis.

    Returns:
        Tensor: Encoded distances [left, top, right, bottom].
    """
    left = points[:, 0] - bbox[:, 0]
    top = points[:, 1] - bbox[:, 1]
    right = bbox[:, 2] - points[:, 0]
    bottom = bbox[:, 3] - points[:, 1]
    
    if max_dis is not None:
        left = left.clamp(min=0, max=max_dis - eps)
        top = top.clamp(min=0, max=max_dis - eps)
        right = right.clamp(min=0, max=max_dis - eps)
        bottom = bottom.clamp(min=0, max=max_dis - eps)
    
    return torch.stack([left, top, right, bottom], -1)


def bbox_overlaps(bboxes1, bboxes2, mode="iou", is_aligned=False, eps=1e-6):
    """
    Calculate overlap between two sets of bboxes.

    Args:
        bboxes1 (Tensor): Shape (m, 4) in <x1, y1, x2, y2> format.
        bboxes2 (Tensor): Shape (n, 4) in <x1, y1, x2, y2> format.
        mode (str): "iou", "iof", or "giou".
        is_aligned (bool): If True, m must equal n.
        eps (float): Small value for numerical stability.

    Returns:
        Tensor: Shape (m, n) if not aligned, else (m,).
    """
    assert mode in ["iou", "iof", "giou"]
    
    rows = bboxes1.size(0)
    cols = bboxes2.size(0)
    
    if rows * cols == 0:
        return bboxes1.new_zeros((rows, cols)) if not is_aligned else bboxes1.new_zeros((rows,))
    
    if is_aligned:
        assert rows == cols
        
        lt = torch.max(bboxes1[:, :2], bboxes2[:, :2])
        rb = torch.min(bboxes1[:, 2:], bboxes2[:, 2:])
        wh = (rb - lt).clamp(min=0)
        overlap = wh[:, 0] * wh[:, 1]
        
        area1 = (bboxes1[:, 2] - bboxes1[:, 0]) * (bboxes1[:, 3] - bboxes1[:, 1])
        area2 = (bboxes2[:, 2] - bboxes2[:, 0]) * (bboxes2[:, 3] - bboxes2[:, 1])
        
        if mode == "iof":
            union = area1
        else:
            union = area1 + area2 - overlap
        
        union = torch.clamp(union, min=eps)
        ious = overlap / union
        
        if mode == "giou":
            enclosed_lt = torch.min(bboxes1[:, :2], bboxes2[:, :2])
            enclosed_rb = torch.max(bboxes1[:, 2:], bboxes2[:, 2:])
            enclosed_wh = (enclosed_rb - enclosed_lt).clamp(min=0)
            enclosed_area = enclosed_wh[:, 0] * enclosed_wh[:, 1]
            enclosed_area = torch.clamp(enclosed_area, min=eps)
            ious = ious - (enclosed_area - union) / enclosed_area
        
        return ious
    else:
        lt = torch.max(bboxes1[:, None, :2], bboxes2[None, :, :2])
        rb = torch.min(bboxes1[:, None, 2:], bboxes2[None, :, 2:])
        wh = (rb - lt).clamp(min=0)
        overlap = wh[:, :, 0] * wh[:, :, 1]
        
        area1 = (bboxes1[:, 2] - bboxes1[:, 0]) * (bboxes1[:, 3] - bboxes1[:, 1])
        area2 = (bboxes2[:, 2] - bboxes2[:, 0]) * (bboxes2[:, 3] - bboxes2[:, 1])
        
        if mode == "iof":
            union = area1[:, None]
        else:
            union = area1[:, None] + area2[None, :] - overlap
        
        union = torch.clamp(union, min=eps)
        ious = overlap / union
        
        if mode == "giou":
            enclosed_lt = torch.min(bboxes1[:, None, :2], bboxes2[None, :, :2])
            enclosed_rb = torch.max(bboxes1[:, None, 2:], bboxes2[None, :, 2:])
            enclosed_wh = (enclosed_rb - enclosed_lt).clamp(min=0)
            enclosed_area = enclosed_wh[:, :, 0] * enclosed_wh[:, :, 1]
            enclosed_area = torch.clamp(enclosed_area, min=eps)
            ious = ious - (enclosed_area - union) / enclosed_area
        
        return ious


def multiclass_nms(boxes, scores, score_thr=0.05, nms_thr=0.6, max_num=100, exclude_last_class=True):
    """
    Multi-class NMS.

    Args:
        boxes (Tensor): Shape (n, 4).
        scores (Tensor): Shape (n, num_classes) or (n, num_classes+1) if background included.
        score_thr (float): Score threshold.
        nms_thr (float): NMS IoU threshold.
        max_num (int): Maximum number of detections.
        exclude_last_class (bool): If True, exclude the last class (assumed background).

    Returns:
        Tuple: (boxes, labels) where boxes is (k, 5) with scores.
    """
    num_classes = scores.size(1)
    
    # Exclude background class (last class) if specified
    if exclude_last_class and num_classes > 1:
        scores = scores[:, :-1]
        num_classes -= 1
    
    # Get max score and class per box
    max_scores, labels = scores.max(dim=1)
    
    # Filter by score threshold
    valid_mask = max_scores > score_thr
    boxes = boxes[valid_mask]
    scores_filtered = max_scores[valid_mask]
    labels = labels[valid_mask]
    
    if boxes.numel() == 0:
        return torch.zeros((0, 5), device=boxes.device), torch.zeros((0,), dtype=torch.long, device=boxes.device)
    
    # NMS per class
    keep_boxes = []
    keep_scores = []
    keep_labels = []
    
    for cls_id in range(num_classes):
        cls_mask = labels == cls_id
        if not cls_mask.any():
            continue
        
        cls_boxes = boxes[cls_mask]
        cls_scores = scores_filtered[cls_mask]
        
        # Sort by score
        _, order = cls_scores.sort(descending=True)
        cls_boxes = cls_boxes[order]
        cls_scores = cls_scores[order]
        
        # NMS
        keep = _nms(cls_boxes, cls_scores, nms_thr)
        keep_boxes.append(cls_boxes[keep])
        keep_scores.append(cls_scores[keep])
        keep_labels.append(torch.full((keep.sum(),), cls_id, dtype=torch.long, device=boxes.device))
    
    if len(keep_boxes) == 0:
        return torch.zeros((0, 5), device=boxes.device), torch.zeros((0,), dtype=torch.long, device=boxes.device)
    
    # Concatenate
    boxes_out = torch.cat(keep_boxes, dim=0)
    scores_out = torch.cat(keep_scores, dim=0)
    labels_out = torch.cat(keep_labels, dim=0)
    
    # Sort by score and limit
    _, order = scores_out.sort(descending=True)
    order = order[:max_num]
    
    boxes_out = boxes_out[order]
    scores_out = scores_out[order]
    labels_out = labels_out[order]
    
    # Combine boxes and scores
    dets = torch.cat([boxes_out, scores_out.unsqueeze(1)], dim=1)
    
    return dets, labels_out


def _nms(boxes, scores, thresh):
    """Simple NMS implementation."""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    
    keep = torch.zeros(len(boxes), dtype=torch.bool, device=boxes.device)
    order = torch.arange(len(boxes), device=boxes.device)
    
    while order.numel() > 0:
        i = order[0]
        keep[i] = True
        
        if order.numel() == 1:
            break
        
        xx1 = torch.clamp(x1[order[1:]], min=x1[i].item())
        yy1 = torch.clamp(y1[order[1:]], min=y1[i].item())
        xx2 = torch.clamp(x2[order[1:]], max=x2[i].item())
        yy2 = torch.clamp(y2[order[1:]], max=y2[i].item())
        
        w = torch.clamp(xx2 - xx1, min=0)
        h = torch.clamp(yy2 - yy1, min=0)
        inter = w * h
        
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        mask = iou <= thresh
        order = order[1:][mask]
    
    return keep
