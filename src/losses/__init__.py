from .focal_loss import QualityFocalLoss, DistributionFocalLoss
from .iou_loss import GIoULoss, IoULoss

__all__ = [
    "QualityFocalLoss",
    "DistributionFocalLoss",
    "GIoULoss",
    "IoULoss",
]
