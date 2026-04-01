"""
Configuration for NanoDet-Plus-Lite Model.

Default class names are set for the Construction Site Safety / PPE
dataset (10 classes).  train.py reads them automatically from the
annotation JSON, so this is only a fallback.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class DataConfig:
    """Dataset paths — point to your COCO-format data directory.

    Defaults use the included demo dataset (data/demo/) so training works
    out of the box.  Override with your own data/coco/ paths for full training.
    """
    train_images: str = "data/demo/train"
    train_annotations: str = "data/demo/train/_annotations.coco.json"
    val_images: str = "data/demo/valid"
    val_annotations: str = "data/demo/valid/_annotations.coco.json"
    test_images: str = "data/demo/valid"
    test_annotations: str = "data/demo/valid/_annotations.coco.json"
    num_workers: int = 4


@dataclass
class ModelConfig:
    """Model architecture configuration.
    
    Official NanoDet-Plus model specifications:
    - NanoDet-Plus-m:      backbone=1.0x, fpn=96,  ~1.17M params, 2.3MB FP16
    - NanoDet-Plus-m-1.5x: backbone=1.5x, fpn=128, ~2.44M params, 4.7MB FP16
    - NanoDet-Plus-m-0.5x: backbone=0.5x, fpn=96,  ~0.95M params (ultra-lite)
    """
    name: str = "NanoDetPlusLite"
    num_classes: int = 10
    input_size: Tuple[int, int] = (320, 320)
    
    # Backbone: 1.0x for NanoDet-Plus-m, 1.5x for m-1.5x, 0.5x for m-0.5x
    backbone: str = "ShuffleNetV2"
    backbone_size: str = "1.0x"  # Default matches official NanoDet-Plus-m
    backbone_pretrained: bool = True
    
    # FPN (96 for m, 128 for m-1.5x)
    fpn_in_channels: List[int] = field(default_factory=lambda: [116, 232, 464])
    fpn_out_channels: int = 96
    
    # Head
    head_channels: int = 96
    stacked_convs: int = 2
    strides: List[int] = field(default_factory=lambda: [8, 16, 32, 64])
    reg_max: int = 7
    
    # Loss weights
    loss_qfl_weight: float = 1.0
    loss_dfl_weight: float = 0.25
    loss_bbox_weight: float = 2.0


@dataclass
class TrainConfig:
    """Training hyperparameters."""
    epochs: int = 300
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 0.05
    warmup_epochs: int = 5
    grad_clip: float = 35.0
    # Validate every N epochs.  5 is a good balance: frequent enough to track
    # mAP improvements without making short runs very slow.
    val_interval: int = 5
    save_dir: str = "workspace/ppe_detector"
    resume: Optional[str] = None


@dataclass
class AugmentConfig:
    """Data augmentation configuration."""
    scale: Tuple[float, float] = (0.6, 1.4)
    stretch: Tuple[Tuple[float, float], Tuple[float, float]] = ((0.8, 1.2), (0.8, 1.2))
    flip_prob: float = 0.5
    brightness: float = 0.2
    contrast: Tuple[float, float] = (0.6, 1.4)
    saturation: Tuple[float, float] = (0.5, 1.2)
    normalize_mean: List[float] = field(default_factory=lambda: [123.675, 116.28, 103.53])  # RGB
    normalize_std: List[float] = field(default_factory=lambda: [58.395, 57.12, 57.375])     # RGB


@dataclass
class Config:
    """Top-level configuration."""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    augment: AugmentConfig = field(default_factory=AugmentConfig)

    # Indoor Objects Detection classes (alphabetically sorted, matching the
    # category_id order produced by scripts/download_indoor_dataset.py).
    # train.py overwrites this at runtime by reading the annotation JSON,
    # so changing this list only affects the fallback / test.py default.
    class_names: List[str] = field(default_factory=lambda: ["Hardhat", "Mask", "NO-Hardhat", "NO-Mask", "NO-Safety Vest", "Person", "Safety Cone", "Safety Vest", "machinery", "vehicle"])


def get_config() -> Config:
    """Return default configuration."""
    return Config()
