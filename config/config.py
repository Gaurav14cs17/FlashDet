"""
Configuration for NanoDet-Plus-Lite Model.
"""

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class DataConfig:
    """Dataset configuration."""
    train_images: str = "dataset_coco/train"
    train_annotations: str = "dataset_coco/train/_annotations.coco.json"
    val_images: str = "dataset_coco/valid"
    val_annotations: str = "dataset_coco/valid/_annotations.coco.json"
    test_images: str = "dataset_coco/test"
    test_annotations: str = "dataset_coco/test/_annotations.coco.json"
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
    """Training configuration."""
    epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 0.001
    weight_decay: float = 0.05
    warmup_steps: int = 500
    grad_clip: float = 35.0
    val_interval: int = 10
    save_dir: str = "workspace/ppe_detector"
    resume: str = None


@dataclass
class AugmentConfig:
    """Data augmentation configuration."""
    scale: Tuple[float, float] = (0.6, 1.4)
    stretch: Tuple[Tuple[float, float], Tuple[float, float]] = ((0.8, 1.2), (0.8, 1.2))
    flip_prob: float = 0.5
    brightness: float = 0.2
    contrast: Tuple[float, float] = (0.6, 1.4)
    saturation: Tuple[float, float] = (0.5, 1.2)
    normalize_mean: List[float] = field(default_factory=lambda: [123.675, 116.28, 103.53])  # RGB order
    normalize_std: List[float] = field(default_factory=lambda: [58.395, 57.12, 57.375])    # RGB order


@dataclass
class Config:
    """Main configuration."""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    augment: AugmentConfig = field(default_factory=AugmentConfig)
    
    # Class names
    class_names: List[str] = field(default_factory=lambda: [
        "Hardhat", "Mask", "NO-Hardhat", "NO-Mask", "NO-Safety Vest",
        "Person", "Safety Cone", "Safety Vest", "machinery", "vehicle"
    ])


def get_config() -> Config:
    """Get default configuration."""
    return Config()
