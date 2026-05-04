"""
Configuration for FlashDet Model.

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
    out of the box.  Override with your own paths for full training.
    """
    train_images: str = "data/indoor/train"
    train_annotations: str = "data/indoor/train/_annotations.coco.json"
    val_images: str = "data/indoor/valid"
    val_annotations: str = "data/indoor/valid/_annotations.coco.json"
    test_images: str = "data/indoor/test"
    test_annotations: str = "data/indoor/test/_annotations.coco.json"
    num_workers: int = 4


@dataclass
class ModelConfig:
    """Model architecture configuration.
    
    Official FlashDet model specifications:
    - FlashDet-m:      backbone=1.0x, fpn=96,  ~1.17M params, 2.3MB FP16
    - FlashDet-m-1.5x: backbone=1.5x, fpn=128, ~2.44M params, 4.7MB FP16
    - FlashDet-m-0.5x: backbone=0.5x, fpn=96,  ~0.49M params, ~0.9MB FP16 (ultra-lite)
    """
    name: str = "FlashDet"
    num_classes: int = 10
    input_size: Tuple[int, int] = (320, 320)
    
    # Backbone: 1.0x for FlashDet-m, 1.5x for m-1.5x, 0.5x for m-0.5x
    backbone: str = "ShuffleNetV2"
    backbone_size: str = "1.0x"  # Default matches official FlashDet-m
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
    save_dir: str = "workspace/default_experiment"
    resume: Optional[str] = None

    # --- torchtune-inspired memory & performance optimizations ---
    enable_activation_checkpointing: bool = False
    enable_activation_offloading: bool = False
    optimizer_in_bwd: bool = False
    use_8bit_optimizer: bool = False
    compile_model: bool = False
    chunked_cross_entropy: bool = False
    ce_chunk_size: int = 1024

    # --- LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning ---
    use_lora: bool = False
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: ["backbone"])

    # --- QLoRA (Quantized LoRA) ---
    use_qlora: bool = False
    qlora_quant_dtype: str = "int8"   # "int8" or "nf4"

    # --- Knowledge Distillation (torchtune-style) ---
    use_kd: bool = False
    kd_teacher_checkpoint: Optional[str] = None
    kd_teacher_model_size: str = "m-1.5x"
    kd_temperature: float = 4.0
    kd_logit_weight: float = 1.0
    kd_feature_weight: float = 0.5
    kd_hard_loss_weight: float = 1.0


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
    class_names: List[str] = field(default_factory=lambda: ["door", "cabinetDoor", "refrigeratorDoor", "window", "chair", "table", "cabinet", "couch", "openedDoor", "pole"])


def get_config() -> Config:
    """Return default configuration."""
    return Config()
