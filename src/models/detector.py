"""
NanoDet-Plus-Lite Detector.
Matches official NanoDet implementation.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional

from .backbone import ShuffleNetV2
from .neck import GhostPAN
from .head import NanoDetPlusHead, SimpleConvHead


class NanoDetPlusLite(nn.Module):
    """
    NanoDet-Plus-Lite object detector.
    
    Ultra-lightweight detector with ShuffleNetV2 backbone.
    Matches official NanoDet-Plus implementation.
    
    Official Model Specs (for reference):
    - NanoDet-Plus-m (1.0x, fpn=96):      ~1.17M params, 2.3MB FP16, 1.2MB INT8
    - NanoDet-Plus-m-1.5x (1.5x, fpn=128): ~2.44M params, 4.7MB FP16, 2.3MB INT8
    - NanoDet-Plus-m-0.5x (0.5x, fpn=96):  ~0.95M params (ultra-lite variant)
    
    Args:
        num_classes: Number of detection classes.
        input_size: Input image size (width, height).
        backbone_size: Backbone variant ("0.5x", "1.0x", "1.5x").
        fpn_channels: FPN output channels.
        strides: Feature map strides.
        reg_max: Max value for distribution focal loss.
        pretrained: Whether to load pretrained backbone.
        use_aux_head: Whether to use auxiliary head for training.
    """
    
    # Output channels from ShuffleNetV2 stages 2, 3, 4
    # These must match the backbone's actual output channels
    BACKBONE_CHANNELS = {
        "0.5x": [48, 96, 192],      # ShuffleNetV2 0.5x: channels[2,3,4] = [48, 96, 192]
        "1.0x": [116, 232, 464],    # ShuffleNetV2 1.0x: channels[2,3,4] = [116, 232, 464]
        "1.5x": [176, 352, 704],    # ShuffleNetV2 1.5x: channels[2,3,4] = [176, 352, 704]
        "2.0x": [244, 488, 976],    # ShuffleNetV2 2.0x: channels[2,3,4] = [244, 488, 976]
    }
    
    def __init__(
        self,
        num_classes: int = 10,
        input_size: Tuple[int, int] = (320, 320),
        backbone_size: str = "0.5x",
        fpn_channels: int = 96,
        strides: List[int] = [8, 16, 32, 64],
        reg_max: int = 7,
        pretrained: bool = True,
        use_aux_head: bool = True,
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.input_size = input_size
        self.strides = strides
        self.use_aux_head = use_aux_head
        self.detach_epoch = 10  # Detach aux head after this epoch
        
        # Backbone
        self.backbone = ShuffleNetV2(
            model_size=backbone_size,
            out_stages=(2, 3, 4),
            pretrained=pretrained,
            activation="LeakyReLU"
        )
        
        # FPN (Neck)
        in_channels = self.BACKBONE_CHANNELS[backbone_size]
        self.fpn = GhostPAN(
            in_channels=in_channels,
            out_channels=fpn_channels,
            kernel_size=5,
            num_extra_level=1,
            use_depthwise=True,
            activation="LeakyReLU"
        )
        
        # Detection head
        self.head = NanoDetPlusHead(
            num_classes=num_classes,
            input_channel=fpn_channels,
            feat_channels=fpn_channels,
            stacked_convs=2,
            kernel_size=5,
            strides=strides,
            reg_max=reg_max,
            activation="LeakyReLU"
        )
        
        # Auxiliary head (for training only)
        if use_aux_head:
            self.aux_head = SimpleConvHead(
                num_classes=num_classes,
                input_channel=fpn_channels * 2,
                feat_channels=fpn_channels * 2,
                stacked_convs=4,
                strides=strides,
                reg_max=reg_max,
                activation="LeakyReLU"
            )
    
    def forward(
        self,
        x: torch.Tensor,
        gt_meta: Dict = None,
        epoch: int = 0,
        compute_loss: bool = False
    ) -> Dict:
        """
        Forward pass.
        
        Args:
            x: Input tensor [B, 3, H, W].
            gt_meta: Ground truth metadata (for training).
            epoch: Current epoch (for aux head detachment).
            compute_loss: If True, compute loss even when not in training mode.
                         Used for validation with proper BatchNorm eval behavior.
            
        Returns:
            In training (or compute_loss=True): Dict with 'loss' and 'loss_states'.
            In inference: Dict with 'preds'.
        """
        # Backbone features
        features = self.backbone(x)
        
        # FPN features
        fpn_features = self.fpn(features)
        
        # Detection head
        preds = self.head(fpn_features)
        
        # Compute loss if in training mode OR if explicitly requested (for validation)
        if (self.training or compute_loss) and gt_meta is not None:
            gt_meta["img"] = x
            
            # Auxiliary head (only during actual training, not validation)
            aux_preds = None
            if self.training and self.use_aux_head and hasattr(self, "aux_head"):
                # Concatenate FPN features with downsampled features for aux head
                aux_features = self._get_aux_features(fpn_features)
                if epoch >= self.detach_epoch:
                    aux_features = [f.detach() for f in aux_features]
                aux_preds = self.aux_head(aux_features)
            
            # Compute loss
            loss, loss_states = self.head.loss(preds, gt_meta, aux_preds)
            
            return {
                "loss": loss,
                "loss_states": loss_states
            }
        else:
            return {"preds": preds}
    
    def _get_aux_features(self, fpn_features: List[torch.Tensor]) -> List[torch.Tensor]:
        """Get features for auxiliary head by concatenating adjacent scales."""
        aux_features = []
        for i in range(len(fpn_features)):
            if i < len(fpn_features) - 1:
                # Upsample next level and concatenate
                h, w = fpn_features[i].shape[2:]
                upsampled = torch.nn.functional.interpolate(
                    fpn_features[i + 1], size=(h, w), mode="bilinear", align_corners=False
                )
                aux_features.append(torch.cat([fpn_features[i], upsampled], dim=1))
            else:
                # Last level: concatenate with itself
                aux_features.append(torch.cat([fpn_features[i], fpn_features[i]], dim=1))
        return aux_features
    
    @torch.no_grad()
    def predict(
        self,
        x: torch.Tensor,
        img_metas: Optional[Dict] = None,
        score_thr: float = 0.05,
        nms_thr: float = 0.6
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Run inference.
        
        Args:
            x: Input tensor [B, 3, H, W].
            img_metas: Image metadata (height, width).
            score_thr: Score threshold.
            nms_thr: NMS threshold.
            
        Returns:
            List of (det_bboxes, det_labels) per image.
        """
        self.eval()
        
        if img_metas is None:
            img_metas = {"img": x}
        else:
            img_metas["img"] = x
        
        output = self.forward(x)
        preds = output["preds"]
        
        results = self.head.get_bboxes(preds, img_metas, score_thr, nms_thr)
        return results
    
    def get_model_info(self) -> Dict:
        """Get model information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "name": "NanoDetPlusLite",
            "num_classes": self.num_classes,
            "input_size": self.input_size,
            "total_params": total_params,
            "trainable_params": trainable_params,
            "params_mb": total_params * 4 / (1024 ** 2),
        }


def build_model(config) -> NanoDetPlusLite:
    """
    Build model from config.
    
    Args:
        config: Model configuration.
        
    Returns:
        NanoDetPlusLite model.
    """
    return NanoDetPlusLite(
        num_classes=config.model.num_classes,
        input_size=config.model.input_size,
        backbone_size=config.model.backbone_size,
        fpn_channels=config.model.fpn_out_channels,
        pretrained=config.model.backbone_pretrained
    )
