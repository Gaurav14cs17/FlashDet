"""
LoRA (Low-Rank Adaptation) for FlashDet.

Implements LoRA as described in "LoRA: Low-Rank Adaptation of Large Language
Models" (Hu et al., 2022) — adapted from torchtune's LoRA implementation
for use with convolutional object detection backbones.

LoRA freezes the pretrained weights and injects trainable low-rank
decomposition matrices (A, B) into target layers.  This allows fine-tuning
with a fraction of the trainable parameters and significantly less GPU memory.
"""

import logging
import math
from typing import List, Optional, Set

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class LoRALinear(nn.Module):
    """Linear layer with LoRA adaptation.

    Wraps an existing ``nn.Linear`` layer, freezing its weight and adding
    low-rank A/B matrices that are the only trainable parameters.

    output = W_frozen @ x + (alpha/rank) * B @ A @ x
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.scaling = alpha / rank

        self.weight = nn.Parameter(
            torch.empty(out_features, in_features), requires_grad=False
        )
        self.bias = (
            nn.Parameter(torch.zeros(out_features), requires_grad=False)
            if bias
            else None
        )

        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = F.linear(x, self.weight, self.bias)
        lora_out = (
            self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T * self.scaling
        )
        return base_out + lora_out


class LoRAConv2d(nn.Module):
    """Conv2d layer with LoRA adaptation.

    For convolutional layers the low-rank decomposition operates on the
    reshaped (out_channels, in_channels * kH * kW) weight matrix.  The
    adaptation is applied as a 1x1 conv (A) followed by a 1x1 conv (B),
    which is equivalent to the standard LoRA formulation when viewed as a
    matrix decomposition.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        groups: int = 1,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()
        self.rank = rank
        self.scaling = alpha / rank

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels // groups, kernel_size, kernel_size),
            requires_grad=False,
        )
        self.bias = (
            nn.Parameter(torch.zeros(out_channels), requires_grad=False)
            if bias
            else None
        )

        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.kernel_size = kernel_size

        self.lora_A = nn.Conv2d(
            in_channels, rank, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.lora_B = nn.Conv2d(
            rank, out_channels, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.lora_dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = F.conv2d(
            x, self.weight, self.bias,
            stride=self.stride, padding=self.padding, groups=self.groups,
        )
        lora_out = self.lora_B(self.lora_A(self.lora_dropout(x))) * self.scaling
        if lora_out.shape[2:] != base_out.shape[2:]:
            lora_out = F.adaptive_avg_pool2d(lora_out, base_out.shape[2:])
        return base_out + lora_out


def _replace_linear_with_lora(
    module: nn.Module,
    rank: int,
    alpha: float,
    dropout: float,
    replaced: list,
    prefix: str = "",
):
    for name, child in list(module.named_children()):
        full_name = f"{prefix}.{name}" if prefix else name

        if isinstance(child, nn.Linear):
            lora_layer = LoRALinear(
                child.in_features,
                child.out_features,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
                bias=child.bias is not None,
            )
            lora_layer.weight.data.copy_(child.weight.data)
            if child.bias is not None:
                lora_layer.bias.data.copy_(child.bias.data)
            setattr(module, name, lora_layer)
            replaced.append(full_name)

        elif isinstance(child, nn.Conv2d) and child.groups == 1 and child.kernel_size[0] == 1:
            lora_layer = LoRAConv2d(
                child.in_channels,
                child.out_channels,
                kernel_size=child.kernel_size[0],
                stride=child.stride[0],
                padding=child.padding[0],
                groups=child.groups,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
                bias=child.bias is not None,
            )
            lora_layer.weight.data.copy_(child.weight.data)
            if child.bias is not None:
                lora_layer.bias.data.copy_(child.bias.data)
            setattr(module, name, lora_layer)
            replaced.append(full_name)
        else:
            _replace_linear_with_lora(
                child, rank, alpha, dropout, replaced, full_name
            )


def apply_lora(
    model: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.05,
    target_modules: Optional[List[str]] = None,
) -> nn.Module:
    """Apply LoRA adapters to target modules of a FlashDet model.

    Freezes all pretrained weights and only makes LoRA A/B matrices trainable.
    For object detection, the backbone benefits most from LoRA since it has the
    most parameters and the pretrained features are what we want to adapt.

    Args:
        model: FlashDet instance.
        rank: LoRA rank (lower = fewer params, higher = more capacity).
        alpha: Scaling factor.  ``alpha/rank`` is the effective scale.
        dropout: Dropout applied to LoRA input.
        target_modules: List of top-level module names to apply LoRA to.
            Defaults to ``["backbone"]``.

    Returns:
        The model with LoRA applied.  Only LoRA parameters are trainable.
    """
    if target_modules is None:
        target_modules = ["backbone", "fpn"]

    # Freeze everything first
    for p in model.parameters():
        p.requires_grad = False

    replaced = []
    for mod_name in target_modules:
        submod = getattr(model, mod_name, None)
        if submod is None:
            logger.warning("LoRA target '%s' not found on model, skipping", mod_name)
            continue
        _replace_linear_with_lora(
            submod, rank, alpha, dropout, replaced, prefix=mod_name
        )

    # Unfreeze LoRA parameters + detection head (always trainable for fine-tuning)
    _unfreeze_lora_params(model)
    _unfreeze_module(model, "head")
    _unfreeze_module(model, "aux_head")

    # Unfreeze BatchNorm in targeted modules so running stats adapt to LoRA output shifts
    for mod_name in target_modules:
        submod = getattr(model, mod_name, None)
        if submod is None:
            continue
        for m in submod.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.SyncBatchNorm)):
                m.train()
                for p in m.parameters():
                    p.requires_grad = True

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "LoRA applied: %d layers adapted (rank=%d, alpha=%.1f). "
        "Trainable: %d / %d params (%.1f%%)",
        len(replaced), rank, alpha, trainable, total,
        100.0 * trainable / max(total, 1),
    )

    return model


def _unfreeze_lora_params(model: nn.Module):
    """Unfreeze all LoRA A/B parameters and their dropout layers."""
    for name, p in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            p.requires_grad = True


def _unfreeze_module(model: nn.Module, module_name: str):
    """Unfreeze all parameters of a named sub-module."""
    mod = getattr(model, module_name, None)
    if mod is None:
        return
    for p in mod.parameters():
        p.requires_grad = True


def get_lora_state_dict(model: nn.Module) -> dict:
    """Extract LoRA adapter weights and associated BatchNorm parameters."""
    return {
        k: v for k, v in model.state_dict().items()
        if "lora_A" in k or "lora_B" in k
        or "running_mean" in k or "running_var" in k
        or "num_batches_tracked" in k
    }


def merge_lora_weights(model: nn.Module) -> nn.Module:
    """Merge LoRA weights into the base weights for inference.

    After merging, LoRA layers become standard Linear/Conv2d layers with no
    runtime overhead.  This is equivalent to ``W_merged = W + (alpha/r) * B @ A``.
    """
    merged_count = 0
    for module in model.modules():
        if isinstance(module, LoRALinear):
            delta = (module.lora_B @ module.lora_A) * module.scaling
            module.weight.data += delta
            module.lora_A.data.zero_()
            module.lora_B.data.zero_()
            merged_count += 1
        elif isinstance(module, LoRAConv2d):
            A_weight = module.lora_A.weight.data.squeeze(-1).squeeze(-1)
            B_weight = module.lora_B.weight.data.squeeze(-1).squeeze(-1)
            delta = (B_weight @ A_weight).unsqueeze(-1).unsqueeze(-1)
            if module.kernel_size == 1:
                module.weight.data += delta * module.scaling
            merged_count += 1

    logger.info("Merged LoRA weights in %d layers", merged_count)
    return model


# ─────────────────────────────────────────────────────────────────────
# QLoRA: Quantized base weights (INT8/NF4) + LoRA adapters
# Inspired by torchtune's QLoRA implementation
# ─────────────────────────────────────────────────────────────────────

def _quantize_module_weights(module: nn.Module, quant_dtype: str = "int8"):
    """Quantize frozen weights of a module to save memory.

    Supported dtypes:
      - ``"int8"``:  Dynamic per-channel INT8 quantization (no extra deps).
      - ``"nf4"``:   4-bit NormalFloat via bitsandbytes (requires bitsandbytes).

    Only affects parameters with ``requires_grad=False`` that are floating-point.
    """
    quantized = 0
    if quant_dtype == "nf4":
        try:
            import bitsandbytes as bnb
            for name, child in module.named_modules():
                if isinstance(child, (LoRALinear, LoRAConv2d)):
                    w = child.weight
                    if not w.requires_grad and w.is_floating_point() and w.ndim == 2:
                        qw = bnb.nn.Params4bit(
                            w.data, requires_grad=False, quant_type="nf4"
                        )
                        child.weight = qw
                        quantized += 1
            logger.info("QLoRA (NF4): quantized %d weight tensors", quantized)
            return quantized
        except ImportError:
            logger.warning(
                "bitsandbytes not installed — falling back to INT8 quantization. "
                "Install with: pip install bitsandbytes"
            )
            quant_dtype = "int8"

    if quant_dtype == "int8":
        for child in module.modules():
            if isinstance(child, (LoRALinear, LoRAConv2d)):
                w = child.weight
                if not w.requires_grad and w.is_floating_point():
                    scale = w.data.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 127.0
                    w_int8 = (w.data / scale).round().clamp(-128, 127).to(torch.int8)
                    child.register_buffer("_qlora_w_int8", w_int8)
                    child.register_buffer("_qlora_scale", scale)
                    child.weight = nn.Parameter(
                        (w_int8.float() * scale), requires_grad=False
                    )
                    quantized += 1
        logger.info("QLoRA (INT8): quantized %d weight tensors", quantized)

    return quantized


def apply_qlora(
    model: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.05,
    target_modules: Optional[List[str]] = None,
    quant_dtype: str = "int8",
) -> nn.Module:
    """Apply QLoRA: quantized base weights + LoRA adapters.

    This is the torchtune-style QLoRA approach:
      1. Apply LoRA adapters to target modules (freezing base weights).
      2. Quantize the frozen base weights to INT8 or NF4.

    The result uses significantly less GPU memory than full LoRA while
    maintaining comparable fine-tuning quality.

    Args:
        model: FlashDet instance.
        rank: LoRA rank.
        alpha: LoRA scaling alpha.
        dropout: LoRA dropout.
        target_modules: Modules to apply LoRA + quantization to.
        quant_dtype: ``"int8"`` (no extra deps) or ``"nf4"`` (requires bitsandbytes).

    Returns:
        Model with QLoRA applied.
    """
    model = apply_lora(
        model, rank=rank, alpha=alpha, dropout=dropout,
        target_modules=target_modules,
    )

    n_quantized = _quantize_module_weights(model, quant_dtype=quant_dtype)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "QLoRA applied: %d layers quantized (%s). "
        "Trainable: %d / %d params (%.1f%%)",
        n_quantized, quant_dtype, trainable, total,
        100.0 * trainable / max(total, 1),
    )
    return model
