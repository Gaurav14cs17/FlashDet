"""
ShuffleNetV2 Backbone for NanoDet-Plus-Lite.
"""

import torch
import torch.nn as nn

MODEL_URLS = {
    "0.5x": "https://download.pytorch.org/models/shufflenetv2_x0.5-f707e7126e.pth",
    "1.0x": "https://download.pytorch.org/models/shufflenetv2_x1-5666bf0f80.pth",
}


def channel_shuffle(x: torch.Tensor, groups: int) -> torch.Tensor:
    """Channel shuffle operation."""
    batch, channels, height, width = x.size()
    channels_per_group = channels // groups
    x = x.view(batch, groups, channels_per_group, height, width)
    x = x.transpose(1, 2).contiguous()
    x = x.view(batch, channels, height, width)
    return x


class ShuffleUnit(nn.Module):
    """ShuffleNetV2 basic unit."""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int, activation: nn.Module):
        super().__init__()
        self.stride = stride
        branch_channels = out_channels // 2
        
        if stride == 2:
            self.branch1 = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, stride, 1, groups=in_channels, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.Conv2d(in_channels, branch_channels, 1, bias=False),
                nn.BatchNorm2d(branch_channels),
                activation,
            )
        else:
            self.branch1 = nn.Sequential()
        
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels if stride > 1 else branch_channels, branch_channels, 1, bias=False),
            nn.BatchNorm2d(branch_channels),
            activation,
            nn.Conv2d(branch_channels, branch_channels, 3, stride, 1, groups=branch_channels, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.Conv2d(branch_channels, branch_channels, 1, bias=False),
            nn.BatchNorm2d(branch_channels),
            activation,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat([x1, self.branch2(x2)], dim=1)
        else:
            out = torch.cat([self.branch1(x), self.branch2(x)], dim=1)
        return channel_shuffle(out, 2)


class ShuffleNetV2(nn.Module):
    """
    ShuffleNetV2 backbone.
    
    Args:
        model_size: Model variant ("0.5x", "1.0x", "1.5x", "2.0x")
        out_stages: Stages to output features from
        pretrained: Whether to load pretrained weights
        activation: Activation function name
    """
    
    STAGE_CONFIGS = {
        "0.5x": ([4, 8, 4], [24, 48, 96, 192, 1024]),
        "1.0x": ([4, 8, 4], [24, 116, 232, 464, 1024]),
        "1.5x": ([4, 8, 4], [24, 176, 352, 704, 1024]),
        "2.0x": ([4, 8, 4], [24, 244, 488, 976, 2048]),
    }
    
    def __init__(
        self,
        model_size: str = "0.5x",
        out_stages: tuple = (2, 3, 4),
        pretrained: bool = True,
        activation: str = "LeakyReLU"
    ):
        super().__init__()
        
        self.out_stages = out_stages
        repeats, channels = self.STAGE_CONFIGS[model_size]
        self.out_channels = [channels[s] for s in out_stages]
        
        if activation == "LeakyReLU":
            act = nn.LeakyReLU(0.1, inplace=True)
        elif activation == "ReLU":
            act = nn.ReLU(inplace=True)
        else:
            act = nn.LeakyReLU(0.1, inplace=True)
        
        # Stem
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, channels[0], 3, 2, 1, bias=False),
            nn.BatchNorm2d(channels[0]),
            act,
        )
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        
        # Stages
        self.stages = nn.ModuleList()
        in_ch = channels[0]
        for i, (repeat, out_ch) in enumerate(zip(repeats, channels[1:-1])):
            stage = []
            for j in range(repeat):
                stride = 2 if j == 0 else 1
                stage.append(ShuffleUnit(in_ch, out_ch, stride, act))
                in_ch = out_ch
            self.stages.append(nn.Sequential(*stage))
        
        if pretrained and model_size in MODEL_URLS:
            self._load_pretrained(model_size)
    
    def _load_pretrained(self, model_size: str):
        """Load pretrained weights."""
        try:
            state_dict = torch.hub.load_state_dict_from_url(
                MODEL_URLS[model_size], progress=True
            )
            self.load_state_dict(state_dict, strict=False)
            print(f"Loaded pretrained ShuffleNetV2 {model_size}")
        except Exception as e:
            print(f"Could not load pretrained weights: {e}")
    
    def forward(self, x: torch.Tensor) -> list:
        """Forward pass returning multi-scale features."""
        x = self.conv1(x)
        x = self.maxpool(x)
        
        outputs = []
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i + 2 in self.out_stages:
                outputs.append(x)
        
        return outputs
