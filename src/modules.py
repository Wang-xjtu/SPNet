import torch.nn as nn
from .custom_blocks import CNBlock, NormLayer
import torch


class Encoder(nn.Module):
    def __init__(
        self,
        in_chans=5,
        dims=[96, 192, 384, 768],
        depths=[3, 3, 9, 3],
        dp_rate=0.0,
        norm_type="CNX",
    ):
        super(Encoder, self).__init__()
        all_dims = [dims[0] // 4, dims[0] // 2] + dims
        self.downsample_layers = nn.ModuleList()
        stem = nn.Conv2d(in_chans, all_dims[0], kernel_size=3, padding=1)
        self.downsample_layers.append(stem)
        for i in range(5):
            downsample_layer = nn.Sequential(
                NormLayer(all_dims[i], norm_type),
                nn.Conv2d(all_dims[i], all_dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        self.stages.append(nn.Identity())
        self.stages.append(nn.Identity())
        dp_rates = [x.item() for x in torch.linspace(0, dp_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[
                    CNBlock(dims[i], norm_type, dp_rates[cur + j])
                    for j in range(depths[i])
                ]
            )
            self.stages.append(stage)
            cur += depths[i]

    def forward(self, x):
        outputs = []
        for i in range(6):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            outputs.append(x)
        return outputs


class Decoder(nn.Module):
    def __init__(self, out_chans=1, dims=[96, 192, 384, 768], norm_type="CNX"):
        super(Decoder, self).__init__()
        all_dims = [dims[0] // 4, dims[0] // 2] + dims
        self.upsample_layers = nn.ModuleList()
        self.fusion_layers = nn.ModuleList()
        for i in range(5):
            upsample_layer = nn.ConvTranspose2d(
                all_dims[i + 1], all_dims[i], kernel_size=2, stride=2
            )
            fusion_layer = nn.Conv2d(2 * all_dims[i], all_dims[i], kernel_size=1)
            self.upsample_layers.append(upsample_layer)
            self.fusion_layers.append(fusion_layer)

        self.stages = nn.ModuleList()
        self.stages.append(nn.Identity())
        self.stages.append(nn.Identity())
        for i in range(3):
            stage = CNBlock(dims[i], norm_type, 0.0)
            self.stages.append(stage)
        self.head = nn.Conv2d(all_dims[0], out_chans, kernel_size=3, padding=1)

    def forward(self, ins):
        x = ins[-1]
        for i in range(4, -1, -1):
            x = self.upsample_layers[i](x)
            x = torch.cat([ins[i], x], dim=1)
            x = self.fusion_layers[i](x)
            x = self.stages[i](x)
        x = self.head(x)
        return x
