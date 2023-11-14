import torch.nn as nn
from .modules import Encoder, Decoder
import torch


class V2Net(nn.Module):
    def __init__(self, dims, depths, dp_rate, norm_type):
        super(V2Net, self).__init__()
        self.encoder = Encoder(5, dims, depths, dp_rate, norm_type)
        self.decoder = Decoder(1, dims, norm_type)
        # initializing
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, rgb, raw, hole_raw):
        x = torch.cat((rgb, raw, hole_raw), dim=1)
        x = self.encoder(x)
        x = self.decoder(x)
        return x
