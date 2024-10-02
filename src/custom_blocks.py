import torch
import torch.nn as nn
from torchvision.ops import StochasticDepth


class LayerNorm(nn.Module):
    """ LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    
    We use channel_first mode for SP-Norm.
    """

    def __init__(self, normalized_shape, eps=1e-6, affine=True):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.weight = nn.Parameter(torch.ones(1, normalized_shape, 1, 1))
            self.bias = nn.Parameter(torch.zeros(1, normalized_shape, 1, 1))

    def forward(self, x):
        s, u = torch.std_mean(x, dim=1, keepdim=True)
        x = (x - u) / (s + self.eps)
        if self.affine:
            x = self.weight * x + self.bias
        return x


class NormLayer(nn.Module):
    def __init__(self, normalized_shape, norm_type):
        super(NormLayer, self).__init__()
        self.norm_type = norm_type

        if self.norm_type == 'LN':
            self.norm = LayerNorm(normalized_shape, affine=True)
        elif self.norm_type == 'BN':
            self.norm = nn.BatchNorm2d(normalized_shape, affine=True)
        elif self.norm_type == 'IN':
            self.norm = nn.InstanceNorm2d(normalized_shape, affine=True)
        elif self.norm_type == 'RZ':
            self.norm = nn.Identity()
        elif self.norm_type in ['CNX', 'CN+X', 'GRN']:
            self.norm = LayerNorm(normalized_shape, affine=False)
            # Use 1*1 conv to implement SLP in the channel dimension. 
            self.conv = nn.Conv2d(normalized_shape, normalized_shape, kernel_size=1)
        elif self.norm_type == 'NX':
            self.norm = LayerNorm(normalized_shape, affine=True)
        elif self.norm_type == 'CX':
            self.conv = nn.Conv2d(normalized_shape, normalized_shape, kernel_size=1)
        else:
            raise ValueError('norm_type error')

    def forward(self, x):
        if self.norm_type in ['LN', 'BN', 'IN', 'RZ']:
            x = self.norm(x)
        elif self.norm_type in ['CNX', 'GRN']:
            x = self.conv(self.norm(x)) * x
        elif self.norm_type == 'CN+X':
            x = self.conv(self.norm(x)) + x
        elif self.norm_type == 'NX':
            x = self.norm(x) * x
        elif self.norm_type == 'CX':
            x = self.conv(x) * x
        else:
            raise ValueError('norm_type error')
        return x


class CNBlock(nn.Module):
    def __init__(self, dim: int, norm_type: str, dp_rate: float):
        super(CNBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim),
            NormLayer(dim, norm_type),
            nn.Conv2d(dim, 4 * dim, kernel_size=1),
            nn.ReLU(inplace=True),
            GRN(4 * dim) if norm_type == 'GRN' else nn.Identity(),
            nn.Conv2d(4 * dim, dim, kernel_size=1),
        )
        self.drop_path = StochasticDepth(dp_rate, mode='batch')
        self.norm_type = norm_type
        if self.norm_type == 'RZ':
            self.alpha = nn.Parameter(torch.tensor(0.))

    def forward(self, x):
        res = self.block(x)
        if self.norm_type == 'RZ':
            res = self.alpha * res
        x = x + self.drop_path(res)
        return x


class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    """

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)
        Nx = Gx / (Gx.mean(dim=1, keepdim=True) + self.eps)
        x = (1 + self.gamma * Nx) * x + self.beta
        return x
