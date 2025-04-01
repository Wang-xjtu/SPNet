import torch
import numpy as np
import random
from PIL import Image


def rgb_read(filename):
    data = Image.open(filename)
    rgb = (np.array(data) / 255.0).astype(np.float32)
    rgb = torch.from_numpy(rgb.transpose((2, 0, 1)))
    data.close()
    return rgb


def depth_read(filename):
    data = Image.open(filename)
    # make sure we have a proper 16bit depth map here.. not 8bit!
    depth = (np.array(data) / 65535.0).astype(np.float32)
    depth = torch.from_numpy(depth).unsqueeze(0)
    data.close()
    return depth


class DataReader(object):
    def __init__(self, device):
        super(DataReader, self).__init__()
        self.device = device

    def read_data(self, rgb_path, raw_path, gt_path):
        rgb = rgb_read(rgb_path).unsqueeze(0).to(torch.float32)
        raw = depth_read(raw_path).unsqueeze(0).to(torch.float32)
        gt = depth_read(gt_path).unsqueeze(0).to(torch.float32)
        hole_raw = (raw > 0).float()
        return (
            rgb.to(self.device),
            raw.to(self.device),
            gt.to(self.device),
            hole_raw.to(self.device),
        )

    def toint32(self, pred):
        pred = pred.squeeze().to("cpu").numpy()
        pred = np.clip(pred * 65535.0, 0, 65535).astype(np.int32)
        return pred


def metrics(pred, gt, scale_factor=256):
    # valid pixels
    mask = gt > 0.0
    valid_nums = torch.sum(mask) + 1e-8
    # scaling value range from [0, 1] to [0, 256] for better visualization (scale_factor=256)
    pred = pred[mask] * scale_factor
    gt = gt[mask] * scale_factor

    diff = pred - gt
    # rmse
    rmse = torch.sqrt(torch.sum(diff**2) / valid_nums)
    # rel
    rel = torch.sum(torch.abs(diff) / (gt + 1e-6)) / valid_nums
    return rmse, rel
