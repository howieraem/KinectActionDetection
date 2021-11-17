"""
Experimental network layers.

TODO: GCN
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from global_configs import MAX_PITCH_YAW_ANGLE
from utils.processing import affine_transform_sequence


class GaussianNoise(nn.Module):
    def __init__(self, stddev: float = 0.125):
        super(GaussianNoise, self).__init__()
        self.stddev = stddev

    def forward(self, x):
        if self.training:
            return x * Variable(torch.randn_like(x) * self.stddev)
        return x


class AffineTransform(nn.Module):
    def __init__(self,
                 max_yaw: float = 0,
                 max_pitch: float = MAX_PITCH_YAW_ANGLE,
                 max_roll: float = 0):
        super(AffineTransform, self).__init__()
        self.max_yaw = max_yaw
        self.max_pitch = max_pitch
        self.max_roll = max_roll

    def forward(self, x):
        if self.training:
            alpha = torch.rand(1).item() * self.max_yaw
            beta = torch.rand(1).item() * self.max_pitch
            gamma = torch.rand(1).item() * self.max_roll
            x = affine_transform_sequence(x, alpha, beta, gamma).unsqueeze(0)
        return x


class MotionInspection(nn.Module):
    def __init__(self, channels: int):
        super(MotionInspection, self).__init__()
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=1)

    def forward(self, x):
        y1 = self.conv1(x)
        y2 = self.conv2(x)
        return torch.cat((x, y1, y2), dim=1)


class SkeletonConvolution(nn.Module):
    def __init__(self):
        super(SkeletonConvolution, self).__init__()
        self.conv1 = nn.Conv1d(1, 1, 3, padding=1)
        self.conv2 = nn.Conv1d(1, 1, 5, padding=2)

    def forward(self, x):
        x1 = self.conv1(x.permute((1, 0, 2)).contiguous())
        x1 = x1.view((x.shape[1], 1, x1.shape[-1]))
        return torch.cat((x, x1.permute(1, 0, 2)), dim=-1)
