
import math
import torch
from torch import nn

import csv
import csv
import SimpleITK as sitk
import os
import pandas as pd

from torchvision import transforms

from pl_bolts.transforms.dataset_normalizations import (
    imagenet_normalization
)

import monai
from monai.transforms import (
    EnsureChannelFirst,
    Compose,
    RandFlip,
    RandRotate,
    SpatialPad,
    RandSpatialCrop,
    CenterSpatialCrop,
    ScaleIntensityRange,
    ScaleIntensity,
    NormalizeIntensity,
    RandAdjustContrast,
    RandGaussianNoise,
    RandGaussianSmooth,
    ToTensor,
    EnsureChannelFirstd,
    RandFlipd,
    RandRotated,
    SpatialPadd,
    RandSpatialCropd,
    CenterSpatialCropd,
    ScaleIntensityRanged,
    ScaleIntensityRangePercentiles,
    ScaleIntensityd,
    NormalizeIntensityd,
    RandAdjustContrastd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    ToTensord
)



### TRANSFORMS
class SaltAndPepper:
    def __init__(self, prob=0.2):
        self.prob = prob
    def __call__(self, x):
        noise_tensor = torch.rand(x.shape)
        salt = torch.max(x)
        pepper = torch.min(x)
        x[noise_tensor < self.prob/2] = salt
        x[noise_tensor > 1-self.prob/2] = pepper
        return x

class NoTransform:
    def __init__(self,size=128,pad=32):
        self.train_transform = Compose(
            [
            EnsureChannelFirst(channel_dim='no_channel'),
            ToTensor(dtype=torch.float32,track_meta=False)
            ]
            )
    def __call__(self, inp):
        transformed_inp = self.train_transform(inp)
        return transformed_inp

class NoEvalTransform:
    def __init__(self,size=128):
        self.test_transform = transforms.Compose(
            [
                EnsureChannelFirst(channel_dim='no_channel'),
                ToTensor(dtype=torch.float32,track_meta=False)
            ]
            )
    def __call__(self, inp):
        transformed_inp = self.test_transform(inp)
        return transformed_inp


class TrainTransforms:
    def __init__(self, size=256, pad=10):
        # image augmentation functions
        calcul_rotate = math.pi/4
        print("Calcul Rotate", calcul_rotate)
        self.train_transform = Compose(
            [
                EnsureChannelFirst(channel_dim='no_channel'),
                # RandFlip(prob=0.5),
                RandRotate(prob=0.5, range_x=calcul_rotate, range_y=calcul_rotate, range_z=calcul_rotate, mode="nearest", padding_mode='zeros'),
                SpatialPad(spatial_size=size + pad),
                RandSpatialCrop(roi_size=size, random_size=False),
                RandGaussianNoise(prob=0.5),
                # RandGaussianSmooth(prob=0.5),
                # ScaleIntensityRangePercentiles(2,99,0,1),
                ScaleIntensity(0,1),
                RandAdjustContrast(prob=0.5),
                # NormalizeIntensity(subtrahend=0,divisor=10),
                ToTensor(dtype=torch.float32, track_meta=False)
            ]
        )
    def __call__(self, inp):
        return self.train_transform(inp)

class EvalTransforms:

    def __init__(self, size=256,pad=10):

        self.test_transform = Compose(
            [
                EnsureChannelFirst(channel_dim='no_channel'),
                SpatialPad(spatial_size=size+pad ),
                CenterSpatialCrop(size),
                # NormalizeIntensity(),
                ScaleIntensity(0,1),
                ToTensor(dtype=torch.float32, track_meta=False)
            ]
        )

    def __call__(self, inp):
        return self.test_transform(inp)

class SpecialTransforms:
    '''
    Transforms for the special case of the VAXI dataset
    '''
    def __init__(self, size=256, pad=10):
        self.sp_train_transform = Compose(
            [
                EnsureChannelFirst(channel_dim='no_channel'),
                SpatialPad(spatial_size=size + pad),
                RandSpatialCrop(roi_size=size, random_size=False),
                RandGaussianNoise(prob=0.5),
                RandGaussianSmooth(prob=0.5),
                ScaleIntensity(0,1),
                # RandAdjustContrast(prob=1),
                ToTensor(dtype=torch.float32, track_meta=False)
            ]
        )
    def __call__(self, inp):
        return self.sp_train_transform(inp)

class SegTrainTransforms:
    def __init__(self, size=128, pad=32):
        self.train_transform = Compose(
            [
                EnsureChannelFirstd(channel_dim='no_channel', keys=['img', 'seg']),
                RandFlipd(prob=0.5, keys=['img', 'seg']),
                RandRotated(prob=0.5, range_x=math.pi, range_y=math.pi, range_z=math.pi, mode="nearest", padding_mode='zeros', keys=['img', 'seg']),
                SpatialPadd(spatial_size=size + pad, keys=['img', 'seg']),
                RandSpatialCropd(roi_size=size, random_size=False, keys=['img', 'seg']),
                ScaleIntensityd(keys=['img']),
                RandAdjustContrastd(prob=0.5, keys=['img']),
                RandGaussianNoised(prob=0.5, keys=['img']),
                RandGaussianSmoothd(prob=0.5, keys=['img']),
                # NormalizeIntensity(keys=['img', 'seg']),
                ToTensord(dtype=torch.float32, track_meta=False, keys=['img']),
                ToTensord(dtype=torch.long, track_meta=False, keys=['seg'])
            ]
        )
    def __call__(self, inp):
        return self.train_transform(inp)

class SegEvalTransforms:

    def __init__(self, size=128):

        self.test_transform = transforms.Compose(
            [
                EnsureChannelFirstd(channel_dim='no_channel', keys=['img', 'seg']),
                CenterSpatialCropd(keys=['img', 'seg'], roi_size=size),
                # NormalizeIntensity(),
                ScaleIntensityd(keys=['img']),
                ToTensord(dtype=torch.float32, track_meta=False, keys=['img']),
                ToTensord(dtype=torch.long, track_meta=False, keys=['seg'])
            ]
        )

    def __call__(self, inp):
        return self.test_transform(inp)


class GaussianNoise(nn.Module):
    def __init__(self, mean=0.0, std=0.1):
        super(GaussianNoise, self).__init__()
        self.mean = torch.tensor(0.0)
        self.std = torch.tensor(0.1)
    def forward(self, x):
        return x + torch.normal(mean=self.mean, std=self.std, size=x.size(), device=x.device)

