
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
    
class CleftTrainTransforms:
    def __init__(self, size=128, pad=32):
        # image augmentation functions        
        self.train_transform = Compose(
            [
                EnsureChannelFirst(channel_dim='no_channel'),                
                RandFlip(prob=0.5),
                RandRotate(prob=0.5, range_x=math.pi, range_y=math.pi, range_z=math.pi, mode="nearest", padding_mode='zeros'),
                SpatialPad(spatial_size=size+pad ),
                RandSpatialCrop(roi_size=size, random_size=False),
                ScaleIntensity(),
                RandAdjustContrast(prob=0.5),
                RandGaussianNoise(prob=0.5),
                RandGaussianSmooth(prob=0.5),
                # NormalizeIntensity(),
                ToTensor(dtype=torch.float32, track_meta=False)
            ]
        )

        # self.output_dir = output_dir
        # if not os.path.exists(output_dir):
        #     os.makedirs(output_dir)
        
        # self.csv_path = os.path.join(output_dir, 'train_transformed.csv')
        # self.csv_data = []

    def __call__(self, inp):
        
        transformed_inp = self.train_transform(inp)
      
        # # output_path = os.path.join(self.output_dir,filename)
        # print('filename',filename)
        # filename = 'Preprocess/' + filename
        # output_path = filename.replace('Preprocess/Output/Resampled','./Output/TransformedImages')
        # print('output_path',output_path)

        # out_dir = os.path.dirname(output_path)
        # if not os.path.exists(out_dir):
        #     os.makedirs(out_dir)
        # # Convert tensor to SimpleITK image and save
        # sitk_image = sitk.GetImageFromArray(transformed_inp.numpy())
        # sitk.WriteImage(sitk_image, output_path)

        # # Record the file path
        # self.csv_data.append([filename, output_path])

        return transformed_inp

    # def save_csv(self):
    #     df = pd.DataFrame(self.csv_data, columns=['file_name', 'path'])
    #     df.to_csv(self.csv_path, index=False)


class CleftEvalTransforms:

    def __init__(self, size=128):

        self.test_transform = transforms.Compose(
            [
                EnsureChannelFirst(channel_dim='no_channel'),
                CenterSpatialCrop(size),
                # NormalizeIntensity(),
                ScaleIntensity(),
                ToTensor(dtype=torch.float32, track_meta=False)
            ]
        )

    def __call__(self, inp):        
        return self.test_transform(inp)


class CleftSegTrainTransforms:
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

class CleftSegEvalTransforms:

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

