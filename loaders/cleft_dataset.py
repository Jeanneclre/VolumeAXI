from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import SimpleITK as sitk
from PIL import Image
import nrrd
import os
import sys
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from monai.data import SmartCacheDataset
import pytorch_lightning as pl

class BasicDataset(Dataset):
    def __init__(self, df, mount_point = "./", img_column='img', class_column='Classification', transform=None):
        self.df = df


        self.mount_point = mount_point
        self.transform = transform
        self.img_column = img_column
        self.class_column = class_column

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        # df_filtered = self.df.dropna(subset=[self.class_column])
        # df_filtered.reset_index(drop=True)

        row = self.df.loc[idx]

        # print('*********idx***********',idx)

        img = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.mount_point, row[self.img_column])))

        if self.transform:
            img = self.transform(img)

        cl = int(row[self.class_column])

        return img, torch.tensor(cl, dtype=torch.long)


class DataModule(pl.LightningDataModule):
    def __init__(self, df_train, df_val, df_test, mount_point="./", batch_size=32, num_workers=4, img_column='img_path', class_column='Classification', train_transform=None, valid_transform=None, test_transform=None, drop_last=False):
        super().__init__()

        self.df_train = df_train
        self.df_val = df_val
        self.df_test = df_test
        self.mount_point = mount_point
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_column = img_column
        self.class_column = class_column
        self.train_transform = train_transform
        self.valid_transform = valid_transform
        self.test_transform = test_transform
        self.drop_last=drop_last

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        base_train_ds = BasicDataset(self.df_train, mount_point=self.mount_point, img_column=self.img_column, class_column=self.class_column, transform=self.train_transform)
        self.train_ds = SmartCacheDataset(base_train_ds, num_replace_workers=self.num_workers,replace_rate=0.3, cache_rate=1.0, cache_num=1)

        base_val_ds = BasicDataset(self.df_val, mount_point=self.mount_point, img_column=self.img_column, class_column=self.class_column, transform=self.valid_transform)
        self.val_ds = SmartCacheDataset(base_val_ds,num_replace_workers=self.num_workers,replace_rate=0.3, cache_rate=1.0, cache_num=1)

        base_test_ds = BasicDataset(self.df_test, mount_point=self.mount_point, img_column=self.img_column, class_column=self.class_column, transform=self.test_transform)
        self.test_ds = SmartCacheDataset(base_test_ds,num_replace_workers=self.num_workers,replace_rate=0.3, cache_rate=1.0, cache_num=1)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True, pin_memory=True, drop_last=self.drop_last, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True, pin_memory=True, drop_last=self.drop_last)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=self.drop_last)


class SegDataset(Dataset):
    def __init__(self, df, mount_point = "./", img_column='img', seg_column='seg', class_column='Classification', transform=None):
        self.df = df
        self.mount_point = mount_point
        self.transform = transform
        self.img_column = img_column
        self.class_column = class_column
        self.seg_column = seg_column

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.loc[idx]
        img = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.mount_point, row[self.img_column])))

        seg = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.mount_point, row[self.seg_column])))

        if self.transform:
            obj = self.transform({"img": img, "seg": seg})
            img = obj["img"]
            seg = obj["seg"]

        cl = row[self.class_column]

        return img, seg, torch.tensor(cl, dtype=torch.long)


class SegDataModule(pl.LightningDataModule):
    def __init__(self, df_train, df_val, df_test, mount_point="./", batch_size=32, num_workers=4, img_column='img_path', class_column='Classification', seg_column='seg', train_transform=None, valid_transform=None, test_transform=None, drop_last=False):
        super().__init__()

        self.df_train = df_train
        self.df_val = df_val
        self.df_test = df_test
        self.mount_point = mount_point
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_column = img_column
        self.seg_column = seg_column
        self.class_column = class_column
        self.train_transform = train_transform
        self.valid_transform = valid_transform
        self.test_transform = test_transform
        self.drop_last=drop_last

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        base_train_ds = SegDataset(self.df_train, mount_point=self.mount_point, img_column=self.img_column, class_column=self.class_column, seg_column=self.seg_column, transform=self.train_transform)
        self.train_ds = SmartCacheDataset(base_train_ds, num_replace_workers=self.num_workers,replace_rate=0.3, cache_rate=1.0, cache_num=1)

        base_val_ds = SegDataset(self.df_val, mount_point=self.mount_point, img_column=self.img_column, class_column=self.class_column, seg_column=self.seg_column, transform=self.valid_transform)
        self.val_ds = SmartCacheDataset(base_val_ds, num_replace_workers=self.num_workers,replace_rate=0.3, cache_rate=1.0, cache_num=1)

        base_test_ds =SegDataset(self.df_test, mount_point=self.mount_point, img_column=self.img_column, class_column=self.class_column, seg_column=self.seg_column, transform=self.test_transform)
        self.test_ds = SmartCacheDataset(base_test_ds, num_replace_workers=self.num_workers,replace_rate=0.3, cache_rate=1.0, cache_num=1)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True, pin_memory=True, drop_last=self.drop_last, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True, pin_memory=True, drop_last=self.drop_last)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=self.drop_last)