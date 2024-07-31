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

import pandas as pd
import numpy as np

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

        row = self.df.loc[idx]

        img = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.mount_point, row[self.img_column])))

        if self.transform:
            img = self.transform(img)

        if pd.isna(row[self.class_column]):
            cl =11 #11 is the class for missing data
        else:
            cl = int(row[self.class_column])

        return img, torch.tensor(cl, dtype=torch.long)

class DatasetFC(Dataset):
    def __init__(self, df, mount_point = "./", img_column='img', nb_classes=2,class_column1 ="Right", class_column2="Left", transform=None):
        self.df = df
        self.mount_point = mount_point
        self.transform = transform
        self.img_column = img_column
        self.class_column_R = class_column1
        self.class_column_L = class_column2
        self.nb_classes = nb_classes

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.loc[idx]
        img = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.mount_point, row[self.img_column])))

        if self.transform:
            img = self.transform(img)

        cl1 = int(row[self.class_column_R])
        cl2 = int(row[self.class_column_L])

        return img, torch.tensor(cl1, dtype=torch.long), torch.tensor(cl2, dtype=torch.long)

class Datasetarget(Dataset):
    def __init__(self, df, mount_point = "./", img_column='img', nb_classes=2,class_column1 ="Right", class_column2="Left", transform=None):
        self.df = df
        self.mount_point = mount_point
        self.transform = transform
        self.img_column = img_column
        self.class_column_R = class_column1
        self.class_column_L = class_column2
        self.nb_classes = nb_classes

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.loc[idx]
        img = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.mount_point, row[self.img_column])))

        if self.transform:
            img = self.transform(img)

        target_vector = np.zeros(self.nb_classes)

        #convert into str both columns
        if not pd.isna(row[self.class_column_R]):
            target_vector[int(row[self.class_column_R])] = 1
        if not pd.isna(row[self.class_column_L]):
            target_vector[int(row[self.class_column_L])] = 1
        if not pd.isna(row[self.class_column_R]) and not pd.isna(row[self.class_column_L]):
            target_vector[int(row[self.class_column_R])] = 0.5
            target_vector[int(row[self.class_column_L])] = 0.5


        return img, torch.tensor(target_vector, dtype=torch.float)

class DataModule(pl.LightningDataModule):
    def __init__(self, df_train,df_val,df_test,df_special, mount_point="./", batch_size=32, num_workers=4, img_column='img_path', class_column='Classification', train_transform=None, valid_transform=None, test_transform=None,special_transform=None,drop_last=False,seed=42):
        super().__init__()

        self.df_train = df_train
        self.df_val = df_val
        self.df_test = df_test
        self.df_special = df_special
        self.mount_point = mount_point
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_column = img_column
        self.class_column = class_column
        self.train_transform = train_transform
        self.valid_transform = valid_transform
        self.test_transform = test_transform
        self.special_transform = special_transform
        self.drop_last=drop_last

    def _set_seed(self, seed):
        torch.manual_seed(seed)

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        self.train_ds = BasicDataset(self.df_train, mount_point=self.mount_point, img_column=self.img_column, class_column=self.class_column, transform=self.train_transform)

        if self.df_special is not None:
            self.special_ds = BasicDataset(self.df_special, mount_point=self.mount_point, img_column=self.img_column, class_column=self.class_column, transform=self.special_transform)

        self.val_ds = BasicDataset(self.df_val, mount_point=self.mount_point, img_column=self.img_column, class_column=self.class_column, transform=self.valid_transform)

        self.test_ds = BasicDataset(self.df_test, mount_point=self.mount_point, img_column=self.img_column, class_column=self.class_column, transform=self.test_transform)


    def train_dataloader(self):
        if self.df_special is not None:
            print('CONCATENATING SPECIAL DATASET')
            #concatenate the special dataset with the training dataset
            train_special_ds = torch.utils.data.ConcatDataset([self.train_ds, self.special_ds])
            return DataLoader(train_special_ds, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True, pin_memory=True, drop_last=self.drop_last, shuffle=True)
        else:
            return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True, pin_memory=True, drop_last=self.drop_last, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True, pin_memory=True, drop_last=self.drop_last)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=self.drop_last)


class DataModuleT(pl.LightningDataModule):
    def __init__(self, df_train,df_val,df_test,df_special, mount_point="./", batch_size=32, num_workers=4, img_column='img_path',  class_column1= 'Right',class_column2='Left', nb_classes=2,train_transform=None, valid_transform=None, test_transform=None,special_transform=None,drop_last=False,seed=42):
        super().__init__()

        self.df_train = df_train
        self.df_val = df_val
        self.df_test = df_test
        self.df_special = df_special
        self.mount_point = mount_point
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_column = img_column
        self.class_column1 = class_column1
        self.class_column2 = class_column2
        self.nb_classes = nb_classes
        self.train_transform = train_transform
        self.valid_transform = valid_transform
        self.test_transform = test_transform
        self.special_transform = special_transform
        self.drop_last=drop_last


    def _set_seed(self, seed):
        torch.manual_seed(seed)

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        self.train_ds = Datasetarget(self.df_train, mount_point=self.mount_point, img_column=self.img_column,  class_column1= self.class_column1, class_column2 = self.class_column2,nb_classes=self.nb_classes,transform=self.train_transform)

        if self.df_special is not None:
            self.special_ds = Datasetarget(self.df_special, mount_point=self.mount_point, img_column=self.img_column, class_column1=self.class_column1,class_column2=self.class_column2,nb_classes=self.nb_classes, transform=self.special_transform)

        self.val_ds = Datasetarget(self.df_val, mount_point=self.mount_point, img_column=self.img_column, class_column1=self.class_column1,class_column2=self.class_column2, nb_classes=self.nb_classes, transform=self.valid_transform)

        self.test_ds = Datasetarget(self.df_test, mount_point=self.mount_point, img_column=self.img_column, class_column1 =self.class_column1, class_column2=self.class_column2,nb_classes=self.nb_classes, transform=self.test_transform)


    def train_dataloader(self):
        if self.df_special is not None:
            #concatenate the special dataset with the training dataset
            train_special_ds = torch.utils.data.ConcatDataset([self.train_ds, self.special_ds])
            return DataLoader(train_special_ds, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True, pin_memory=True, drop_last=self.drop_last, shuffle=True)
        else:
            return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True, pin_memory=True, drop_last=self.drop_last, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True, pin_memory=True, drop_last=self.drop_last)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=self.drop_last)


class DataModuleFC(pl.LightningDataModule):
    def __init__(self, df_train,df_val,df_test,df_special, mount_point="./", batch_size=32, num_workers=4, img_column='img_path',  class_column1= 'Right',class_column2='Left', nb_classes=2,train_transform=None, valid_transform=None, test_transform=None,special_transform=None,drop_last=False,seed=42):
        super().__init__()

        self.df_train = df_train
        self.df_val = df_val
        self.df_test = df_test
        self.df_special = df_special
        self.mount_point = mount_point
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_column = img_column
        self.class_column1 = class_column1
        self.class_column2 = class_column2
        self.nb_classes = nb_classes
        self.train_transform = train_transform
        self.valid_transform = valid_transform
        self.test_transform = test_transform
        self.special_transform = special_transform
        self.drop_last=drop_last

    def _set_seed(self, seed):
        torch.manual_seed(seed)

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        self.train_ds = DatasetFC(self.df_train, mount_point=self.mount_point, img_column=self.img_column,  class_column1= self.class_column1, class_column2 = self.class_column2,nb_classes=self.nb_classes,transform=self.train_transform)

        if self.df_special is not None:
            self.special_ds = DatasetFC(self.df_special, mount_point=self.mount_point, img_column=self.img_column, class_column1=self.class_column1,class_column2=self.class_column2,nb_classes=self.nb_classes, transform=self.special_transform)

        self.val_ds = DatasetFC(self.df_val, mount_point=self.mount_point, img_column=self.img_column, class_column1=self.class_column1,class_column2=self.class_column2, nb_classes=self.nb_classes, transform=self.valid_transform)

        self.test_ds = DatasetFC(self.df_test, mount_point=self.mount_point, img_column=self.img_column, class_column1 =self.class_column1, class_column2=self.class_column2,nb_classes=self.nb_classes, transform=self.test_transform)


    def train_dataloader(self):
        if self.df_special is not None:
            #concatenate the special dataset with the training dataset
            train_special_ds = torch.utils.data.ConcatDataset([self.train_ds, self.special_ds])
            return DataLoader(train_special_ds, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True, pin_memory=True, drop_last=self.drop_last, shuffle=True)
        else:
            return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True, pin_memory=True, drop_last=self.drop_last, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True, pin_memory=True, drop_last=self.drop_last)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=self.drop_last)