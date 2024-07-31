from nets.classification import NetTarget

import torch.nn as nn
import torch
import torchmetrics

import monai
from monai.networks.nets import DenseNet, DenseNet169, DenseNet201, DenseNet264, SEResNet50, ResNet, resnet18, EfficientNetBN

import pytorch_lightning as pl

class LeftModel(pl.LightningModule):
    def __init__(self, args = None, class_weights=None, base_encoder="DenseNet",seed=42,num_classesR=4,num_classesL=4,num_classes= 4):
        super(LeftModel, self).__init__()

        self.save_hyperparameters()
        self.args = args
        self.class_weights = class_weights

        if(class_weights is not None):
            class_weights = torch.tensor(class_weights).to(torch.float32)

        # self.loss = CustomLossTarget(penalty_weight=0.1,class_weights=class_weights)
        self.loss = nn.CrossEntropyLoss(weight=class_weights)
        self.accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=self.hparams.num_classes)
        self.softmax = nn.Softmax(dim=1)

        if self.hparams.base_encoder == 'DenseNet':
            self.model = monai.networks.nets.DenseNet(spatial_dims=3, in_channels=1,out_channels=self.hparams.num_classes)
        if self.hparams.base_encoder == 'DenseNet169':
            self.model = monai.networks.nets.DenseNet169(spatial_dims=3, in_channels=1,out_channels=self.hparams.num_classes)
        if self.hparams.base_encoder == 'DenseNet201':
            self.model = monai.networks.nets.DenseNet201(spatial_dims=3, in_channels=1,out_channels=512)
        if self.hparams.base_encoder == 'DenseNet264':
            self.model = monai.networks.nets.DenseNet264(spatial_dims=3, in_channels=1,out_channels=self.hparams.num_classes)

        if self.hparams.base_encoder == 'SEResNet50':
            self.model = monai.networks.nets.SEResNet50(spatial_dims=3, in_channels=1, num_classes=self.hparams.num_classes)
        # elif self.hparams.base_encoder == 'ResNet':
        #     self.model = monai.networks.nets.ResNet(spatial_dims=3, n_input_channels=1, num_classes=self.hparams.num_classes)
        elif self.hparams.base_encoder == 'resnet18' or base_encoder=='ResNet':
           self.model = monai.networks.nets.resnet18(spatial_dims=3, n_input_channels=1, num_classes=self.hparams.num_classes)
        if base_encoder == 'efficientnet-b0' or base_encoder == 'efficientnet-b1' or base_encoder == 'efficientnet-b2' or base_encoder == 'efficientnet-b3' or base_encoder == 'efficientnet-b4' or base_encoder == 'efficientnet-b5' or base_encoder == 'efficientnet-b6' or base_encoder == 'efficientnet-b7' or base_encoder == 'efficientnet-b8':
           self.model = monai.networks.nets.EfficientNetBN(base_encoder, spatial_dims=3, in_channels=1, num_classes=self.hparams.num_classes)

        self.fcl = nn.Linear(512, self.hparams.num_classesL)


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.args.lr)
        return optimizer

    def forward(self, x):
        ret =self.model(x)
        retL = self.fcl(ret)

        return retL

    def training_step(self, train_batch, batch_idx):
        x, yR, yL = train_batch
        xL = self(x)

        lossL = self.loss(xL, yL)

        accL =self.accuracy(xL, yL)

        return lossL

    def validation_step(self, val_batch, batch_idx):
        x, yR,yL = val_batch
        xL= self(x)

        lossL = self.loss(xL, yL)
        accL =self.accuracy(xL, yL)


class RightModel(pl.LightningModule):
    def __init__(self, args = None, class_weights=None, base_encoder="DenseNet",seed=42,num_classesR=4,num_classesL=4,num_classes= 4):
        super(RightModel, self).__init__()

        self.save_hyperparameters()
        self.args = args
        self.class_weights = class_weights

        if(class_weights is not None):
            class_weights = torch.tensor(class_weights).to(torch.float32)

        # self.loss = CustomLossTarget(penalty_weight=0.1,class_weights=class_weights)
        self.loss = nn.CrossEntropyLoss(weight=class_weights)
        self.accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=self.hparams.num_classes)
        self.softmax = nn.Softmax(dim=1)

        if self.hparams.base_encoder == 'DenseNet':
            self.model = monai.networks.nets.DenseNet(spatial_dims=3, in_channels=1,out_channels=self.hparams.num_classes)
        if self.hparams.base_encoder == 'DenseNet169':
            self.model = monai.networks.nets.DenseNet169(spatial_dims=3, in_channels=1,out_channels=self.hparams.num_classes)
        if self.hparams.base_encoder == 'DenseNet201':
            self.model = monai.networks.nets.DenseNet201(spatial_dims=3, in_channels=1,out_channels=512)
        if self.hparams.base_encoder == 'DenseNet264':
            self.model = monai.networks.nets.DenseNet264(spatial_dims=3, in_channels=1,out_channels=self.hparams.num_classes)

        if self.hparams.base_encoder == 'SEResNet50':
            self.model = monai.networks.nets.SEResNet50(spatial_dims=3, in_channels=1, num_classes=self.hparams.num_classes)
        # elif self.hparams.base_encoder == 'ResNet':
        #     self.model = monai.networks.nets.ResNet(spatial_dims=3, n_input_channels=1, num_classes=self.hparams.num_classes)
        elif self.hparams.base_encoder == 'resnet18' or base_encoder=='ResNet':
           self.model = monai.networks.nets.resnet18(spatial_dims=3, n_input_channels=1, num_classes=self.hparams.num_classes)
        if base_encoder == 'efficientnet-b0' or base_encoder == 'efficientnet-b1' or base_encoder == 'efficientnet-b2' or base_encoder == 'efficientnet-b3' or base_encoder == 'efficientnet-b4' or base_encoder == 'efficientnet-b5' or base_encoder == 'efficientnet-b6' or base_encoder == 'efficientnet-b7' or base_encoder == 'efficientnet-b8':
           self.model = monai.networks.nets.EfficientNetBN(base_encoder, spatial_dims=3, in_channels=1, num_classes=self.hparams.num_classes)

        self.fcr = nn.Linear(512, self.hparams.num_classesR)


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.args.lr)
        return optimizer

    def forward(self, x):
        ret =self.model(x)
        retR = self.fcr(ret)

        return retR

    def training_step(self, train_batch, batch_idx):
        x, yR, yL = train_batch
        xR = self(x)

        lossR = self.loss(xR, yR)

        accR =self.accuracy(xR, yR)

        return lossR

    def validation_step(self, val_batch, batch_idx):
        x, yR,yL = val_batch
        xR= self(x)

        lossR = self.loss(xR, yR)
        accR =self.accuracy(xR, yR)

