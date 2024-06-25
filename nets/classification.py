import math
from typing import Optional, Tuple

import numpy as np

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from torchvision import models
from torchvision import transforms
import torchmetrics

import monai
from monai.networks.nets import AutoEncoder
from monai.networks.blocks import Convolution

from useful_readibility import printRed, printBlue,printGreen
import pytorch_lightning as pl

from pl_bolts.transforms.dataset_normalizations import (
    imagenet_normalization
)

class DotProductAttention(nn.Module):
    def __init__(self):
        super(DotProductAttention, self).__init__()

    def forward(self, query: Tensor, value: Tensor) -> Tuple[Tensor, Tensor]:
        batch_size, hidden_dim, input_size = query.size(0), query.size(2), value.size(1)

        score = torch.bmm(query, value.transpose(1, 2))
        attn = F.softmax(score.view(-1, input_size), dim=1).view(batch_size, -1, input_size)
        context = torch.bmm(attn, value).view(batch_size, -1, hidden_dim)
        context = torch.sum(context, dim=1)

        return context, attn

class SelfAttention(nn.Module):
    def __init__(self, in_units, out_units):
        super(SelfAttention, self).__init__()

        self.W1 = nn.Linear(in_units, out_units)
        self.V = nn.Linear(out_units, 1)

    def forward(self, query, values):

        score = self.V(nn.Tanh()(self.W1(query)))

        score = nn.Sigmoid()(score)
        sum_score = torch.sum(score, 1, keepdim=True)
        attention_weights = score / sum_score

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * values
        context_vector = torch.sum(context_vector, dim=1)

        return context_vector, score

class TimeDistributed(nn.Module):
    def __init__(self, module):
        super(TimeDistributed, self).__init__()
        self.module = module

    def forward(self, input_seq):
        assert len(input_seq.size()) > 2

        # reshape input data --> (samples * timesteps, input_size)
        # squash timesteps

        size = input_seq.size()

        batch_size = size[0]
        time_steps = size[1]

        size_reshape = [batch_size*time_steps] + list(size[2:])
        reshaped_input = input_seq.contiguous().view(size_reshape)

        output = self.module(reshaped_input)

        output_size = output.size()
        output_size = [batch_size, time_steps] + list(output_size[1:])
        output = output.contiguous().view(output_size)

        return output

class CustomAccuracy(nn.Module):
    def __init__(self):
        super(CustomAccuracy, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self,preds,targets):
        preds_sig = self.sigmoid(preds)


        demi_len = int(preds_sig.shape[1]/2)
        for j in range(preds_sig.shape[0]):

            classR = None
            classL = None

            best_probR,idxR = torch.max(preds_sig[j,:demi_len],dim=0)
            best_probL,idxL = torch.max(preds_sig[j,demi_len:],dim=0)

            if best_probR.item() > 0.59:
               classR = idxR
            if best_probL.item() > 0.59:
                classL = idxL
            if classR is None and classL is None:
                if best_probR > best_probL:
                    classR = idxR
                else:
                    classL = idxL
            #find class from target vector,where '1' or '0.5' is present
            #if 1 is present
            if 1 in targets[j,:demi_len]:
                targetR = torch.where(targets[j,:demi_len] == 1)[0]
            if 1 in targets[j,demi_len:]:
                targetL = torch.where(targets[j,demi_len:] == 1)[0]
            if 0.5 in targets[j,:]:
                targetR = torch.where(targets[j,:demi_len] == 0.5)[0]
                targetL = torch.where(targets[j,demi_len:] == 0.5)[0]



class CustomLossTarget(torch.nn.Module):
    def __init__(self, penalty_weight=0.1, class_weights=None):
        super(CustomLossTarget, self).__init__()
        self.base_loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
        self.penalty_weight = penalty_weight
        self.sigmoid = nn.Sigmoid()



    def forward(self, preds, targets):
        # Compute the base loss
        base_loss = self.base_loss_fn(preds,targets)
        #create prediction_vector
        preds_sig = self.sigmoid(preds)
        preds_sigF = F.sigmoid(preds)
        demi_len = int(preds_sig.shape[1]/2)
        prediction_vector = torch.zeros(preds_sig.shape[0],preds_sig.shape[1])

        penalty_fp = 0
        penalty_fn = 0
        penalty_class = 0
        penalty_bonus = 0
        penalty_weight = self.penalty_weight
        for j in range(preds_sig.shape[0]):

            noneR=0
            noneL=0

            best_probR,idxR = torch.max(preds_sig[j,:demi_len],dim=0)
            best_probL,idxL = torch.max(preds_sig[j,demi_len:],dim=0)

            if best_probR.item() > 0.65:
                prediction_vector[j,idxR] = 1

            if best_probL.item() > 0.65:
                prediction_vector[j,idxL+demi_len] = 1

            #check the sum of the vector, if it's 2, we need to divide by 2 each value, if it's 0, we need to put the best score to 1
            if prediction_vector[j].sum() > 1:
                prediction_vector[j] = prediction_vector[j]/prediction_vector[j].sum()
            elif prediction_vector[j].sum() == 0:
                if best_probR > best_probL:
                    prediction_vector[j,idxR] = 1
                else:
                    prediction_vector[j,idxL+demi_len] = 1

            # # # # If there is a prediction for a non-existing canine, penalize
            # left_false_positive = True if (prediction_vector[j, :3].sum() > 0) & (targets[j, :3].sum() == 0) else False
            # right_false_positive = True if (prediction_vector[j, 3:].sum() > 0) & (targets[j, 3:].sum() == 0) else False

            # If there is no prediction for an existing canine, penalize
            left_false_negative = True if (prediction_vector[j, :3].sum() == 0) & (targets[:, :3].sum() > 0) else False
            right_false_negative = True if (prediction_vector[:, 3:].sum() == 0) & (targets[:, 3:].sum() > 0) else False

            # if (left_false_positive or right_false_positive) or (left_false_positive and right_false_positive):
            #     penalty_fp +=1
            if left_false_negative or right_false_negative:
                penalty_fn +=1

            # # penalize if it predicts a class 2 (resp 5) when it's a class 0 (resp 3) and vice versa
            # if (prediction_vector[j, 0] == 1 and targets[j, 2] == 1) or (prediction_vector[j, 2] == 1 and targets[j, 0] == 1):
            #     penalty_class+=0.5
            # if (prediction_vector[j, 3] == 1 and targets[j, 5] == 1) or (prediction_vector[j, 5] == 1 and targets[j, 3] == 1):
            #     penalty_class+=0.5



        print('preds_sig',preds_sig)
        printBlue(f'targets {targets}')

        penalty = penalty_weight * (penalty_fp + penalty_fn + penalty_class-penalty_bonus)
        # penalty = penalty_weight * penalty_fn
        printRed(f'penalty: {round(penalty,3)}')

        total_loss = base_loss + penalty
        return total_loss

class NetTarget(pl.LightningModule):
    def __init__(self, args = None, class_weights=None, base_encoder="DenseNet",seed=42,num_classes=6):
        super(NetTarget, self).__init__()

        self.save_hyperparameters()
        self.args = args
        self._set_seed(seed)
        self.class_weights = class_weights

        if(class_weights is not None):
            class_weights = torch.tensor(class_weights).to(torch.float32)

        self.loss = CustomLossTarget(penalty_weight=0.1,class_weights=class_weights)
        self.accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=self.hparams.num_classes)

        if self.hparams.base_encoder == 'DenseNet':
            self.model = monai.networks.nets.DenseNet(spatial_dims=3, in_channels=1,out_channels=self.hparams.num_classes)
        if self.hparams.base_encoder == 'DenseNet169':
            self.model = monai.networks.nets.DenseNet169(spatial_dims=3, in_channels=1,out_channels=self.hparams.num_classes)
        if self.hparams.base_encoder == 'DenseNet201':
            self.model = monai.networks.nets.DenseNet201(spatial_dims=3, in_channels=1,out_channels=self.hparams.num_classes)
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

    def _set_seed(self, seed):
        torch.manual_seed(seed)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.args.lr)
        return optimizer

    def forward(self, x):
        ret =self.model(x)
        return ret

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        x = self(x)

        loss = self.loss(x, y)
        self.log('train_loss', loss)

        self.accuracy(x, y)
        self.log("train_acc", self.accuracy)

        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        x = self(x)

        loss = self.loss(x, y)
        self.log('val_loss', loss, sync_dist=True)

        self.accuracy(x, y)
        self.log("val_acc", self.accuracy)

class Net(pl.LightningModule):
    def __init__(self, args = None, class_weights=None, base_encoder="efficientnet-b0", seed = 42,num_classes=3):
        super(Net, self).__init__()

        self.save_hyperparameters()
        self.args = args
        self._set_seed(seed)
        self.class_weights = class_weights

        if(class_weights is not None):
            class_weights = torch.tensor(class_weights).to(torch.float32)

        self.loss = nn.CrossEntropyLoss(weight=class_weights)
        self.bce_loss = nn.BCEWithLogitsLoss(weight=class_weights)

        self.accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=self.hparams.num_classes)
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

        if self.hparams.base_encoder == 'DenseNet':
            self.model = monai.networks.nets.DenseNet(spatial_dims=3, in_channels=1,out_channels=self.hparams.num_classes)
        if self.hparams.base_encoder == 'DenseNet169':
            self.model = monai.networks.nets.DenseNet169(spatial_dims=3, in_channels=1,out_channels=self.hparams.num_classes)
        if self.hparams.base_encoder == 'DenseNet201':
            self.model = monai.networks.nets.DenseNet201(spatial_dims=3, in_channels=1,out_channels=self.hparams.num_classes)
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

    def _set_seed(self, seed):
        torch.manual_seed(seed)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.args.lr)
        return optimizer

    def forward(self, x):

        ret =self.model(x)

        return ret

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        x = self(x)

        loss = self.loss(x, y)
        self.log('train_loss', loss)

        bce_loss = self.bce_loss(x, y)
        self.log('train_bce_loss', bce_loss)

        self.accuracy(x, y)
        self.log("train_acc", self.accuracy)

        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        x = self(x)

        loss = self.loss(x, y)
        self.log('val_loss', loss, sync_dist=True)
        bce_loss = self.bce_loss(x, y)
        self.log('val_bce_loss', bce_loss, sync_dist=True)

        self.accuracy(x, y)
        self.log("val_acc", self.accuracy)


class SegNet(pl.LightningModule):
    def __init__(self, args = None, class_weights=None, base_encoder="efficientnet-b0", num_classes=3):
        super(SegNet, self).__init__()

        self.save_hyperparameters()
        self.args = args

        self.class_weights = class_weights

        if(class_weights is not None):
            class_weights = torch.tensor(class_weights).to(torch.float32)

        self.loss = nn.CrossEntropyLoss(weight=class_weights)
        self.accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=self.hparams.num_classes)

        # if self.hparams.base_encoder == 'SEResNet50':
        #     self.model = monai.networks.nets.SEResNet50(spatial_dims=3, in_channels=2, num_classes=self.hparams.num_classes)
        # elif self.hparams.base_encoder == 'resnet18':
        #     self.model = monai.networks.nets.resnet18(spatial_dims=3, n_input_channels=2, num_classes=self.hparams.num_classes)
        # else:
        #     self.model = monai.networks.nets.EfficientNetBN(self.hparams.base_encoder, spatial_dims=3, in_channels=2, num_classes=self.hparams.num_classes)

        if hasattr(monai.networks.nets, self.hparams.base_encoder):
            template_model = getattr(monai.networks.nets, self.hparams.base_encoder)
        elif hasattr(models, self.hparams.base_encoder):
            template_model = getattr(models, self.hparams.base_encoder)
        else:
            raise "{base_encoder} not in monai networks or torchvision".format(base_encoder=self.hparams.base_encoder)

        model_params = eval('dict(%s)' % self.hparams.base_encoder_params.replace(' ',''))

        self.model = template_model(**model_params)


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.args.lr)
        return optimizer

    def forward(self, x):
        ret =self.model(x)

        return ret

    def training_step(self, train_batch, batch_idx):
        x0, x1, y = train_batch

        x = torch.cat([x0, x1], dim=1)

        x = self(x)

        loss = self.loss(x, y)

        self.log('train_loss', loss)
        self.accuracy(x, y)
        self.log("train_acc", self.accuracy)

        return loss

    def validation_step(self, val_batch, batch_idx):
        x0, x1, y = val_batch

        x = torch.cat([x0, x1], dim=1)
        x = self(x)

        loss = self.loss(x, y)

        self.log('val_loss', loss, sync_dist=True)
        self.accuracy(x, y)
        self.log("val_acc", self.accuracy)