import argparse

import math
import os
import pandas as pd
import numpy as np
import SimpleITK as sitk

import torch

from nets.classification import Net, SegNet
from loaders.cleft_dataset import DataModule, SegDataModule
from transforms.volumetric import TrainTransforms, EvalTransforms, SegTrainTransforms, SegEvalTransforms, NoTransform, NoEvalTransform
from callbacks.logger import ImageLogger, SegImageLogger, ImageLoggerNeptune

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.loggers import NeptuneLogger, TensorBoardLogger

from sklearn.utils import class_weight
import time

def main(args):
    torch.cuda.empty_cache()
    start_time = time.time()

    if(os.path.splitext(args.csv_train)[1] == ".csv"):
        df_train = pd.read_csv(args.csv_train)
        df_val = pd.read_csv(args.csv_valid)
        df_test = pd.read_csv(args.csv_test)
    else:
        df_train = pd.read_parquet(args.csv_train)
        df_val = pd.read_parquet(args.csv_valid)
        df_test = pd.read_parquet(args.csv_test)

    # unique_classes = np.sort(np.unique(df_train[args.class_column]))
    # unique_classes= np.delete(unique_classes, 3)


    df_filtered_train = df_train.dropna(subset=[args.class_column])
    df_filtered_train = df_filtered_train.reset_index(drop=True)
    unique_classes = np.unique(df_filtered_train[args.class_column])
    unique_class_weights = np.array(class_weight.compute_class_weight(class_weight='balanced', classes=unique_classes, y=df_filtered_train[args.class_column]))


    class_replace = {}
    for cn, cl in enumerate(unique_classes):
        class_replace[int(cl)] = cn
    print(unique_classes, unique_class_weights, class_replace)

    df_filtered_train[args.class_column] = df_filtered_train[args.class_column].replace(class_replace)

    df_val[args.class_column] = df_val[args.class_column].replace(class_replace)
    df_test[args.class_column] = df_test[args.class_column].replace(class_replace)

    img_size = df_filtered_train.shape[0]


    if args.seg_column is None:

        data = DataModule(df_filtered_train, df_val, df_test, mount_point=args.mount_point, batch_size=args.batch_size, num_workers=args.num_workers, img_column=args.img_column, class_column=args.class_column, train_transform= TrainTransforms(img_size), valid_transform=EvalTransforms(img_size), test_transform=EvalTransforms(img_size))

        model = Net(args, num_classes=unique_classes.shape[0], class_weights=unique_class_weights, base_encoder=args.base_encoder)

        torch.backends.cudnn.benchmark = True # optimizing the training process when using NVIDIA GPUs with CUDA. Input must be fixed size

        if args.neptune_project:
            image_logger = ImageLoggerNeptune()
        else:
            image_logger = ImageLogger()
    else:

        data = SegDataModule(df_train, df_val, df_test, mount_point=args.mount_point, batch_size=args.batch_size, num_workers=args.num_workers, img_column=args.img_column, class_column=args.class_column, train_transform=SegTrainTransforms(img_size), valid_transform=SegEvalTransforms(img_size), test_transform=SegEvalTransforms(img_size))

        model = SegNet(args, num_classes=unique_classes.shape[0], class_weights=unique_class_weights, base_encoder=args.base_encoder)

        image_logger = SegImageLogger()


    checkpoint_callback = ModelCheckpoint(
        dirpath=args.out,
        filename='{epoch}-{val_loss:.3f}',
        save_top_k=2,
        monitor='val_loss'
    )

    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=args.patience, verbose=True, mode="min")

    if args.tb_dir:
        logger = TensorBoardLogger(save_dir=args.tb_dir, name=args.tb_name)
    elif args.neptune_project:
        logger = NeptuneLogger(project=args.neptune_project, tags=args.neptune_tag, api_key=os.environ["NEPTUNE_API_TOKEN"])
    else:
        logger = None

    trainer = Trainer(
        logger=logger,
        max_epochs=args.epochs,
        callbacks=[early_stop_callback, checkpoint_callback, image_logger],
        devices=torch.cuda.device_count(),
        accelerator="gpu",
        strategy=DDPStrategy(find_unused_parameters=False),
        log_every_n_steps=args.log_every_n_steps
    )

    trainer.fit(model, datamodule=data, ckpt_path=args.model)
    torch.cuda.empty_cache()
    end_time = time.time()

    # format time
    hours, rem = divmod(end_time - start_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print("Training took {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))

if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Cleft classification Training')
    parser.add_argument('--csv_train', required=True, type=str, help='Train CSV')
    parser.add_argument('--csv_valid', required=True, type=str, help='Valid CSV')
    parser.add_argument('--csv_test', required=True, type=str, help='Test CSV')
    parser.add_argument('--img_column', type=str, default='img', help='Name of image column')
    parser.add_argument('--class_column', type=str, default='Label', help='Name of class column')
    parser.add_argument('--seg_column', type=str, default=None, help='Name of segmentation column')
    parser.add_argument('--base_encoder', type=str, default='efficientnet-b0', help='Type of base encoder')
    parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float, help='Learning rate')
    parser.add_argument('--model', help='Model path to continue training', type=str, default=None)
    parser.add_argument('--epochs', help='Max number of epochs', type=int, default=400)
    parser.add_argument('--log_every_n_steps', help='Log every n steps', type=int, default=10)
    parser.add_argument('--out', help='Output', type=str, default="./")
    parser.add_argument('--mount_point', help='Dataset mount directory', type=str, default="./")
    parser.add_argument('--num_workers', help='Number of workers for loading', type=int, default=4)
    parser.add_argument('--batch_size', help='Batch size', type=int, default=2)
    parser.add_argument('--patience', help='Patience for early stopping', type=int, default=50)

    # tensorboard
    parser.add_argument('--tb_dir', help='Tensorboard output dir', type=str, default=None)
    parser.add_argument('--tb_name', help='Tensorboard experiment name', type=str, default="classification")

    # neptune
    parser.add_argument('--neptune_project', help='Neptune project name', type=str, default=None)
    parser.add_argument('--neptune_tag', help='Neptune tag', type=str, default="Left Canine Classification")


    args = parser.parse_args()

    main(args)
