import argparse
from argparse import Namespace

import math
import os
import pandas as pd
import numpy as np
import SimpleITK as sitk

import torch

from nets.classification import Net, SegNet
from loaders.cleft_dataset import DataModule, SegDataModule
from transforms.volumetric import TrainTransforms, EvalTransforms,SpecialTransforms, SegTrainTransforms, SegEvalTransforms, NoTransform, NoEvalTransform
import classification_predict
import classification_eval_VAXI
from useful_readibility import printRed, printGreen,printOrange, printBlue
from callbacks.logger import tensorboard_neptune_logger

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.loggers import NeptuneLogger, TensorBoardLogger

from sklearn.utils import class_weight
from sklearn.model_selection import StratifiedKFold, train_test_split
import time
import glob

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # If using CuDNN

def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory

def get_val_loss(checkpoint_dir):
    val_loss_dict = {}
    for file in os.listdir(checkpoint_dir):
        if file.endswith(".ckpt"):
            val_loss = float(file.split("-")[1].split("=")[1].split('.ckpt')[0])
            val_loss_dict[val_loss] = file
    return val_loss_dict


def get_best_checkpoint(checkpoint_dir):
    # Create a dictionnary with all the val loss in keys
    # for each model saved in the checkpoint_dir folder
    val_loss_dict = get_val_loss(checkpoint_dir)

    sorted_val_loss = sorted(val_loss_dict.keys())
    best_model = val_loss_dict[sorted_val_loss[0]]
    best_val_loss = sorted_val_loss[0]

    return best_model,best_val_loss

def get_argparse_dict(parser):
    # Get the default arguments from the parser
    default = {}
    for action in parser._actions:
        if action.dest != "help":
            default[action.dest] = action.default
    return default


def main(args):
    torch.cuda.empty_cache()
    set_seed(args.seed)
    start_time = time.time()

    img_size = args.img_size

    # Load data
    if(os.path.splitext(args.csv)[1] == ".csv"):
        df = pd.read_csv(args.csv)
    else:
        df = pd.read_parquet(args.csv)

    df_filtered_train = df.dropna(subset=[args.class_column])
    df_filtered_train = df_filtered_train.reset_index(drop=True)

    unique_classes = np.unique(df_filtered_train[args.class_column])
    unique_class_weights = np.array(class_weight.compute_class_weight(class_weight='balanced', classes=unique_classes, y=df_filtered_train[args.class_column]))

    class_replace = {}
    for cn, cl in enumerate(unique_classes):
        class_replace[int(cl)] = cn
    print(unique_classes, unique_class_weights, class_replace)

    #save the parameters of the model
    outpath_modelInfo = args.out + "/modelParams.csv"
    if not os.path.exists(outpath_modelInfo):
        df_modelInfo = pd.DataFrame(columns=['model', 'img_size', 'num_classes', 'class_weights','args'])
        df_modelInfo = df_modelInfo._append({'model': args.base_encoder, 'img_size': args.img_size, 'num_classes': unique_classes.shape[0], 'class_weights': unique_class_weights, 'args': args}, ignore_index=True)
        df_modelInfo.to_csv(outpath_modelInfo, index=False)
    else:
        df_modelInfo = pd.read_csv(outpath_modelInfo)
        df_modelInfo = df_modelInfo._append({'model': args.base_encoder, 'img_size': args.img_size, 'num_classes': unique_classes.shape[0], 'class_weights': unique_class_weights, 'args': args}, ignore_index=True)
        df_modelInfo.to_csv(outpath_modelInfo, index=False)

    df_filtered_train[args.class_column] = df_filtered_train[args.class_column].replace(class_replace)

    # Load special data
    if args.csv_special is not None:
        df_special = pd.read_csv(args.csv_special)
        df_filtered_special = df_special.dropna(subset=[args.class_column])
        df_filtered_special = df_filtered_special.reset_index(drop=True)
        df_filtered_special[args.class_column] = df_filtered_special[args.class_column].replace(class_replace)
        special_tf = SpecialTransforms(img_size)
    else:
        df_filtered_special = None
        special_tf=None


    # Cross validation
    nb_fold_perEncoder= args.split / len(args.base_encoder)
    if not nb_fold_perEncoder.is_integer():
        # increment number of split to be divisible by the number of base encoder to test
        args.split = math.ceil(args.split / len(args.base_encoder)) * len(args.base_encoder)
        printOrange('Number of splits has been incremented to be divisible by the number of base encoder to test, new split: {args.split}')
        nb_fold_perEncoder = args.split / len(args.base_encoder)

    kf = StratifiedKFold(n_splits=args.split, shuffle=True, random_state=args.seed)


    power2 = math.ceil(math.log2(img_size)) # get the next power of 2
    pad_size = ((2**power2) - args.img_size) // 2
    print('pad_size',pad_size)

    best_metric = 0
    best_model_fold = ""

    if len(args.base_encoder) != 1:
        nb_split_perEncoder= args.split / len(args.base_encoder)
    else:
        nb_split_perEncoder = args.split


    idx_changeEncoder = 0

    for i, (train_index, test_index) in enumerate(kf.split(df_filtered_train, df_filtered_train[args.class_column])):
        #################################
        #                               #
        #          Training             #
        #                               #
        #################################
        print('len train_index',len(train_index))
        print('len test_index',len(test_index))
        df_train = df_filtered_train.iloc[train_index]
        df_test = df_filtered_train.iloc[test_index]

        #create csv for testing set and training
        outpath_test = args.out + f"/fold_{i}/test.csv"
        if not os.path.exists(os.path.dirname(outpath_test)):
            os.makedirs(os.path.dirname(outpath_test))
        df_test.to_csv(outpath_test, index=False)

        outpath_train = args.out + f"/fold_{i}/train.csv"
        if not os.path.exists(os.path.dirname(outpath_train)):
            os.makedirs(os.path.dirname(outpath_train))
        df_train.to_csv(outpath_train, index=False)


        # split train in training and validation set with args.val_size
        df_train_inner, df_val = train_test_split(df_train, test_size=args.val_size, random_state=args.seed, stratify=df_train[args.class_column])

        df_train_inner = df_train_inner.reset_index(drop=True)
        df_val = df_val.reset_index(drop=True)
        df_test = df_test.reset_index(drop=True)

        if i > int(nb_split_perEncoder)-1:
            idx_changeEncoder += 1
            nb_split_perEncoder+= nb_split_perEncoder

        base_encoder = args.base_encoder[idx_changeEncoder]
        printBlue(f'base_encoder in use {base_encoder}')


        if args.seg_column is None:
            data = DataModule(df_train_inner, df_val,df_test, df_filtered_special, mount_point=args.mount_point, batch_size=args.batch_size, num_workers=args.num_workers, img_column=args.img_column, class_column=args.class_column,
                              train_transform= TrainTransforms(img_size,pad_size), valid_transform=EvalTransforms(img_size),test_transform=EvalTransforms(img_size), special_transform = special_tf,seed=args.seed)


            #restart the training to the fold of the model then continue
            if args.checkpoint is not None:
                #find at what folder its stopped:
                folder_last_model = os.path.dirname(args.checkpoint).split('/')[-1]
                #take the int in the name fold_X
                folder_nb_lm = int(folder_last_model.split('_')[-1])
                if folder_nb_lm > i:
                    continue
                elif folder_nb_lm == i:
                    prediction_folder = os.path.dirname(args.checkpoint).replace(folder_last_model,'Predictions')+f'/{folder_last_model}'
                    if os.path.exists(prediction_folder):
                        continue
                    model = Net.load_from_checkpoint(args.checkpoint, num_classes=unique_classes.shape[0], class_weights=unique_class_weights, base_encoder=base_encoder,seed=args.seed)
                    ckpt_path = args.checkpoint
                else:
                    model = Net(args, num_classes=unique_classes.shape[0], class_weights=unique_class_weights, base_encoder=base_encoder,seed=args.seed)
                    ckpt_path = None
            # if args.model is not None and i==0:
            #     model = Net.load_from_checkpoint(args.model, num_classes=unique_classes.shape[0], class_weights=unique_class_weights, base_encoder=base_encoder,seed=args.seed)
            else:
                model = Net(args, num_classes=unique_classes.shape[0], class_weights=unique_class_weights, base_encoder=base_encoder,seed=args.seed)
                ckpt_path = None

            # torch.backends.cudnn.benchmark = True

        # else:
        #     data = SegDataModule(df_train_inner, df_val,df_test, mount_point=args.mount_point, batch_size=args.batch_size, num_workers=args.num_workers, img_column=args.img_column, class_column=args.class_column,
        #                           train_transform=SegTrainTransforms(img_size), valid_transform=SegEvalTransforms(img_size),test_transform=SegEvalTransforms(img_size))

        #     model = SegNet(args, num_classes=unique_classes.shape[0], class_weights=unique_class_weights, base_encoder=base_encoder)

        # Create a folder for each fold
        checkpoint_dir =args.out + f"/fold_{i}"
        checkpoint_dir = ensure_directory_exists(checkpoint_dir)
        checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename='{epoch}-{val_loss:.3f}',
            save_top_k=2,
            monitor='val_loss'
        )

        early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=args.patience, verbose=True, mode="min")

        try:
            logger,image_logger = tensorboard_neptune_logger(args)
            if logger is None:
                raise ValueError("No logger")
            if image_logger is None:
                raise ValueError("No image logger")
            if logger is None and image_logger is None:
                raise ValueError("No logger and image logger")

        except Exception as e:
            printRed('Error in logger setup')
            print(e)
            return

        trainer = Trainer(
            logger=logger,
            max_epochs=args.epochs,
            callbacks=[early_stop_callback, checkpoint_callback,image_logger],
            devices=torch.cuda.device_count(),
            accelerator="gpu",
            strategy=DDPStrategy(find_unused_parameters=False),
            log_every_n_steps=args.log_every_n_steps,
            precision=16, # reduce memory usage
        )


        trainer.fit(model, datamodule=data, ckpt_path=ckpt_path)
        torch.cuda.empty_cache()

        #################################
        #                               #
        #          Prediction           #
        #                               #
        #################################

        # Get the best model
        best_model,best_val_loss = get_best_checkpoint(checkpoint_dir)
        best_model = os.path.join(checkpoint_dir, best_model)

        printBlue(f'Best model of the fold {i}: {best_model}')
        prediction_args = get_argparse_dict(classification_predict.get_argparse())
        prediction_args['csv']= outpath_test
        prediction_args['csv_train']= outpath_train
        prediction_args['model']= best_model
        prediction_args['mount_point']= args.mount_point

        prediction_args['img_column']= args.img_column
        prediction_args['class_column']= args.class_column
        prediction_args['seg_column']= args.seg_column
        prediction_args['pred_column']= "Prediction"

        prediction_args['base_encoder']= base_encoder
        prediction_args['img_size']= args.img_size

        outdir_prediction = args.out + f"/Predictions/fold_{i}/"
        prediction_args['out']= outdir_prediction

        prediction_args['num_workers']= args.num_workers
        prediction_args['batch_size']= args.batch_size
        prediction_args['lr']= args.lr

        prediction_args['seed']= args.seed

        prediction_args= Namespace(**prediction_args)
        ext = os.path.splitext(outpath_test)[1]
        out_prediction = os.path.join(prediction_args.out, os.path.basename(best_model), os.path.basename(outpath_test).replace(ext, "_prediction" + ext))
        if not os.path.exists(out_prediction):
            classification_predict.main(prediction_args)

        #################################
        #                               #
        #           Testing             #
        #                               #
        #################################

        evaluation_args = get_argparse_dict(classification_eval_VAXI.get_argparse())
        predict_csv_path = outdir_prediction + os.path.basename(outpath_test).replace(ext, "_prediction" + ext)

        evaluation_args['csv']= predict_csv_path
        evaluation_args['csv_true_column']= args.class_column
        evaluation_args['csv_prediction_column']= "Prediction"
        evaluation_args['title']= f"Confusion matrix fold {i}"
        evaluation_args['out']=  f"fold_{i}_eval.png"

        evaluation_args= Namespace(**evaluation_args)

        metric = classification_eval_VAXI.main(evaluation_args) # AUC or F1

        if best_metric < metric:
            best_metric = metric
            best_model_fold = best_model
            printGreen(f"Best model fold : {best_model_fold}")

    # Save the best model
    best_model_dir = args.out+ "/best_model"
    # copy file best model to best_model_dir
    if not os.path.exists(best_model_dir):
        os.makedirs(best_model_dir)
    os.system(f"cp {best_model_fold} {best_model_dir}")


    end_time = time.time()
    # format time
    hours, rem = divmod(end_time - start_time, 3600)
    minutes, seconds = divmod(rem, 60)
    printGreen("Training took {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Cleft classification Training')
    parser.add_argument('--csv', required=True, type=str, help='CSV with all the path and labels')
    parser.add_argument('--csv_special', type=str, default=None, help='CSV with all the path and labels for a specific dataset to add to the training [OPTIONAL]')

    parser.add_argument('--img_column', type=str, default='img', help='Name of image column')
    parser.add_argument('--class_column', type=str, default='Label', help='Name of class column')
    parser.add_argument('--seg_column', type=str, default=None, help='Name of segmentation column')
    parser.add_argument('--base_encoder', nargs="+", default='efficientnet-b0', help='Type of base encoder')
    parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float, help='Learning rate')
    parser.add_argument('--epochs', help='Max number of epochs', type=int, default=400)
    parser.add_argument('--log_every_n_steps', help='Log every n steps', type=int, default=10)
    parser.add_argument('--out', help='Output', type=str, default="./")
    parser.add_argument('--mount_point', help='Dataset mount directory', type=str, default="./")
    parser.add_argument('--num_workers', help='Number of workers for loading', type=int, default=4)
    parser.add_argument('--batch_size', help='Batch size', type=int, default=4)
    parser.add_argument('--patience', help='Patience for early stopping', type=int, default=50)
    parser.add_argument('--img_size', help='Image size of the dataset', type=int, default=224)

    # Arguments to avoid training from scratch [OPTIONAL]
    parser.add_argument('--model', help='Model path to continue training of this model with new data', type=str, default=None) #not implemented yet
    # don't work for real yet (cheats, uses the checkpoint to start training in each fold so the training set of the previous fold has been seen (including the validation set of the new fold)):
    parser.add_argument('--checkpoint', help='Path/URL to the checkpoint from which training is resumed', type=str, default=None)

    # tensorboard
    parser.add_argument('--tb_dir', help='Tensorboard output dir', type=str, default=None)
    parser.add_argument('--tb_name', help='Tensorboard experiment name', type=str, default="classification")

    # neptune
    parser.add_argument('--neptune_project', help='Neptune project name', type=str, default=None)
    parser.add_argument('--neptune_tag', help='Neptune tag', type=str, default="Left Canine Classification")

    # seed
    parser.add_argument('--seed', help='Seed', type=int, default=42)

    #Cross validation
    cv_group = parser.add_argument_group('Cross validation')
    cv_group.add_argument('--split', type=int, default=5, help='Number of splits for cross validation')
    cv_group.add_argument('--test_size', type=float, default=0.15, help='Test size')
    cv_group.add_argument('--val_size', type=float, default=0.15, help='Validation size')

    args = parser.parse_args()

    if not os.path.exists(args.out):
        os.makedirs(args.out)

    main(args)
