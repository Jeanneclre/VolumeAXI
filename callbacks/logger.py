from pytorch_lightning.callbacks import Callback
import torchvision
import torch

import neptune as neptune
import os
import pandas as pd
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import random as rd
from pytorch_lightning.loggers import NeptuneLogger, TensorBoardLogger

def tensorboard_neptune_logger(args):
    image_logger=None
    logger = None
    try:
        tensorboard = args.tensorboard
        tensorboard = True
    except AttributeError:
        tensorboard = False

    try:
        neptune = args.neptune_project
        neptune =True
    except AttributeError:
        neptune = False

    if tensorboard:
        print('[DEBUG] USING TENSORBOARD LOGGER')
        logger = TensorBoardLogger(args.log_dir, name=args.name)
        if args.seg_column is None:
            image_logger = ImageLogger()
        else:
            image_logger = SegImageLogger()

    if neptune:
        print('[DEBUG] USING NEPTUNE LOGGER')
        logger = NeptuneLogger(project=args.neptune_project, tags=args.neptune_tag, api_key=os.environ["NEPTUNE_API_TOKEN"])
        if args.seg_column is None:
            image_logger = ImageLoggerNeptune(log_steps=args.log_every_n_steps, mode= args.mode)
        else:
            print('Neptune logger not implemented for segmentation')

    if logger is None:
        raise ValueError("No logger specified. Please specify either tensorboard or neptune")
    if image_logger is None:
        raise ValueError("No image logger specified. Please specify either ImageLogger or ImageLoggerNeptune")

    print('[DEBUG] logger:', logger)
    print('[DEBUG] image_logger:', image_logger)
    return logger, image_logger

class ImageLoggerNeptune(Callback):
    def __init__(self, num_images=4, log_steps=10,mode='CV'):
        self.log_steps = log_steps
        self.num_images = num_images
        self.idx =0
        self.modelType = mode

    def on_train_batch_end(self, trainer, pl_module, output, batch, batch_idx):

        if batch_idx % self.log_steps == 0:
            # print(f"[DEBUG] on_train_batch_end: batch_idx={batch_idx}")
            try:
                if self.modelType == 'CV_2fclayer':
                    x, y, y2= batch
                else:
                    x,y = batch
                # print('[DEBUG] x type:', type(x), 'x shape:', x.shape)
                # num_images = min(self.num_images, x.shape[0])
                with torch.no_grad():
                    # Normalize the image tensor
                    x = torch.clip(x, 0, 1)

                    # Create a slice of the 3D volume to get a 2D image
                    # tensor shape : [BS, C, D, H, W]
                    slices_to_watch = [(int(x.shape[2])/2) -20, int(x.shape[2])/2, (int(x.shape[2])/2) + 20]
                    #slices_idx = torch.randint(low=0, high=x.shape[0], size=(num_images,))
                    # slices = x[0,:,slices_idx[0],:,:]
                    # slices2 = x[0,:,:,slices_idx[0],:]
                    # slices3 = x[0,:,:,:,slices_idx[0]]
                    rand_idx = rd.randint(0, len(slices_to_watch)-1)
                    slices_idx = int(slices_to_watch[rand_idx])

                    slices = x[0,:,slices_idx,:,:]  #Axial
                    slices2 = x[0,:,:,slices_idx,:] #Coronal
                    slices3 = x[0,:,:,:,slices_idx] #Sagittal

                    permuted_slices = torch.permute(slices, (1,2,0))
                    permuted_slices2 = torch.permute(slices2, (1,2,0))
                    permuted_slices3 = torch.permute(slices3, (1,2,0))

                    # Create a grid of images
                    grid_x = torchvision.utils.make_grid(permuted_slices)
                    grid_x2 = torchvision.utils.make_grid(permuted_slices2)
                    grid_x3 = torchvision.utils.make_grid(permuted_slices3)

                    fig = plt.figure(figsize=(10, 10))
                    plt.subplot(1, 3, 1)
                    plt.imshow(grid_x.cpu().numpy(),cmap='gray')
                    plt.axis('off')
                    plt.title(f'Axial')

                    plt.subplot(1, 3, 2)
                    plt.imshow(grid_x2.cpu().numpy(),cmap='gray')
                    plt.axis('off')
                    plt.title(f'Coronal')

                    plt.subplot(1, 3, 3)
                    plt.imshow(grid_x3.cpu().numpy(),cmap='gray')
                    plt.axis('off')
                    plt.title(f'Sagittal')

                    trainer.logger.experiment[f'x/train_batch_{batch_idx}_slice{slices_idx}'].upload(fig)
                    plt.close(fig)
                    self.idx+=1
            except Exception as e:
                print(f"Error in on_train_batch_end: {e}")


def read_data(file_path):
    if os.path.splitext(file_path)[1] == ".csv":
        return pd.read_csv(file_path)
    elif os.path.splitext(file_path)[1] == ".parquet":
        return pd.read_parquet(file_path)
    else:
        raise ValueError("Unsupported file format: must be .csv or .parquet")



class ImageLogger(Callback):
    def __init__(self, num_images=256, log_steps=100):
        self.log_steps = log_steps
        self.num_images = num_images
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, unused=0):

        if batch_idx % self.log_steps == 0:

            x, y,y2 = batch

            x = torch.permute(x[0], (1, 0, 2, 3)) # get the first image in the batch and permute the channels dimension with i dimension, leave j, k dimensions.
            x = x - torch.min(x)
            x = x/torch.max(x)

            grid_x = torchvision.utils.make_grid(x[0:self.num_images])
            trainer.logger.experiment.add_image('x', grid_x, batch_idx)

class SegImageLogger(Callback):
    def __init__(self, num_images=256, log_steps=100):
        self.log_steps = log_steps
        self.num_images = num_images
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, unused=0):

        if batch_idx % self.log_steps == 0:
            x0, x1, y = batch

            x = torch.permute(x0[0], (1, 0, 2, 3)) # get the first image in the batch and permute the channels dimension with i dimension, leave j, k dimensions.
            x = x - torch.min(x)
            x = x/torch.max(x)

            grid_x = torchvision.utils.make_grid(x[0:self.num_images])
            trainer.logger.experiment.add_image('x', grid_x, batch_idx)



            x = torch.permute(x1[0], (1, 0, 2, 3)) # get the first image in the batch and permute the channels dimension with i dimension, leave j, k dimensions.
            x = x - torch.min(x)
            x = x/torch.max(x)

            grid_x = torchvision.utils.make_grid(x[0:self.num_images])
            trainer.logger.experiment.add_image('seg', grid_x, batch_idx)