from pytorch_lightning.callbacks import Callback
import torchvision
import torch

import neptune.new as neptune
import os
import pandas as pd
import matplotlib.pyplot as plt

class ImageLoggerNeptune(Callback):
    def __init__(self, num_images=256, log_steps=100):
        self.log_steps = log_steps
        self.num_images = num_images
        self.idx =0

    def on_train_batch_end(self, trainer, pl_module, output,batch, batch_idx,unused=0):
        if batch_idx % self.log_steps == 0:
            x, y = batch

            num_images = min(self.num_images, x.shape[0])
            x=x.to(pl_module.device, non_blocking=True)
            y=y.to(pl_module.device, non_blocking=True)

            with torch.no_grad():
                # Normalize the image tensor
                x = x - torch.min(x)
                x = x / torch.max(x)

                # Create a slice of the 3D volume to get a 2D image
                slices = x[0:num_images,:,:,:,0:int(x.shape[-1]/2)]
                slice2 = x[0:num_images,:,:,int(x.shape[-1]/2):,:]

                permuted_slices = torch.permute(slices[0], (1,2,0,3))

                # Create a grid of images
                grid_x = torchvision.utils.make_grid(permuted_slices[0:num_images])
                grid_x2 = torchvision.utils.make_grid(slice2[0,0:num_images,0:1,:,:])
                print('grid_x:',grid_x.shape)
                print('grid_x2:',grid_x2.shape)
                fig = plt.figure(figsize=(10, 10))
                ax= plt.subplot(1, 2, 1)

                ax = plt.imshow(grid_x.cpu().numpy())
                ax2 = plt.subplot(1, 2, 2)
                ax2 = plt.imshow(grid_x2.permute(1, 2, 0).cpu().numpy())
                trainer.logger.experiment[f'x/train_batch_{batch_idx}_{self.idx}'].upload(fig)
                plt.close(fig)
                self.idx+=1


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
            x, y = batch

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