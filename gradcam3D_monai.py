import os
import SimpleITK as sitk
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import monai
from monai.visualize import GradCAM
import matplotlib.pyplot as plt

from nets.classification import Net
from transforms.volumetric import EvalTransforms

import cv2
import argparse
import pandas as pd
from tqdm import tqdm

from loaders.cleft_dataset import BasicDataset

"""
Script to visualize Grad-CAM for 3D images using MONAI library.
Usable on csv file or image path.

The output is a grayscale image of the Grad-CAM.
You need to do the overlay in 3D slicer with the original image.
You can use either the module 'Colors" or 'Volumes' to choose a Colormap for the Grad-CAM.
"""
# Custom Dataset class
class CustomDataset(Dataset):
    def __init__(self, img_path, transforms):
        self.img_path = img_path
        self.transforms = transforms

    def __len__(self):
        return len(self.img_path)  # Adjust based on your data

    def __getitem__(self, idx):
        img = sitk.ReadImage(self.img_path[idx], sitk.sitkFloat32)
        img_array = sitk.GetArrayFromImage(img)
        img_tensor = self.transforms(img_array)
        label = 0  #default, can't be used in this case
        return img_tensor, label

# Grad-CAM function
def grad_cam(batch, model, target_layer,class_index=1):
    grad_cam = monai.visualize.GradCAM(nn_module=model, target_layers=target_layer)
    cam = grad_cam(batch, class_idx=class_index)  # Example for class index 1
    return cam

# Function to blend two SimpleITK images with a specified alpha
def blend_images(image1, image2, alpha=0.6):
    array1 = sitk.GetArrayFromImage(image1)
    print('tensor shape original:', array1.shape)
    #normalize the image if not already done
    array1 = (array1 - np.min(array1)) / (np.max(array1) - np.min(array1))

    array2 = sitk.GetArrayFromImage(image2)

    if np.max(array2) > 1.0 or np.min(array2) < 0.0:
        array2 = (array2 - np.min(array2)) / (np.max(array2) - np.min(array2))

    blended_array = (array1 * alpha + array2 * (1 - alpha))

    #where image1 array are "0", stay "0"
    blended_array = np.where(array1 == 0, 0, blended_array)

    #copy image1 information for recreating the image
    blended_img= sitk.GetImageFromArray(blended_array)
    blended_img.SetSpacing(image1.GetSpacing())
    blended_img.SetOrigin(image1.GetOrigin())

    return blended_img

def graytorgb(image):
    '''
    Convert each 2D slice of a 3D grayscale image to RGB
    using OpenCV, then reconstruct the 3D RGB image with SimpleITK.
    '''
    # Extract the numpy array from the sitk image
    image_array = sitk.GetArrayFromImage(image)

    if len(image_array.shape) >3:
        image_array= np.squeeze(image_array)
        if len(image_array.shape) > 3:

            image_array = image_array[0,:,:,:]
            image = sitk.GetImageFromArray(image_array)
    # Convert the image to 8-bit if necessary

    if image_array.dtype != np.uint8:
        # Normalize the pixel values to 0-255 and convert to uint8
        image_array = (255 * (image_array - image_array.min()) / (image_array.max() - image_array.min())).astype(np.uint8)
    # Prepare an empty array for the RGB data
    rgb_array = np.zeros((image_array.shape[0], image_array.shape[1], image_array.shape[2], 3), dtype=np.uint8)

    # Convert each slice to RGB
    for i in range(image_array.shape[0]):
        colored_slice = cv2.applyColorMap(image_array[i], cv2.COLORMAP_JET)
        # colored_slice = cv2.cvtColor(colored_slice, cv2.COLOR_BGR2RGB)
        rgb_array[i] = colored_slice


    # Create a new SimpleITK image from the RGB numpy array
    rgb_image = sitk.GetImageFromArray(rgb_array)

    # Copy the spatial and calibration information from the original image
    rgb_image.CopyInformation(image)

    return rgb_image


def plot_gradcam(image1, image2,image3, slice_index,cam_save_dir, class_index, patient_name,args):

    plt.figure(figsize=(18, 18))

    plt.subplot(1, 3, 1)
    plt.imshow(image1[slice_index], cmap='gray')
    plt.title(f'Original Slice {slice_index}')
    plt.axis('off')

    plt.subplot(1,3,2)
    plt.imshow(image2[slice_index], cmap='jet')
    plt.title(f'GradCam Last Layer')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(image3[slice_index], cmap='jet')

    plt.title(f'GradCam sum rgb')
    plt.axis('off')

    if args.show_plots:
        plt.show()


    # Save individual figures
    plt.savefig(f'{cam_save_dir}/{patient_name}_classIdx_{class_index}_slice_{slice_index}.png')
    plt.close()

def get_cam_sum(data,model,class_index, layers):
    cam_sum =0
    for layer in layers:
        cam = grad_cam(data, model, layer, class_index)

        cam_np = cam.squeeze().detach().cpu().numpy()

        cam_sum += cam_np
        # Normalize the CAM for better visualization

        # Normalize cam_sum
        cam_sum_norm = (cam_sum - np.min(cam_sum)) / (np.max(cam_sum) - np.min(cam_sum))

    # Normalize last layer to plot it
    cam_min, cam_max = cam_np.min(), cam_np.max()
    cam_normalized = (cam_np - cam_min) / (cam_max - cam_min)

    return cam_normalized,cam_sum_norm


def main(args):
    # Parameters
    nb_of_classes = args.nb_classes
    class_index = args.class_index
    layer_name = args.layer_name
    out_dir = args.out

    # Data loading and transformations
    if args.csv_test is not None:
        df = pd.read_csv(args.csv_test)
        unique_classes = np.unique(df[args.class_column])
        class_replace = {}
        for cn, cl in enumerate(unique_classes):
            class_replace[int(cl)] = cn
        df[args.class_column] = df[args.class_column].replace(class_replace)
        # img_path = ['Preprocess/Preprocessed_data/Resampled/Left/MN080_scan_MB_Masked.nii.gz']
        # dataset = CustomDataset(img_path, transforms=EvalTransforms(256))
        transforms =EvalTransforms(args.img_size)
        dataset = BasicDataset(df,mount_point=args.mount_point,img_column= args.img_column,class_column=args.class_column,transform= transforms)
        data_loader = DataLoader(dataset, batch_size=1, num_workers=1)

    if args.img_path is not None:
        img_path = [args.img_path]
        dataset = CustomDataset(img_path, transforms=EvalTransforms(args.img_size))
        data_loader = DataLoader(dataset, batch_size=1, num_workers=1)
        # create pbar so we have 3 values to unpack when data_loader has only 1
        pbar = tqdm(enumerate(data_loader), total=1)
        print('values to unpack pbar:', len(data_loader))
        print('pbars:', pbar)

    else:
        pbar = tqdm(enumerate(data_loader), total=len(data_loader))
    # Model preparation
    model_path = args.model_path
    checkpoint = torch.load(model_path)

    if 'loss.weight' in checkpoint['state_dict']:
        del checkpoint['state_dict']['loss.weight']

    model = Net(
        args = checkpoint.get('args'),
        class_weights=checkpoint.get('class_weights'),
        base_encoder=args.base_encoder,
        num_classes=nb_of_classes
    ).cuda()
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    # if data_loader has not 2 values to unpack, add one
    if args.img_path is not None:
        given_path = True
    else:
        given_path = False

    # Save directories
    cam_save_dir = f'{out_dir}/cam_images'
    if os.path.exists(cam_save_dir) is False:
        os.makedirs(cam_save_dir)


    for batch,(X,y) in pbar:
        print('batch:', batch)
        print('====')
        data= X.cuda()

        cam_normalized, cam_sum_norm = get_cam_sum(data,model,class_index,layer_name)
        original_img_np= data.squeeze().detach().cpu().numpy()


        if given_path:
            patient_name_list = os.path.basename(img_path[0]).split('_')

        else:
            patient_name_list = os.path.basename(df[batch]['Path']).split('_')

        if "MB" in patient_name_list :
            patient_name = patient_name_list[0] + '_' + "MB"
        elif "ML" in patient_name_list:
            patient_name = patient_name_list[0] + '_' + "ML"
        elif "MR" in patient_name_list:
            patient_name = patient_name_list[0] + '_' + "MR"
        else:
            patient_name = patient_name_list[0]

        plot_gradcam(original_img_np, cam_normalized, cam_sum_norm, args.slice_idx, cam_save_dir, class_index, patient_name,args)

        # Save the CAM as a Nifti image in grayscale levels
        if args.csv_test is not None:
            true_class = df.loc[batch]['Label']
            predicted_class = df.loc[batch]['Predictions']

        else:
            true_class = 'X'
            predicted_class = 'X'
        if args.csv_test is not None:
            img_fn = df.loc[batch][args.img_column]
        else:
            img_fn = args.img_path
        patient_name = os.path.basename(img_fn).split('_')[0]

        #Save GradCam
        img = sitk.ReadImage(os.path.join(args.mount_point, img_fn))
        saved_cam2 = sitk.GetImageFromArray(cam_sum_norm)
        saved_cam2.SetSpacing(img.GetSpacing())
        saved_cam2.SetOrigin(img.GetOrigin())
        saved_cam2.SetDirection(img.GetDirection())

        sitk.WriteImage(saved_cam2, f'{cam_save_dir}/{patient_name}_classIdx_{class_index}_trueClass_{true_class}_predClass_{predicted_class}.nii.gz')


if __name__== "__main__":
    parser = argparse.ArgumentParser(description='Classification Visualization Volumes')
    # Creating a mutually exclusive group
    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument('--csv_test', type=str, help='Testing set csv to load')
    parser.add_argument('--img_column', type=str, default='Path', help='Name of image column')
    parser.add_argument('--class_column', type=str, default='Label', help='Name of class column with the labels')

    group.add_argument('--img_path', type=str, help='Path to the image to load')

    parser.add_argument('--mount_point', help='Dataset mount directory', type=str, default="./")
    parser.add_argument('--model_path', help='Model path to use', type=str, default='Training_Left/SEResNet50/Models/epoch_200-val_loss_0.607.ckpt')
    parser.add_argument('--out', help='Output folder with vizualisation files', type=str, default="Training_Left/SEResNet50/Predictions/GRADCAM/onebyone")

    parser.add_argument('--img_size', help='Image size of the dataset', type=int, default=256)
    parser.add_argument('--nb_classes', help='Number of classes', type=int, default=3)
    parser.add_argument('--class_index', help='Class index for GradCAM', type=int, default=1)
    parser.add_argument('--layer_name', help='Layer name for GradCAM', nargs="+", default=['model.layer2','model.layer3','model.layer4'])

    parser.add_argument('--base_encoder', type=str, default="SEResNet50", help='Type of base encoder')
    parser.add_argument('--show_plots', help='Show plots', type=bool, default=False)
    parser.add_argument('--slice_idx', help='Slice index to plot', type=int, default=120)
    args = parser.parse_args()

    main(args)