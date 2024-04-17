# import SimpleITK as sitk
# import itk
import argparse
import os
import glob
import time
import nibabel as nib
import numpy as np

def binarize_nifti(args):
    '''
    function to binarize a nifti image from Hounsfield Unit value to 0 or 1
    '''
    input_path = args.input
    output_path = args.out_dir
    HU_threshold = args.HUthreshold
    # get the image in the folder and all the subfolders

    # find every file in the directory and subdirectory
    for file in glob.iglob(input_path + '**/**', recursive=True):
        if file.endswith(".nii.gz"):
            # print('file name ', file)
            time.sleep(1)
            output_filename = file.replace(args.mount_point, output_path)


            # Remplacez ceci par le chemin de votre fichier NIfTI
            file_path = file
            # Charger l'image NIfTI
            nifti_image = nib.load(file_path)
            image_data = nifti_image.get_fdata()

            # Définir un seuil
            threshold_value = HU_threshold

            # Appliquer le seuil
            thresholded_data = np.where(image_data > threshold_value, image_data, 0)

            # Créer un nouvel objet NIfTI pour l'image seuillée
            thresholded_image = nib.Nifti1Image(thresholded_data, nifti_image.affine, nifti_image.header)

            min = thresholded_data.min()
            max = thresholded_data.max()


            out_dir = os.path.dirname(output_filename)
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)

            # Enregistrer l'image seuillée
            print('writing file ', output_filename)
            nib.save(thresholded_image, output_filename)
            # Load the NIfTI image
            # image = sitk.ReadImage(file)

            # # Binarize the image using the given threshold
            # binary_image = sitk.BinaryThreshold(image, lowerThreshold=threshold, upperThreshold=255, insideValue=1, outsideValue=0)
            # normalized = sitk.Normalize(image)
            ##############################
            # PixelType = itk.UC
            # Dimension = 3

            # ImageType = itk.Image[PixelType, Dimension]
            # print('ImageType ', ImageType)
            # reader = itk.ImageFileReader[ImageType].New()
            # reader.SetFileName(file)

            # image = reader.GetOutput()
            # # Calcul des statistiques
            # statisticsFilter = itk.StatisticsImageFilter[ImageType].New()
            # statisticsFilter.SetInput(image)
            # statisticsFilter.Update()

            # # Obtention du minimum et du maximum
            # min_intensity = statisticsFilter.GetMinimum()
            # max_intensity = statisticsFilter.GetMaximum()

            # print(f'Minimum Intensity: {min_intensity}')
            # print(f'Maximum Intensity: {max_intensity}')


            # # Accès à un voxel spécifique - remplacez i, j, k par les indices du voxel
            # i, j, k = 68, 60, 89  # Exemple de coordonnées de voxel
            # index = itk.Index[Dimension]()
            # index.SetElement(0, i)
            # index.SetElement(1, j)
            # index.SetElement(2, k)

            # # Obtention de la valeur d'intensité du voxel
            # voxel_value = image.GetPixel(index)

            # print(f"La valeur du voxel à l'index {index} est : {voxel_value}")

            # thresholdFilter = itk.BinaryThresholdImageFilter[ImageType, ImageType].New()
            # thresholdFilter.SetInput(reader.GetOutput())

            # thresholdFilter.SetLowerThreshold(Lower_threshold)
            # thresholdFilter.SetUpperThreshold(Upper_threshold)
            # thresholdFilter.SetOutsideValue(0)
            # thresholdFilter.SetInsideValue(1)

            # out_dir = os.path.dirname(output_filename)
            # if not os.path.exists(out_dir):
            #     os.makedirs(out_dir)

            # # Get the number of color channels (components)
            # num_channels = reader.GetOutput().GetNumberOfComponentsPerPixel()

            # print(f'Number of color channels: {num_channels}')


            # # Save the binarized image
            # print('writing file ', output_filename)

            # writer = itk.ImageFileWriter[ImageType].New()
            # writer.SetFileName(output_filename)
            # writer.SetInput(thresholdFilter.GetOutput())

            # writer.Update()


            # sitk.WriteImage(binary_image, output_filename)
            time.sleep(1)







if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Binarize dataset')
    parser.add_argument('--input', required=True, type=str, help='CSV')
    parser.add_argument('--out_dir', required=True, type=str, help='Output directory')
    parser.add_argument('--mount_point', required=False, type=str, help='Mount point')
    parser.add_argument('--HUthreshold', required=False, type=int, help='Threshold in Hounsfield Unit, value of the pixel in CT scans (in bold in slicer)',default=700)

    args = parser.parse_args()


    #create the output directory if non existent
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)


    binarize_nifti(args)


# import itk
# import argparse

# parser = argparse.ArgumentParser(description="Threshold An Image Using Binary.")
# parser.add_argument("input_image")
# parser.add_argument("output_image")
# parser.add_argument("lower_threshold", type=int)
# parser.add_argument("upper_threshold", type=int)
# parser.add_argument("outside_value", type=int)
# parser.add_argument("inside_value", type=int)
# args = parser.parse_args()
