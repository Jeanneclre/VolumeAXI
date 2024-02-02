import SimpleITK as sitk

def get_nifti_info(file_path):
    # Read the NIfTI file
    image = sitk.ReadImage(file_path)

    # Get information
    info = {
        "Dimensions": image.GetDimension(),
        "Size": image.GetSize(),
        "Spacing": image.GetSpacing(),
        "Origin": image.GetOrigin(),
        "Direction": image.GetDirection(),
        "Pixel Type": sitk.GetPixelIDValueAsString(image.GetPixelID()),
        "Number of Components per Pixel": image.GetNumberOfComponentsPerPixel()
    }

    return info


# Call get nifti info for every nifti file in the folder and subfolders "Data_test_AREG"
import os
import json
import pandas as pd
import argparse

def get_nifti_info_folder(args):
    input_folder = args.input
    # Get all nifti files in the folder
    nifti_files = []
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.endswith(".nii") or file.endswith(".nii.gz"):
                nifti_files.append(os.path.join(root, file))



    # Get nifti info for every nifti file
    nifti_info = []
    for file in nifti_files:
        info = get_nifti_info(file)
        # print('info ', info)
        # nifti_info.append(info)
        # Create a csv file with the information of the nifti files
        basename = os.path.basename(file)
        df = pd.DataFrame([info])
        outpath = os.path.join(args.output, f"Nifti_info_{basename}.csv")
        df.to_csv(outpath, index=False)


    return nifti_info




if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Get nifti info')
    parser.add_argument('--input', type=str, default='Data_test_AREG', help='Input folder')
    parser.add_argument('--output', type=str, default='nifti_info.json', help='Output file')
    args = parser.parse_args()

    #if output folder does not exist, create it
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    get_nifti_info_folder(args)