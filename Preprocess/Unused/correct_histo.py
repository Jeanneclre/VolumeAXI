import os
import argparse
import SimpleITK as sitk
import numpy as np
'''
Author: Jeanne Claret
Date: 2024-April-12
Description: Script to apply adaptive histogram equalization to a folder of nii.gz files.

Background: correct_histo.py wasn't working as expected. The histogram was not being corrected as expected.

'''

def equalize_histogram(input_folder, output_folder):
    cpt=0
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.endswith(".nii.gz"):
                input_path = os.path.join(root, file)
                output_path = os.path.join(output_folder, file)

                # Load the nii.gz file
                img = sitk.ReadImage(input_path)

                # #clip intensity range
                # p2,p98 = np.percentile(sitk.GetArrayFromImage(img), (2,98))
                # img = sitk.IntensityWindowing(img, p2, p98)

                # Apply adaptive histogram equalization
                equalized_img = sitk.AdaptiveHistogramEqualization(img)

                # Save the equalized histogram scan
                sitk.WriteImage(equalized_img, output_path)
                print(f"Saved {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adaptive Histogram Equalization")
    parser.add_argument("--input", help="Path to the input folder")
    parser.add_argument("--output", help="Path to the output folder")
    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    equalize_histogram(args.input, args.output)