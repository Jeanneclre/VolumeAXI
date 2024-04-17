# Script to use in case there are not enough data in the dataset
# Rotation

import itk
import SimpleITK as sitk
import math
import argparse
import os
import glob

def rotation_ITK(filename:str,out:str,args):
    '''
    Apply a rotation to the nifti scans
    '''
    angle = args.angle
    axis = args.axis

    img = sitk.ReadImage(filename)
    angle_rad = angle * math.pi / 180.0

    rotation = sitk.VersorTransform(axis, angle_rad)

     # Resample the image with the transform
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(img)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetTransform(rotation)
    resampled_img = resampler.Execute(img)


    # Write the output
    print("Writing:", out)
    sitk.WriteImage(resampled_img, out, useCompression=True)


def main_rotation(args):
    filenames= []
    out_dir = args.out
    normpath = os.path.normpath("/".join([args.dir, '**', '*']))
    for img in glob.iglob(normpath, recursive=True):
        if os.path.isfile(img) and True in [ext in img for ext in [".nrrd", ".nii", ".nii.gz", ".mhd", ".dcm", ".DCM", ".jpg", ".png"]]:
            fobj = {}
            fobj["img"] = img
            fobj["out"] = os.path.normpath(out_dir + "/" + img.replace(args.dir, ''))

            if not os.path.exists(os.path.dirname(fobj["out"])):
                os.makedirs(os.path.dirname(fobj["out"]))
            if not os.path.exists(fobj["out"]):
                rotation_ITK(fobj["img"],fobj['out'],args)



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Rotate an image', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    file_group = parser.add_argument_group('File parameters')
    file_group.add_argument('--dir', type=str, help='Input directory')
    file_group.add_argument('--out', type=str, help='Output directory')

    param_group = parser.add_argument_group('Rotation parameters')
    param_group.add_argument('--angle', type=float, help='Angle of rotation', default=90)
    param_group.add_argument('--axis', type=list, help='Axis of rotation ', default=(0,0,1))
    args = parser.parse_args()

    main_rotation(args)



