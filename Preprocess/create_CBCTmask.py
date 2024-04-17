import SimpleITK as sitk
import numpy as np
import argparse
import os

'''
Create a CBCT scan from a Mask (segmentation) and its corresponding CBCT (full scan).
The mask is applied to the CBCT to create a new CBCT with only the region of interest.

'''

def applyMask(image, mask, label):
    """Apply a mask to an image."""
    # Cast the image to float32
    # image = sitk.Cast(image, sitk.sitkFloat32)
    array = sitk.GetArrayFromImage(mask)
    if label is not None and label in np.unique(array):
        array = np.where(array == label, 1, 0)

        mask = sitk.GetImageFromArray(array)
        #convert to uint8 because of package update
        mask= sitk.Cast(mask, sitk.sitkUInt8)

        mask.CopyInformation(image)

    # Ensure the mask is uint8
    if mask.GetPixelID() != sitk.sitkUInt8:
        mask = sitk.Cast(mask, sitk.sitkUInt8)

    return sitk.Mask(image, mask)

if __name__== "__main__":
    parser = argparse.ArgumentParser(description="Apply a mask to an image")
    parser.add_argument("--img", help="Input image")
    parser.add_argument("--mask", help="Input mask")
    parser.add_argument("--label", type=int, help="Label to apply the mask",default=1)
    parser.add_argument("--output", help="Output image")
    args = parser.parse_args()

    # if output folder does not exist, create it
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    word_to_match = ['MB','ML']
    cpt =0

    # Read the image from the input folder and the mask from from the mask folder
    # create a loop to match the mask with the image
    match_imgMask= []
    for img in os.listdir(args.img):
        for mask in os.listdir(args.mask):
            if img.split('_')[0] == mask.split('_')[0] and any(word not in img for word in word_to_match) and any(word not in mask for word in word_to_match):
                match_imgMask.append((os.path.join(args.img, img), os.path.join(args.mask, mask)))
            elif img.split('_')[0] == mask.split('_')[0] and word_to_match[0] in img and word_to_match[0] in mask:
                match_imgMask.append((os.path.join(args.img, img), os.path.join(args.mask, mask)))
            elif img.split('_')[0] == mask.split('_')[0] and word_to_match[1] in img and word_to_match[1] in mask:
                match_imgMask.append((os.path.join(args.img, img), os.path.join(args.mask, mask)))

    for img_path, mask_path in match_imgMask:
        print('working on:', img_path, mask_path)
        image = sitk.ReadImage(img_path)
        read_mask = sitk.ReadImage(mask_path)
        output = applyMask(image, read_mask, args.label)

        filename = os.path.basename(img_path).replace('.nii.gz','_Masked.nii.gz')

        sitk.WriteImage(output, os.path.join(args.output,filename))
        cpt+=1
        if cpt%10 == 0:
            print(f"Mask applied to {cpt} images, still {len(match_imgMask)-cpt} to go...")


