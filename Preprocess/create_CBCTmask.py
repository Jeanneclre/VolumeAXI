import SimpleITK as sitk
import numpy as np
import argparse
import os

'''
Create a CBCT scan from a Mask (segmentation) and its corresponding CBCT (full scan).
The mask is applied to the CBCT to create a new CBCT with only the region of interest.

'''
def get_background_value(image):
    """Compute the background value of the CBCT scan."""
    return np.min(sitk.GetArrayFromImage(image))

def applyMask(image, mask, label,dilation_radius):
    """
    Apply a mask to an image.
    Use the background value of the CBCT scan as the value where the mask is not applied.

    """

    mask_array = sitk.GetArrayFromImage(mask)

    if label is not None and label in np.unique(mask_array):
        mask_array = np.where(mask_array == label, 1, 0)
        # pad the segmentation if not none to get a wider region of interest
        if dilation_radius is not None:
            # Create a binary image from mask_array
            binary_mask = sitk.GetImageFromArray(mask_array.astype(np.uint8))
            binary_mask.CopyInformation(mask)

            # Define the structuring element for dilation
            kernel = sitk.sitkBall
            radius = dilation_radius
            dilate_filter = sitk.BinaryDilateImageFilter()
            dilate_filter.SetKernelType(kernel)
            dilate_filter.SetKernelRadius(radius)

            # Perform the dilation
            dilated_mask = dilate_filter.Execute(binary_mask)

            # Update mask_array with dilated mask
            mask_array = sitk.GetArrayFromImage(dilated_mask)

        masked_image = np.where(mask_array == 1, sitk.GetArrayFromImage(image), get_background_value(image))

        masked_image = sitk.GetImageFromArray(masked_image)
        masked_image.CopyInformation(image)

    return masked_image

if __name__== "__main__":
    parser = argparse.ArgumentParser(description="Apply a mask to an image")
    parser.add_argument("--img", help="Input image")
    parser.add_argument("--mask", help="Input mask")
    parser.add_argument("--label", type=int, help="Label to apply the mask",default=1)
    parser.add_argument("--output", help="Output image")
    parser.add_argument("--dilatation_radius", type=int, help="Radius of the dilatation to apply to the mask",default=None)
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
            if img.split('_')[0] == mask.split('_')[0]:

                for i in range(len(word_to_match)):
                    if word_to_match[i] in img and word_to_match[i] in mask:
                        match_imgMask.append((os.path.join(args.img, img), os.path.join(args.mask, mask)))
                        break

                if (word_to_match[0] not in img and word_to_match[0] not in mask) and (word_to_match[1] not in img or word_to_match[1] not in mask):
                    match_imgMask.append((os.path.join(args.img, img), os.path.join(args.mask, mask)))



    for img_path, mask_path in match_imgMask:
        print('working on:', img_path, mask_path)
        image = sitk.ReadImage(img_path)
        read_mask = sitk.ReadImage(mask_path)
        output = applyMask(image, read_mask, args.label,args.dilatation_radius)

        filename = os.path.basename(img_path).replace('.nii.gz','_Masked.nii.gz')

        sitk.WriteImage(output, os.path.join(args.output,filename))
        cpt+=1
        if cpt%10 == 0:
            print(f"Mask applied to {cpt} images, still {len(match_imgMask)-cpt} to go...")


