import SimpleITK as sitk
import numpy as np
import argparse
import os
import glob
import sys
import csv

def resample_image_with_custom_size(img,segmentation, args):
    '''
    Use segmentation to crop the image and then pad it to the target size
    '''
    target_size = args.size

    print("===== IMG INFO =====")
    print("Size:", img.GetSize())
    print("Seg Size:", segmentation.GetSize())
    print("Spacing:", img.GetSpacing())
    print("Origin:", img.GetOrigin())

    label_shape_filter = sitk.LabelShapeStatisticsImageFilter()
    label_shape_filter.Execute(segmentation)
    bounding_box = label_shape_filter.GetBoundingBox(1)  # Assuming label of interest is 1

    print('bounding box:', bounding_box)

    roi = sitk.RegionOfInterestImageFilter()
    roi.SetRegionOfInterest(bounding_box)
    roi.SetSize([bounding_box[i + 3] for i in range(img.GetDimension())])
    roi_img = roi.Execute(img)

    print("roi_img size:", roi_img.GetSize())
    # Pad the ROI to the target size
     # Check if the target size is larger than the image size, pad if so
    axes_to_pad_Up = [0]*img.GetDimension()
    axes_to_pad_Down = [0]*img.GetDimension()
    for dim in range(img.GetDimension()):
        if args.crop == False:
            if roi_img.GetSize()[dim] < target_size[dim]:
                pad_size = target_size[dim] - roi_img.GetSize()[dim]
                pad_size_Up = pad_size // 2
                pad_size_Down = pad_size - pad_size_Up
                axes_to_pad_Up[dim] = pad_size_Up
                axes_to_pad_Down[dim] = pad_size_Down
                pad_filter = sitk.ConstantPadImageFilter()
                # Get the minimum value of the image
                min_val = float(np.min(sitk.GetArrayFromImage(img)))
                pad_filter.SetConstant(min_val)
                pad_filter.SetPadLowerBound(axes_to_pad_Down)
                pad_filter.SetPadUpperBound(axes_to_pad_Up)
                img_padded = pad_filter.Execute(roi_img)

            # if roi_img.GetSize()[dim] > target_size[dim]:
            #     # Crop the image
            #     crop_size_up = (roi_img.GetSize()[dim] - target_size[dim]) // 2
            #     crop_size_down = roi_img.GetSize()[dim] - target_size[dim] - crop_size_up
            #     axes_to_pad_Up[dim] = crop_size_up
            #     axes_to_pad_Down[dim] = crop_size_down
            #     crop_filter = sitk.CropImageFilter()
            #     crop_filter.SetLowerBoundaryCropSize(axes_to_pad_Up)
            #     crop_filter.SetUpperBoundaryCropSize(axes_to_pad_Down)
            #     img_padded = crop_filter.Execute(roi_img)

            # #check if the image is the same size as the target size
            # if img_padded.GetSize()[dim] == target_size[dim]:
            #     img_padded = img_padded

        else:
            img_padded = roi_img

    print("New size:", img_padded.GetSize())
    return img_padded


def resample_fn(img, args):
    '''
    Resample an image to a new size and spacing
    '''
    output_size = args.size
    fit_spacing = args.fit_spacing
    iso_spacing = args.iso_spacing
    pixel_dimension = args.pixel_dimension
    center = args.center

    # if(pixel_dimension == 1):
    #     zeroPixel = 0
    # else:
    #     zeroPixel = np.zeros(pixel_dimension)

    if args.linear:
        InterpolatorType = sitk.sitkLinear
    else:
        InterpolatorType = sitk.sitkNearestNeighbor


    spacing = img.GetSpacing()
    size = img.GetSize()

    output_origin = img.GetOrigin()
    output_size = [si if o_si == -1 else o_si for si, o_si in zip(size, output_size)]
    # print(output_size)

    if(fit_spacing):
        print('Fit spacing')
        output_spacing = [sp*si/o_si for sp, si, o_si in zip(spacing, size, output_size)]
    else:
        output_spacing = spacing

    if(iso_spacing ):
        print('Iso spacing' )
        output_spacing_filtered = [sp for si, sp in zip(args.size, output_spacing) if si != -1]
        # print(output_spacing_filtered)
        max_spacing = np.max(output_spacing_filtered)
        output_spacing = [sp if si == -1 else max_spacing for si, sp in zip(args.size, output_spacing)]
        # print(output_spacing)

    if(args.spacing is not None):
        output_spacing = args.spacing

    if(args.origin is not None):
        output_origin = args.origin

    if(center):
        output_physical_size = np.array(output_size)*np.array(output_spacing)
        input_physical_size = np.array(size)*np.array(spacing)
        output_origin = np.array(output_origin) - (output_physical_size - input_physical_size)/2.0

    print("Input size:", size)
    print("Input spacing:", spacing)
    print("Output size:", output_size)

    print("Output spacing:", output_spacing)
    print("Output origin:", output_origin)

    min_val = float(np.min(sitk.GetArrayFromImage(img)))
    resampleImageFilter = sitk.ResampleImageFilter()
    resampleImageFilter.SetInterpolator(InterpolatorType)
    resampleImageFilter.SetOutputSpacing(output_spacing)
    resampleImageFilter.SetSize(output_size)
    resampleImageFilter.SetOutputDirection(img.GetDirection())
    resampleImageFilter.SetOutputOrigin(output_origin)
    resampleImageFilter.SetDefaultPixelValue(min_val)


    return resampleImageFilter.Execute(img)


def Resample(img_filename, segm, args):

    output_size = args.size
    fit_spacing = args.fit_spacing
    iso_spacing = args.iso_spacing
    img_dimension = args.image_dimension
    pixel_dimension = args.pixel_dimension

    print("Reading:", img_filename)
    img = sitk.ReadImage(img_filename)

    if(args.img_spacing):
        img.SetSpacing(args.img_spacing)

    if args.segmentation is not None:
        seg =   sitk.ReadImage(segm)
        return resample_image_with_custom_size(img, seg, args)
    else:

        return resample_fn(img, args)






if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Resample an image', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    in_group = parser.add_mutually_exclusive_group(required=True)

    in_group.add_argument('--img', type=str, help='image to resample')
    in_group.add_argument('--dir', type=str, help='Directory with image to resample')
    in_group.add_argument('--csv', type=str, help='CSV file with column img with paths to images to resample')

    csv_group = parser.add_argument_group('CSV extra parameters')
    csv_group.add_argument('--csv_column', type=str, default='image', help='CSV column name (Only used if flag csv is used)')
    csv_group.add_argument('--csv_root_path', type=str, default=None, help='Replaces a root path directory to empty, this is use to recreate a directory structure in the output directory, otherwise, the output name will be the name in the csv (only if csv flag is used)')
    csv_group.add_argument('--csv_use_spc', type=int, default=0, help='Use the spacing information in the csv instead of the image')
    csv_group.add_argument('--csv_column_spcx', type=str, default=None, help='Column name in csv')
    csv_group.add_argument('--csv_column_spcy', type=str, default=None, help='Column name in csv')
    csv_group.add_argument('--csv_column_spcz', type=str, default=None, help='Column name in csv')

    transform_group = parser.add_argument_group('Transform parameters')
    transform_group.add_argument('--ref', type=str, help='Reference image. Use an image as reference for the resampling', default=None)
    transform_group.add_argument('--size', nargs="+", type=int, help='Output size, -1 to leave unchanged', default=None)
    transform_group.add_argument('--img_spacing', nargs="+", type=float, default=None, help='Use this spacing information instead of the one in the image')
    transform_group.add_argument('--spacing', nargs="+", type=float, default=None, help='Output spacing')
    transform_group.add_argument('--origin', nargs="+", type=float, default=None, help='Output origin')
    transform_group.add_argument('--linear', type=bool, help='Use linear interpolation.', default=True)
    transform_group.add_argument('--center', type=bool, help='Center the image in the space', default=True)
    transform_group.add_argument('--fit_spacing', type=bool, help='Fit spacing to output', default=False)
    transform_group.add_argument('--iso_spacing', type=bool, help='Same spacing for resampled output', default=True)
    transform_group.add_argument('--crop', type=bool, help='Only Crop the image to the segmentation (used only if args.segmentation is)', default=False)

    img_group = parser.add_argument_group('Image parameters')
    img_group.add_argument('--image_dimension', type=int, help='Image dimension', default=3)
    img_group.add_argument('--pixel_dimension', type=int, help='Pixel dimension', default=1)
    img_group.add_argument('--rgb', type=bool, help='Use RGB type pixel', default=False)
    img_group.add_argument('--segmentation', type=str, help='Segmentation image to resample', default=None)

    out_group = parser.add_argument_group('Ouput parameters')
    out_group.add_argument('--ow', type=int, help='Overwrite', default=1)
    out_group.add_argument('--out', type=str, help='Output image/directory', default="./out.nrrd")
    out_group.add_argument('--out_ext', type=str, help='Output extension type', default=None)

    args = parser.parse_args()

    filenames = []
    filenames_seg = []
    if(args.img):
        fobj = {}
        fobj["img"] = args.img
        fobj["out"] = args.out
        if args.segmentation is not None:
            fobj["seg"] = args.segmentation
        filenames.append(fobj)
    elif(args.dir):
        out_dir = args.out
        normpath = os.path.normpath("/".join([args.dir, '**','*']))

        fobj = {}
        for img in glob.iglob(normpath, recursive=True):
            if os.path.isfile(img) and True in [ext in img for ext in [".nrrd", ".nii", ".nii.gz", ".mhd", ".dcm", ".DCM", ".jpg", ".png"]]:
                if args.segmentation is not None:
                    patient=os.path.basename(img).split('_')[0]
                    basename=os.path.basename(img)
                    for word in ['MB','ML','MR']:
                        if word in basename:
                            patient= patient +f'_{word}'
                            break

                    if patient not in fobj:
                        fobj[patient] = {}
                    fobj[patient]["img"] = img
                    fobj[patient]["out"] = os.path.normpath(out_dir + "/" + img.replace(args.dir, ''))
                    if args.out_ext is not None:
                        out_ext = args.out_ext
                        if out_ext[0] != ".":
                            out_ext = "." + out_ext
                        fobj[patient]["out"] = os.path.splitext(fobj[patient]["out"])[0] + out_ext

                    if not os.path.exists(os.path.dirname(fobj[patient]["out"])):
                        os.makedirs(os.path.dirname(fobj[patient]["out"]))
                    if not os.path.exists(fobj[patient]["out"]) or args.ow:
                        filenames.append(fobj)
                else:

                    fobj = {}
                    fobj["img"] = img
                    fobj["out"] = os.path.normpath(out_dir + "/" + img.replace(args.dir, ''))

                    if args.out_ext is not None:
                        out_ext = args.out_ext
                        if out_ext[0] != ".":
                            out_ext = "." + out_ext
                        fobj["out"] = os.path.splitext(fobj["out"])[0] + out_ext

                    if not os.path.exists(os.path.dirname(fobj["out"])):
                        os.makedirs(os.path.dirname(fobj["out"]))

                    if not os.path.exists(fobj["out"]) or args.ow:
                        filenames.append(fobj)

    elif(args.csv):
        replace_dir_name = args.csv_root_path
        with open(args.csv) as csvfile:
            csv_reader = csv.DictReader(csvfile)
            for row in csv_reader:
                fobj = {}
                fobj["img"] = row[args.csv_column]
                fobj["out"] = row[args.csv_column]
                if(replace_dir_name):
                    fobj["out"] = fobj["out"].replace(replace_dir_name, args.out)
                if args.csv_use_spc:
                    img_spacing = []
                    if args.csv_column_spcx:
                        img_spacing.append(row[args.csv_column_spcx])
                    if args.csv_column_spcy:
                        img_spacing.append(row[args.csv_column_spcy])
                    if args.csv_column_spcz:
                        img_spacing.append(row[args.csv_column_spcz])
                    fobj["img_spacing"] = img_spacing

                if "ref" in row:
                    fobj["ref"] = row["ref"]

                if args.out_ext is not None:
                    out_ext = args.out_ext
                    if out_ext[0] != ".":
                        out_ext = "." + out_ext
                    fobj["out"] = os.path.splitext(fobj["out"])[0] + out_ext
                if not os.path.exists(os.path.dirname(fobj["out"])):
                    os.makedirs(os.path.dirname(fobj["out"]))
                if not os.path.exists(fobj["out"]) or args.ow:
                    filenames.append(fobj)
    else:
        raise "Set img or dir to resample!"

    if args.segmentation is not None:
        normpath_seg= os.path.normpath("/".join([args.segmentation, '**', '*']))
        fobj_seg = {}
        for seg in glob.iglob(normpath_seg, recursive=True):
            if os.path.isfile(seg) and True in [ext in seg for ext in [".nrrd", ".nii", ".nii.gz", ".mhd", ".dcm", ".DCM", ".jpg", ".png"]]:

                patient_seg=os.path.basename(seg).split('_')[0]
                basename_seg=os.path.basename(seg)
                for word in ['MB','ML','MR']:
                    if word in basename_seg:
                        patient_seg= patient_seg +f'_{word}'
                # if the key patient doesn't exist, create it
                if patient_seg not in fobj_seg:
                    fobj_seg[patient_seg] = {}
                fobj_seg[patient_seg]["seg"] = seg
                filenames_seg.append(fobj_seg)

    if(args.rgb):
        if(args.pixel_dimension == 3):
            print("Using: RGB type pixel with unsigned char")
        elif(args.pixel_dimension == 4):
            print("Using: RGBA type pixel with unsigned char")
        else:
            print("WARNING: Pixel size not supported!")

    if args.ref is not None:
        print(args.ref)
        ref = sitk.ReadImage(args.ref)
        args.size = ref.GetSize()
        args.spacing = ref.GetSpacing()
        args.origin = ref.GetOrigin()

    if args.segmentation is None:
        for fobj in filenames:

            # try:

            if "ref" in fobj and fobj["ref"] is not None:
                ref = sitk.ReadImage(fobj["ref"])
                args.size = ref.GetSize()
                args.spacing = ref.GetSpacing()
                args.origin = ref.GetOrigin()

            if args.size is not None:
                img = Resample(fobj["img"], args.segmentation, args)
            else:

                img = sitk.ReadImage(fobj["img"])
                size = img.GetSize()
                physical_size = np.array(size)*np.array(img.GetSpacing())
                new_size = [int(physical_size[i]//args.spacing[i]) for i in range(img.GetDimension())]
                args.size = new_size
                img = Resample(fobj["img"],args.segmentation, args)



            print("Writing:", fobj["out"])
            writer = sitk.ImageFileWriter()
            writer.SetFileName(fobj["out"])
            writer.UseCompressionOn()
            writer.Execute(img)

            # except Exception as e:
            #     print(f"Error processing {fobj['img']}")
            #     print(e, file=sys.stderr)

    else:

        for key, img in fobj.items():
            cpt=0
            for key_seg in fobj_seg.keys():
                if key == key_seg:

                    img = Resample(fobj[key]['img'], fobj_seg[key_seg]['seg'], args)

                    print("Writing:",fobj[key]['img'])
                    writer = sitk.ImageFileWriter()
                    writer.SetFileName(fobj[key]['out'])
                    writer.UseCompressionOn()
                    writer.Execute(img)

