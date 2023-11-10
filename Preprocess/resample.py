import SimpleITK as sitk
import numpy as np
import argparse
import os
import glob
import sys
import csv

from statistics import mean

import time
import sys

def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=50, fill='X'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    sys.stdout.write(f'\r{prefix} |{bar}| {percent}% {suffix}')
    sys.stdout.flush()  # Flushes the buffer
    if iteration == total: 
        print()  # Print New Line on Complete

def resample_fn(img, output_spacing,args):
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

    # if(fit_spacing):
    #     output_spacing = [sp*si/o_si for sp, si, o_si in zip(spacing, size, output_size)]
    # else:
    #     output_spacing = spacing

    # if(iso_spacing):
    #     print('in iso spacing ****************')
    #     output_spacing_filtered = [sp for si, sp in zip(args.size, output_spacing) if si != -1]
    #     # print(output_spacing_filtered)
    #     max_spacing = np.max(output_spacing_filtered)
    #     output_spacing = [sp if si == -1 else max_spacing for si, sp in zip(args.size, output_spacing)]
    #     # print(output_spacing)

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

    resampleImageFilter = sitk.ResampleImageFilter()
    resampleImageFilter.SetInterpolator(InterpolatorType)   
    resampleImageFilter.SetOutputSpacing(output_spacing)
    resampleImageFilter.SetSize(output_size)
    resampleImageFilter.SetOutputDirection(img.GetDirection())
    resampleImageFilter.SetOutputOrigin(output_origin)
    # resampleImageFilter.SetDefaultPixelValue(zeroPixel)
    

    return resampleImageFilter.Execute(img)

def AddSpacing(img_filename:str, output_spacing:list, output_spacing3:list, args):
    '''
    Add proportional spacing to a list to get all the spacing used in the images
    '''
    img = sitk.ReadImage(img_filename)

    fit_spacing = args.fit_spacing
    output_size = args.size

    if(args.img_spacing):
        img.SetSpacing(args.img_spacing)

    spacing = img.GetSpacing()
    size = img.GetSize()
    output_size = [si if o_si == -1 else o_si for si, o_si in zip(size, output_size)]
    # print(output_size)
    if 'sp1' in img_filename:
        if(fit_spacing):
            new_spacing = [sp*si/o_si for sp, si, o_si in zip(spacing, size, output_size)]
            output_spacing.append(new_spacing)
        else:
            output_spacing.append(spacing)
        
    else:
        if(fit_spacing):
            output_spacing3.append([sp*si/o_si for sp, si, o_si in zip(spacing, size, output_size)])
        else:
            output_spacing3.append(spacing)
    
    return output_spacing, output_spacing3

def operationSpacing(output_spacing, args):
    spacing_wMax= False 
    
    if (spacing_wMax):
    # output_spacing_filtered = [sp for si, sp in zip(args.size, output_spacing) if si != -1] 
    # print('output spacing filtered',output_spacing_filtered)
        operation_spacing = np.max(output_spacing)
    
    else:
        #calculate the mean spacing
        operation_spacing = [sum(x)/len(output_spacing) for x in zip(*output_spacing)]
        # Take the max spacing to have the same spacing for all the axis
        operation_spacing = np.max(operation_spacing)

    output_spacing = [sp if si == -1 else round(operation_spacing,3) for si, sp in zip(args.size, output_spacing)]
    
    return output_spacing

def Resample(img_filename,output_spacing, args):

    print("Reading:", img_filename) 
    img = sitk.ReadImage(img_filename)

    return resample_fn(img, output_spacing, args)

def main_resample(args):
    filenames = []
    if(args.img):
        fobj = {}
        fobj["img"] = args.img
        fobj["out"] = args.out
        filenames.append(fobj)
    elif(args.dir):
        out_dir = args.out
        normpath = os.path.normpath("/".join([args.dir, '**', '*']))
        for img in glob.iglob(normpath, recursive=True):
            if os.path.isfile(img) and True in [ext in img for ext in [".nrrd", ".nii", ".nii.gz", ".mhd", ".dcm", ".DCM", ".jpg", ".png"]]:
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

    output_spacing_list = []
    output_spacing_other_list = []
    idx =0
    for fobj in filenames:
        output_spacing_list, output_spacing_other_list= AddSpacing(fobj["img"],output_spacing_list,output_spacing_other_list, args)
        idx+=1
        time.sleep(0.01)
        if idx%10 == 0:
            print_progress_bar(idx, len(filenames), prefix='Progress:', suffix='Complete ', length=50)
  
    if (args.iso_spacing):
        output_spacing = operationSpacing(output_spacing_list, args)  
        output_spacing_other = operationSpacing(output_spacing_other_list, args)
    
    for fobj in filenames:
        try:

            if "ref" in fobj and fobj["ref"] is not None:
                ref = sitk.ReadImage(fobj["ref"])
                args.size = ref.GetSize()
                args.spacing = ref.GetSpacing()
                args.origin = ref.GetOrigin()

            if args.size is not None and args.suffix_res in fobj["img"]:
                img = Resample(fobj["img"],output_spacing, args)
            elif args.size is not None:
                img = Resample(fobj["img"],output_spacing_other, args)
            else:
                img = sitk.ReadImage(fobj["img"])

            print("Writing:", fobj["out"])
            writer = sitk.ImageFileWriter()
            writer.SetFileName(fobj["out"])
            writer.UseCompressionOn()
            writer.Execute(img)
            
        except Exception as e:
            print(e, file=sys.stderr)


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
    transform_group.add_argument('--spacing_wMax', type=bool, help='Use the max of the spacing or the mean (if False)', default=False)
    transform_group.add_argument('--origin', nargs="+", type=float, default=None, help='Output origin')
    transform_group.add_argument('--linear', type=bool, help='Use linear interpolation.', default=False)
    transform_group.add_argument('--center', type=bool, help='Center the image in the space', default=True)
    transform_group.add_argument('--fit_spacing', type=bool, help='Fit spacing to output', default=True)
    transform_group.add_argument('--iso_spacing', type=bool, help='Same spacing for resampled output', default=True)

    img_group = parser.add_argument_group('Image parameters')
    img_group.add_argument('--suffix_res', type=str,help='Suffix saying the resolution of the input image. Can only process with 2 different resolution', default='sp1')
    img_group.add_argument('--image_dimension', type=int, help='Image dimension', default=2)
    img_group.add_argument('--pixel_dimension', type=int, help='Pixel dimension', default=1)
    img_group.add_argument('--rgb', type=bool, help='Use RGB type pixel', default=False)

    out_group = parser.add_argument_group('Ouput parameters')
    out_group.add_argument('--ow', type=int, help='Overwrite', default=1)
    out_group.add_argument('--out', type=str, help='Output image/directory', default="./out.nrrd")
    out_group.add_argument('--out_ext', type=str, help='Output extension type', default=None)

    args = parser.parse_args()

    main_resample(args)