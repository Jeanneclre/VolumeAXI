import SimpleITK as sitk
import numpy as np
import argparse
import cv2
import matplotlib.pyplot as plt
import os

def main(args,img,output_filename):

    print("Reading:", img)
    sitk_img = sitk.ReadImage(img)
    #check image is in Grey level
    if sitk_img.GetNumberOfComponentsPerPixel() > 1:
        print("Image is not in Grey level")
        #convert into grey level
        sitk_img = sitk.VectorIndexSelectionCast(sitk_img, 0, sitk.sitkFloat32)

    img_norm = sitk.Normalize(sitk_img)
    sitk_out = CorrectHisto(img_norm, args.min_percentile, args.max_percentile, args.i_min, args.i_max,args.num_bins,output_filename)

    print("Writing:", output_filename)

    sitk.WriteImage(sitk_out, output_filename)


def CorrectHisto(sitk_img, min_percentile=0.01, max_percentile = 0.99, i_min=0, i_max=255, num_bins=1000,output_filename='file.nii.gz'):


    print("Correcting scan contrast...")
    # Normalize img before histogram correction
    img_np_BC = sitk.GetArrayFromImage(sitk_img)


    ######3 Region test
    # Calculate the percentile values for outlier cutoff
    lower_bound = np.percentile(img_np_BC, 1)
    upper_bound = np.percentile(img_np_BC, 90)

    # Clip the pixel values to the lower and upper bounds
    img_np = np.clip(img_np_BC, lower_bound, upper_bound)
    # end region test

    img_min = np.min(img_np)
    img_max = np.max(img_np)
    img_range = img_max - img_min
    print('img_min:',img_min, 'img_max:',img_max, 'img_range:',img_range)

    sum_voxel = img_np.size
    print('sum_voxel:',sum_voxel)

    # definition = num_bins
    definition = num_bins
    histo = np.histogram(img_np, definition)

    cdf = np.cumsum(histo[0])
    cdf_normalized = cdf / cdf.max()
    print('cdf shape norm:',cdf_normalized.shape)

    cum = cdf - np.min(cdf)
    cum = cum / np.max(cum)
    print('cum shape:',cum.shape)

    plt.figure(figsize=(10, 4))
    plt.plot(cum, label='CDF')
    plt.axhline(y=min_percentile, color='r', linestyle='-', label=f'Min percentile ({min_percentile})')
    plt.axhline(y=max_percentile, color='g', linestyle='-', label=f'Max percentile ({max_percentile})')
    plt.legend()
    plt.show()


    # res_high = np.argmax(cum > max_percentile)

    res_high = list(map(lambda i: i> max_percentile, cum)).index(True)
    res_max = (res_high * img_range)/definition + img_min

    # res_low = np.argmax(cum > min_percentile)
    res_low = list(map(lambda i: i> min_percentile, cum)).index(True)
    res_min = (res_low * img_range)/definition + img_min


    res_min = max(res_min, i_min)
    res_max = min(res_max, i_max)


    print("Min:", res_min, "Max:", res_max)

    img_np = np.where(img_np > res_max, res_max, img_np)
    img_np = np.where(img_np < res_min, res_min, img_np)


    nb_pixel_intensity_null =np.where(img_np == 0)[0].shape[0]
    print('nb_pixel_intensity_null:',nb_pixel_intensity_null)

    output = sitk.GetImageFromArray(img_np)
    output.SetSpacing(sitk_img.GetSpacing())
    output.SetDirection(sitk_img.GetDirection())
    output.SetOrigin(sitk_img.GetOrigin())
    output = sitk.Cast(output, sitk.sitkInt16)

    output_hist = np.histogram(img_np, definition)
    cdf_output = np.cumsum(output_hist[0])
    cdf_output_normalized = cdf_output / cdf_output.max()

    # calcul cdf wanted which is a linear function where a =0.000875 and b =0.475
    # cdf_wanted = np.linspace(0,1,definition)

    #print min intensity level in the output image
    print('min intensity level:',np.min(output))
    print('max intensity level:',np.max(output))

    intensity_level= np.linspace(res_min,res_max,definition)
    cdf_wanted = 0.000875*intensity_level + 0.475
    plt.figure(figsize=(20, 10))
    plt.subplot(2, 3, 1)
    plt.plot(intensity_level,histo[0],color = 'b')
    plt.title('Initial Histogram')

    plt.subplot(2, 3, 4)
    plt.plot(intensity_level,cdf_normalized, color = 'r')
    plt.title('Cumulative Distribution Function')

    plt.subplot(2, 3, 2)
    plt.plot(intensity_level,output_hist[0])
    plt.title('Corrected Histogram')

    plt.subplot(2, 3, 5)
    plt.plot(intensity_level,cdf_output_normalized)
    plt.title('Corrected Cumulative Distribution Function')

    plt.subplot(2, 3, 3)
    plt.plot(intensity_level,histo[0],color = 'b')
    plt.plot(intensity_level,output_hist[0],color = 'r')
    plt.legend(['Initial Histogram','Corrected Histogram'])
    plt.title('Initial and Corrected Histogram')

    plt.subplot(2, 3, 6)
    plt.plot(intensity_level,cdf_normalized, color = 'b')
    plt.plot(intensity_level,cdf_output_normalized, color = 'r')
    plt.plot(intensity_level,cdf_wanted, color = 'g')
    plt.legend(['Initial cdf','Corrected cdf','Wanted cdf'])
    plt.title('Initial and Corrected CDF')

    dir_name = os.path.dirname(output_filename)
    print('dir_name:',dir_name)
    plt.savefig(f'{dir_name}/histo_maxP{max_percentile}_minP{min_percentile}_bins{definition}.png')

    plt.show()



    return output

if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Correct image histo')
    parser.add_argument('--folder', type=str, help='Image folder to correct histogram', required=True)
    parser.add_argument('--min_percentile', type=float, help='Minimum percentile', default=0.20)
    parser.add_argument('--max_percentile', type=float, help='Maximum percentile', default=0.85)
    parser.add_argument('--i_min', type=int, help='Minimum intensity', default=-1500)
    parser.add_argument('--i_max', type=int, help='Maximum intensity', default=4000)
    parser.add_argument('--num_bins', type=int, help='Number of bins', default=50)
    parser.add_argument('--out', type=str, help='Output folder for image ', default='/Hist_Corrected')

    args = parser.parse_args()

    if not os.path.exists(args.out):
        os.makedirs(args.out)

    for root, dirs, files in os.walk(args.folder):
        print('root:',root)
        in_folder = root.split('/')[0]

        for file in files:
            filename_out = file.replace('.nii.gz','_corrected.nii.gz')
            if file.endswith(".nii.gz"):
                img = os.path.join(root, file)
                out_root = root.replace(in_folder, args.out)
                if not os.path.exists(out_root):
                    os.makedirs(out_root)
                output_filename = os.path.join(out_root, filename_out)
                main(args,img,output_filename)
