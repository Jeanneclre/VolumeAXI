import Preprocess.resample as resample
import Preprocess.dataset_augmentation as augm
import Preprocess.create_input as inp

import argparse

def main(args):
    input_dir = args.input_dir
    output_dir = args.output_dir

    size = args.size

    angle = args.angle
    axis = args.axis

    filename = args.filename

    resample.main_resample(input_dir,output_dir,size)

    output_dir_rot= output_dir + '/rotated'

    augm.main_rotation(output_dir_rot,angle,axis)

    output_dir_csv= output_dir + '/input'

    inp.main_input(output_dir,filename,output_dir_csv)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='main of VolumeAXI', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    file_group = parser.add_argument_group('File(s) parameters')
    file_group.add_argument('--input_dir', type=str, help='Input directory')
    file_group.add_argument('--output_dir', type=str, help='Output directory')

    resample_param = parser.add_argument_group('Resample parameters')
    resample_param.add_argument('--size', nargs="+", type=int, help='Output size of the image, -1 to leave unchanged', default=None)

    rotation_param = parser.add_argument_group('Rotation parameters')
    rotation_param.add_argument('--angle', type=float, help='Angle of rotation', default=90)
    rotation_param.add_argument('--axis', type=list, help='Axis of rotation ', default=(0,0,1))

    complete_csv = parser.add_argument_group('Completion of the input csv')
    complete_csv.add_argument('--filename',help='name of the xlsx file with the different patients and class to complete',type=str,default='./')
    complete_csv.add_argument('--side',help='list with all the side of the scan possible',type=int,default=['Left','Right','Bilateral'])

    args = parser.parse_args()

    main(args)