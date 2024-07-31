import os
import shutil
import argparse

'''
Script to group all the scans files in the same folder.
Practical to count and preprocess the scans.

'''

def regroup_scans(input_dir, output_dir):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Iterate over all subdirectories in the input directory
    for root, dirs, files in os.walk(input_dir):
        print("root", root)
        print("files", files)
        for file in files:
            # Check if the file is a NIfTI file and doesn't contain 'sp1' or 'Seg' in its name
            if file.endswith('.nii.gz') and 'sp1' not in file and 'Seg' not in file:
                # Construct the source and destination paths
                src_path = os.path.join(root, file)
                dst_path = os.path.join(output_dir, file)
                print('dst_path',dst_path)

                # Copy the file to the output directory
                shutil.copy2(src_path, dst_path)



if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Regroup scans')
    parser.add_argument('--input', type=str, help='Input directory containing the scans')
    parser.add_argument('--output', type=str, help='Output directory to copy the scans')
    args = parser.parse_args()

    regroup_scans(args.input, args.output)