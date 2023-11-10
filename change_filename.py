import os
import shutil
import argparse

def rename_and_copy_files(args):
    src_dir = args.dir
    dest_dir = args.out_dir
    for root, dirs, files in os.walk(src_dir):
        # Determine the path to the destination directory, replacing the top-level directory name
        relative_path = os.path.relpath(root, src_dir)
        new_relative_path = relative_path
        if relative_path == '.':
            new_relative_path = dest_dir
        else:
            new_relative_path = os.path.join(dest_dir, relative_path)
        new_dest_path = os.path.join(dest_dir, new_relative_path)

        # Create the destination directory if it doesn't already exist
        os.makedirs(new_dest_path, exist_ok=True)

        # Rename and copy each file
        for file in files:
            if "Sara" in file:
                print('Sara')
                new_file_name = file.replace('_0', '_MN')
                print('new_file_name', new_file_name)
            else:
                new_file_name = file

            src_file_path = os.path.join(root, file)
            dest_file_path = os.path.join(new_dest_path, new_file_name)
            shutil.copy2(src_file_path, dest_file_path)
         

         

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir',help='directory with the input files',type=str,default='./')
    parser.add_argument('--out_dir',help='directory where the output files are stored',type=str,default='./output')
    args = parser.parse_args()


    rename_and_copy_files(args)
