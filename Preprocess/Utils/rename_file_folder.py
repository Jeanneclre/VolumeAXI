import os
import argparse
import os
import shutil

'''
Script to rename subfolder and files of an input folder.

For example:
input:
Input_Folder/Bilateral/Dylan_CP01/Dylan_CP01_scan_sp0-3.nii.gz

output:
Output_Folder/Bilateral/CP01/CP01_scan.nii.gz

'''

def rename_files_folders(input_folder, old_words, new_words, output_folder,same_fold):
    for root, dirs, files in os.walk(input_folder, topdown=True):

        # Skip renaming the top-level subfolders
        depth = root[len(input_folder):].count(os.sep)
        if depth == 0 and same_fold ==True:
            for name in files + dirs:
                if name.endswith('.nii.gz'):  # Check if it's a file to be renamed or a deeper directory
                    original_path = os.path.join(root, name)
                    new_name = name

                    for old_word, new_word in zip(old_words, new_words):
                        new_name = new_name.replace(old_word, new_word)

                    relative_path = os.path.relpath(root, input_folder)
                    new_path = os.path.join(output_folder, relative_path, new_name)

                    # Create directory if it doesn't exist
                    os.makedirs(os.path.dirname(new_path), exist_ok=True)
                    # Rename file or directory
                    # Create a copy of the file written in the new path
                    shutil.copy2(original_path, new_path)

                    # os.rename(original_path, new_path)
                    print(f"Renamed: {original_path} -> {new_path}")

        elif depth==0 and same_fold ==False:
            continue  # This skips the top level directories

        else:
            # Renaming directories and files in deeper levels
            for name in files + dirs:
                if name.endswith('.nii.gz') or depth > 0:  # Check if it's a file to be renamed or a deeper directory
                    original_path = os.path.join(root, name)
                    new_name = name
                    relative_path = os.path.relpath(root, input_folder)
                    new_subfolder = str(relative_path.split('/')[1]) # Choose the subfolder to rename

                    for old_word, new_word in zip(old_words, new_words):
                        new_name = new_name.replace(old_word, new_word)
                        if old_word in new_subfolder:
                            new_subfolder = new_subfolder.replace(old_word, new_word)
                        else:
                            new_subfolder = new_subfolder

                    new_relative_path = relative_path.split('/')[0]+ '/'+new_subfolder

                    new_path = os.path.join(output_folder, new_relative_path, new_name)

                    # Create directory if it doesn't exist
                    os.makedirs(os.path.dirname(new_path), exist_ok=True)
                    # Rename file or directory
                    # Create a copy of the file written in the new path
                    shutil.copy2(original_path, new_path)

                    # os.rename(original_path, new_path)
                    print(f"Renamed: {original_path} -> {new_path}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rename files and folders")
    parser.add_argument("--input", help="Path to the input folder")
    parser.add_argument("--old_words", nargs="+", help="Words to be replaced")
    parser.add_argument("--new_words", nargs="+", help="New words")
    parser.add_argument("--same_fold", help="True if the top-level folders are the same", default=True, type=bool)
    parser.add_argument("--output", help="Path to the output folder")
    args = parser.parse_args()

    old_words = ['MRMatrix_mirror','MBMatrix_mirror','__MBMatrix_mirror','_cropped']
    new_words = ['MR', 'MB','_MB','']
    rename_files_folders(args.input, old_words, new_words, args.output, args.same_fold)
