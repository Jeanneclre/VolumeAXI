import os
import argparse

def get_patient_name(file_path, word_list):

    file_name = os.path.basename(file_path).split('_')[0]

    for word in word_list:
        if word in file_path:
            if file_name == 'CP47':
                print('file_path',file_path)
                print(file_name + f'_{word}')

            return file_name + f'_{word}'
        else:

            if file_name == 'CP47':
                print('file_path',file_path)
                print('filename',file_name)

            return file_name

def compare_folders(folder1, folder2):
    files1 = [get_patient_name(file, ['MB', 'MR', 'MR']) for file in os.listdir(folder1)]
    files2 = [get_patient_name(file, ['MB', 'MR', 'MR']) for file in os.listdir(folder2)]

    print('len files1', len(files1))
    print('len files2', len(files2))
    missing_files = [file for file in files2 if file not in files1]
    duplicate_files = [file for file in files1 if file in files2]

    return missing_files,duplicate_files

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare files in two folders")
    parser.add_argument("--folder1", help="Path to the first folder")
    parser.add_argument("--folder2", help="Path to the second folder")
    args = parser.parse_args()

    missing_files,duplicate_files = compare_folders(args.folder1, args.folder2)
    print("Missing files in folder1:")
    for file in missing_files:
        print(file)

    print("Duplicate files in folder1:")
    for file in duplicate_files:
        print(file)

