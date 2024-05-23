'''
In the case of the use of special data (cropped images from the same dataset for example) to improve the model performance,
it is necessary to make sure we don't put special data that 'are' already in the testing set to avoid overfitting.

To create the special csv file, we will look at all the patients names in the training set and make sure we only add the
same patients in the special data csv file.
'''
import os
import pandas as pd
import argparse
import glob

def main(args):
    # Load csv file
    df_train = pd.read_csv(args.csv_train)
    df_special = pd.read_csv(args.csv_special)

    patients_list = df_train['Name'].unique()

    # loop over the directory with all the scans
    # use iglob
    header = ['Path', 'Name', 'Label']

    #define new csv file initialized with data from df_train
    df_train_special = pd.DataFrame(columns=header)

    cpt=0
    for idx,row in df_special.iterrows():
        patient = row['Name']
        if patient in patients_list:
            cpt+=1
            df_train_special = df_train_special._append(row, ignore_index=True)

    df_train_special.to_csv(args.output_csv, index=False)
    print(f"Number of patients in the special data added to training: {cpt}")

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Create a special csv file')
    parser.add_argument('--csv_train', required=True, type=str, help='Training CSV file')
    parser.add_argument('--csv_special', required=True, type=str, help='Special data CSV file')
    parser.add_argument('--output_csv', required=True, type=str, help='Output path to the CSV file')
    args = parser.parse_args()
    main(args)

