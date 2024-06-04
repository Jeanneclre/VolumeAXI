import csv

import pandas as pd
import argparse

def main(args,dict):
    # Load csv file
    df = pd.read_csv(args.csv_file)

    # Change the name of the class
    for idx, row in df.iterrows():
        patient_name = row[args.patient_column]
        for key in dict.keys():
            if key in patient_name:
                df.loc[idx, args.class_column] = dict[key][row[args.class_column]]
                break
    # save the new csv
    df.to_csv(args.output, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Label Modification')
    parser.add_argument('--csv_file', required=False, type=str, help='CSV file')
    parser.add_argument('--class_column', type=str, default='Label', help='Name of class column')
    parser.add_argument('--patient_column', type=str, default='Name', help='Name of patient column')
    parser.add_argument('--output', type=str, default='output.csv', help='Output CSV file')
    args = parser.parse_args()

    dict = {
        '_R': {
                0:0,
                1:1,
                2:2,
            },
        '_L': {
                0:3,
                1:4,
                2:5,
            }
    }
    # print('dict keys:',dict.keys())

    main(args,dict)