'''
Date: November 2023
Author: Jeanne Claret

Description: This file contains the information about the dataset
such as the number of data per type, the number of classes.

Used to rename the classes in the dataset.
'''

import csv
import argparse
import pandas as pd

# Count the number of different classes in the column "Position" of te csv file

def count_classes(csv_file,word_class='Classification',dict_classes={}):
    reader = pd.read_csv(csv_file)
    classes = {}
    output_file = csv_file.split('.')[0] + '_classes.csv'

    for index, row in reader.iterrows():
        key_name = str(row[word_class]).split(' ')[0]
        ## For Sara's dataset
        if key_name == 'Lingual':
            key_name  = 'Palatal'
        if key_name == 'Bucccal':
            key_name = 'Buccal'
        if key_name == 'BuccaL':
            key_name = 'Buccal'
        if classes.get(key_name) is None:
            classes[key_name] = 1
        classes[key_name] += 1

        # Change the name of the classes
        reader.loc[index,word_class] = dict_classes[key_name]


        reader.to_csv(output_file, index=False)


    return classes

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Dataset information')
    parser.add_argument('--input', required=True, type=str, help='CSV')
    parser.add_argument('--class_column', type=str, default='Classification', help='Name of class column')

    args = parser.parse_args()
    dict_classes = {
        "Bicortical":0,
        "Palatal": 1,
        "Buccal": 2,
        "nan": '' ,
    }

    classes = count_classes(args.input,args.class_column,dict_classes)
    print(classes)