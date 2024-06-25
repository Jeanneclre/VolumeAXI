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
        #if key_name is already an integer, count the number of similar int
        #remove all empty value for row[word_class]
        if pd.isnull(row[word_class]):
            #delete the row
            # reader = reader.drop(index)
            # reader.to_csv(output_file, index=False)
            # print(f'[INFO] Deleted row {index} because of empty value')
            # put the empty value to None
            reader.loc[index,word_class] = 10
            key_name = reader.loc[index,word_class]


        else:
            if isinstance(row[word_class],int):
                key_name = row[word_class]


            elif isinstance(row[word_class],float):
                #change type of float to int
                key_name = int(row[word_class])
                #rewrite the csv file
                reader.loc[index,word_class] = key_name
                reader.to_csv(output_file, index=False)

            elif isinstance(row[word_class],str):
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
        else:
            classes[key_name] += 1


        if not args.change_classes:
            if not isinstance(row[word_class],int):
                if isinstance(row[word_class],float):
                    continue
                # Change the name of the classes
                reader.loc[index,word_class] = dict_classes[key_name]
                reader.to_csv(output_file, index=False)
        else:
            reader.loc[index,word_class] = dict_classes[key_name]
            reader.to_csv(output_file, index=False)


    return classes

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Dataset information')
    parser.add_argument('--input', required=True, type=str, help='CSV to count and rename classes')
    parser.add_argument('--class_column', type=str, default='Label', help='Name of class column')
    parser.add_argument('--change_classes', type=bool, default=False, help='Change the name of the classes (uses the dict)')

    args = parser.parse_args()
    # Classification of the position
    # dict_classes = {
    #     "Buccal": 0,
    #     "Bicortical":1,
    #     "Palatal": 2,
    #     "nan": '' ,
    # }

    # Classification No Damage (0) /Damage (1)
    # dict_classes = {
    #     0:0,
    #     1:1,
    #     2:1,
    #     3:1,
    # }

    # No damage:0, Mild damage:1, Severe + Extreme damage:2
    # dict_classes={
    #     0:0,
    #     1:1,
    #     2:2,
    #     3:2,
    # }

    # dict_classes ={
    #     10: 0,
    #     0: 1,
    #     1: 2,
    #     2: 3,

    # }

    # dict_classes = {
    #     10: 4,
    #     3: 5,
    #     4: 6,
    #     5: 7,
    # }
    classes = count_classes(args.input,args.class_column,dict_classes)
    print(classes)