import pandas as pd
import numpy as np
import sklearn

from sklearn.model_selection import StratifiedKFold
import os

import argparse
from useful_readibility import printRed
def main(args):
    # Load data
    df = pd.read_csv(args.csv)

    df_filtered_train = df.dropna(subset=[args.class_column])
    df_filtered_train = df_filtered_train.reset_index(drop=True)

    unique_classes = np.unique(df_filtered_train[args.class_column])
    # unique_class_weights = np.array(class_weight.compute_class_weight(class_weight='balanced', classes=unique_classes, y=df_filtered_train[args.class_column]))

    class_replace = {}
    for cn, cl in enumerate(unique_classes):
        class_replace[int(cl)] = cn


    df_filtered_train = df.dropna(subset=[args.class_column])
    df_filtered_train = df_filtered_train.reset_index(drop=True)
    df_filtered_train[args.class_column] = df_filtered_train[args.class_column].replace(class_replace)
    kf = StratifiedKFold(n_splits=6, shuffle=True, random_state=42)
    for i, (train_index, test_index) in enumerate(kf.split(df_filtered_train, df_filtered_train[args.class_column])):
            #################################
            #                               #
            #          Training             #
            #                               #
            #################################
            printRed(f'FOLD: {i}')
            print('len train_index',len(train_index))
            print('len test_index',len(test_index))
            print('len df_filtered_train',len(df_filtered_train))

            df_train = df_filtered_train.iloc[train_index]
            df_test = df_filtered_train.iloc[test_index]

            print('len df_train',len(df_train))
            print('len df_test',len(df_test))

            # Save the train and test data
            output_dir = './'

            df_train.to_csv(os.path.join(args.csv.split('.')[0] + f'_train_{i}.csv'), index=False)
            df_test.to_csv(os.path.join(args.csv.split('.')[0] + f'_test_{i}.csv'), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Dataset information')
    parser.add_argument('--csv', required=True, type=str, help='CSV to count and rename classes')
    parser.add_argument('--class_column', type=str, default='Classification', help='Name of class column')
    args = parser.parse_args()
    main(args)
