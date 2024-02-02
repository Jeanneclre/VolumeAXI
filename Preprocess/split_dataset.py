'''
Creating csv files for train, validation and test sets

Author: Jeanne Claret
'''

import argparse
import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

def main(args):
    df = pd.read_csv(args.input)
    df_train, df_test = train_test_split(df, test_size=args.test_size, random_state=args.seed)
    df_train, df_val = train_test_split(df_train, test_size=args.val_size, random_state=args.seed)


    output_filename = args.out_dir + '/' + args.output_train
    df_train.to_csv(output_filename, index=False)

    output_filename = args.out_dir + '/' + args.output_val
    df_val.to_csv(output_filename, index=False)

    output_filename = args.out_dir + '/' + args.output_test
    df_test.to_csv(output_filename, index=False)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Split dataset into train, validation and test sets')
    parser.add_argument('--input', required=True, type=str, help='CSV')
    parser.add_argument('--out_dir', required=True, type=str, help='Output directory')
    parser.add_argument('--output_train', required=True, type=str, help='CSV')
    parser.add_argument('--output_val', required=True, type=str, help='CSV')
    parser.add_argument('--output_test', required=True, type=str, help='CSV')
    parser.add_argument('--test_size', type=float, default=0.2, help='Test size')
    parser.add_argument('--val_size', type=float, default=0.2, help='Validation size')
    parser.add_argument('--seed', type=int, default=42, help='Seed')

    args = parser.parse_args()


    #create the output directory if non existent
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)


    main(args)
