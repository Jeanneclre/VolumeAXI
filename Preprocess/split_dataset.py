'''
Creating csv files for train, validation and test sets

Author: Jeanne Claret
'''
import argparse
import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

def out_train_test_split(args):
    '''
    Function to split the dataset into training and testing sets.
    The training set will be used in Cross Validation to be splitted into training, testing and validation sets.
    The testing set will be used to evaluate the best model of the Cross Validation.
    '''
    df= pd.read_csv(args.input)
    df_train, df_test = train_test_split(df, test_size=args.test_size, random_state=args.seed,stratify=df[args.class_column]) #stratify to keep the same distribution of classes in the train and test set

    output_train = args.out_dir + '/out_train_set.csv'
    df_train.to_csv(output_train, index=False)

    output_test = args.out_dir + '/out_test_set.csv'
    df_test.to_csv(output_test, index=False)

def train_test_validation_split(args):
    '''
    Function for the first version of the training. We need in input train, test and validation sets.
    '''
    df = pd.read_csv(args.input)
    df_train, df_test = train_test_split(df, test_size=args.test_size, random_state=args.seed,stratify=df[args.class_column]) #stratify to keep the same distribution of classes in the train and test set
    df_train, df_val = train_test_split(df_train, test_size=args.val_size, random_state=args.seed,stratify=df_train[args.class_column])


    output_filename = args.out_dir + '/' + args.output_train
    df_train.to_csv(output_filename, index=False)

    output_filename = args.out_dir + '/' + args.output_val
    df_val.to_csv(output_filename, index=False)

    output_filename = args.out_dir + '/' + args.output_test
    df_test.to_csv(output_filename, index=False)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Split dataset into train, validation and test sets')
    in_group = parser.add_argument_group('Input, output and split option')
    in_group.add_argument('--input', required=True, type=str, help='CSV')
    in_group.add_argument('--out_dir', required=True, type=str, help='Output directory')
    in_group.add_argument('--split_option', required=True, type=str, help='Option to split the dataset: out_train_test_split (TT) or train_test_validation_split (TTV)', choices=["TT","TTV"])

    split_TTV_group = parser.add_argument_group('Split Train, Test and Validation')
    split_TTV_group.add_argument('--output_train', type=str, help='CSV')
    split_TTV_group.add_argument('--output_val', type=str, help='CSV')
    split_TTV_group.add_argument('--output_test', type=str, help='CSV')

    parser.add_argument('--test_size', type=float, default=0.15, help='Test size')
    parser.add_argument('--val_size', type=float, default=0.15, help='Validation size')
    parser.add_argument('--class_column', type=str, default='Label', help='Name of class column')
    parser.add_argument('--seed', type=int, default=42, help='Seed')

    args = parser.parse_args()

    #create the output directory if non existent
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    if args.split_option == "TT":
        out_train_test_split(args)

    elif args.split_option == "TTV":
        train_test_validation_split(args)

