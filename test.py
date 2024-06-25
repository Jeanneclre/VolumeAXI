import pandas as pd
import numpy as np
import sklearn

from sklearn.model_selection import StratifiedKFold
import torch.nn as nn
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

def pred_vector():
    import torch
    penalty_weight =0.1
    prediction_vector = [[0, 1, 0, 0, 1, 0],[1,0,0,0,0,0],[0,0,0,0,0,1]]
    targets = [[0, 0, 0, 0, 1, 0],[1,0,0,0,0,0],[0,0.5,0,0,0,0.5]]
    targets2= [[0,1,0,0,1,0],[1,0,0,0,0,0],[0,0,0,0,0,1]]
    prediction_vector = torch.tensor(prediction_vector, dtype=torch.float32)
    targets = torch.tensor(targets, dtype=torch.float32)
    targets2 = torch.tensor(targets2, dtype=torch.float32)

    logits = torch.log(prediction_vector / (1 - prediction_vector + 1e-9))
    print('LOGITS',logits)

    base_loss = nn.functional.binary_cross_entropy(prediction_vector, targets, reduction='none')
    print('base_loss',base_loss)
    base_loss_S = nn.functional.binary_cross_entropy(prediction_vector, targets, reduction='sum')
    print('base_loss_S',base_loss_S)
    base_loss_M = nn.functional.binary_cross_entropy(prediction_vector, targets, reduction='mean')
    print('base_loss_M',base_loss_M)

    CE_loss = torch.nn.CrossEntropyLoss()
    CE_loss = CE_loss(prediction_vector, targets.argmax(dim=1))
    print('CE_loss',CE_loss)

    True_CE_loss = torch.nn.CrossEntropyLoss()
    True_CE_loss = True_CE_loss(prediction_vector, targets2.argmax(dim=1))
    print('True_CE_loss',True_CE_loss)

    loss_function = torch.nn.BCEWithLogitsLoss()
    # Compute the loss
    loss = loss_function(prediction_vector, targets)
    print('loss',loss)

    loss_true = loss_function(logits, targets2)
    print('loss_true',loss_true)

    print('----- Loss with probability multi-class -----')
    prediction_vector = torch.tensor([[0.1, 0.9, 0.1, 0.1, 0.9, 0.1],[0.9,0.1,0.1,0.1,0.1,0.1],[0.1,0.1,0.1,0.1,0.1,0.9]], dtype=torch.float32)
    bce_loss = torch.nn.BCELoss()
    bce_loss_false = bce_loss(prediction_vector, targets)
    bce_loss_true = bce_loss(prediction_vector, targets2)
    print('bce_loss_false',bce_loss_false)
    print('bce_loss_true',bce_loss_true)

    bce_logit_loss = torch.nn.BCEWithLogitsLoss()
    bce_logit_loss_false = bce_logit_loss(prediction_vector, targets)
    bce_logit_loss_true = bce_logit_loss(prediction_vector, targets2)
    print('bce_logit_loss_false',bce_logit_loss_false)
    print('bce_logit_loss_true',bce_logit_loss_true)

    # Calculate penalty for false positives on left or right
    print('prediction_vector',prediction_vector.shape)
    penalty_fp = 0
    penalty_fn = 0
    for j in range(prediction_vector.shape[0]):
        print('prediction_vector[j, :3]',prediction_vector[j, :3])
        left_false_positive = [ True if (prediction_vector[j, :3].sum() > 0) & (targets[j, :3].sum() == 0) else False]
        print('fp',left_false_positive)
        right_false_positive = [True if (prediction_vector[j, 3:].sum() > 0) & (targets[j, 3:].sum() == 0) else False]

        # Calculate penalty for false negatives on left or right
        left_false_negative = [True if (prediction_vector[j, :3].sum() == 0) & (targets[:, :3].sum() > 0) else False]
        print('fn',left_false_negative)
        right_false_negative = [True if (prediction_vector[:, 3:].sum() == 0) & (targets[:, 3:].sum() > 0) else False]

        if left_false_positive or right_false_positive:
            penalty_fp +=1
        if left_false_negative or right_false_negative:
            penalty_fn +=1


    penalty = penalty_weight * (penalty_fp + penalty_fn)

    print('penalty',penalty)

    total_loss = True_CE_loss + penalty
    print('total',total_loss)


def test_vector():
    import torch
    targets= [[0, 0, 0, 0, 1, 0],[1,0,0,0,0,0],[0,0.5,0,0,0,0.5]]
    prediction_vector = [[0, 1, 0, 0, 1, 0],[1,0,0,0,0,0],[0,0.5,0,0.5,0,0]]

    #to tensor
    targets = torch.tensor(targets, dtype=torch.float32)
    prediction_vector = torch.tensor(prediction_vector, dtype=torch.float32)
    penalty_bonus=0
    for j in range(prediction_vector.shape[0]):
        best_probL = prediction_vector[j, :3].max()
        best_probR = prediction_vector[j, 3:].max()
        idx_non_zero_target = torch.where(targets[j] != 0)
        idx_non_zero_pred = torch.where(prediction_vector[j] != 0)
        # get the value from the tensor
        # idx_non_zero_target = idx_non_zero_target[0].tolist()
        # idx_non_zero_pred = dict(zip(idx_non_zero_pred[0].tolist(), idx_non_zero_pred[1].tolist()))
        print('idx_non_zero_target shape',[tensor.size()[0] for tensor in idx_non_zero_target])
        print('idx_non_zero_target',[tensor[0] for tensor in idx_non_zero_target])
        print('idx_non_zero_pred',idx_non_zero_pred)
        # if idx_non_zero_pred == idx_non_zero_target:
        #     if best_probR > 0.7:
        #         penalty_bonus += 1
        #     if best_probL > 0.7:
        #         penalty_bonus += 1
    print('penalty_bonus',penalty_bonus)


def rocAUC():
    import torch
    import sklearn.metrics as mt
    from sklearn.metrics import roc_auc_score
    from monai.metrics import compute_roc_auc

    preds_sig =  [[3.9941e-01, 1.5503e-01, 2.3230e-01, 7.5146e-01, 9.2676e-01, 6.6345e-02],
        [5.0812e-02, 1.1871e-02, 9.9756e-01, 4.8267e-01, 1.5479e-01, 2.6050e-01],
        [1.1467e-02, 4.6015e-04, 8.5156e-01, 5.1807e-01, 9.7119e-01, 9.3945e-01],
        [3.3283e-04, 1.0321e-01, 9.2334e-01, 6.2132e-04, 9.8975e-01, 9.8682e-01]]
    targets =[[0.0000, 1, 0.0000, 0.0000,1, 0.0000],
        [0.0000, 0.0000, 1, 0.0000, 0.0000, 0.0000],
        [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1],
        [0.0000, 0.0000, 0.0000, 0.0000, 0, 1]]

    target_label = [[2],[5],[4]]

    preds_soft =nn.functional.softmax(torch.tensor(preds_sig), dim=1)
    print('preds_soft',preds_soft)

    preds_sig = torch.tensor(preds_sig, dtype=torch.float32)
    targets = torch.tensor(targets, dtype=torch.float32)
    target_label = torch.tensor(target_label, dtype=torch.float32)

    print('targets:',targets[:,2])
    print('preds_soft[0,0]:',preds_soft[:,2])

    auc_monai = compute_roc_auc(preds_soft[:,2], targets[:,2])
    print('AUC monai',auc_monai)
    auc = roc_auc_score(targets[:,2], preds_soft[:,2], average='macro',multi_class='ovr')
    print('AUC',auc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Dataset information')
    parser.add_argument('--csv', required=False, type=str, help='CSV to count and rename classes')
    parser.add_argument('--class_column', type=str, default='Classification', help='Name of class column')
    args = parser.parse_args()

    test_vector()
