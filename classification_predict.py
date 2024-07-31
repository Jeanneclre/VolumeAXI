import argparse

import math
import os
import pandas as pd
import numpy as np

import torch
from torch.utils.data import DataLoader

from nets.classification import Net, NetTarget, NetFC
from loaders.dataset import BasicDataset, Datasetarget, DatasetFC
from transforms.volumetric import EvalTransforms, SegEvalTransforms, NoEvalTransform
from callbacks.logger import ImageLogger

from sklearn.utils import class_weight
from sklearn.metrics import classification_report, roc_curve, auc

from monai.metrics import compute_roc_auc

from useful_readibility import printRed, printBlue, printGreen
from tqdm import tqdm

import pickle
import matplotlib.pyplot as plt
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

def plot_roc_curve_final(probs, truths, class_idx,out_path):
    '''
    function used as final step to plot the last interested class roc curve to the plot
    '''
    fpr, tpr, _ = roc_curve(truths, probs)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f'Class {class_idx} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC AUC ')
    plt.legend(loc="lower right")
    plt.savefig(out_path)
    plt.close()

def add_roc_curve(probs, truths, class_idx):
    '''
    function used to add a roc curve to the plot
    '''
    fpr, tpr, _ = roc_curve(truths, probs)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f'Class {class_idx} (AUC = {roc_auc:.2f})')

def MultiBranchPred(args):
    '''
    Prediction function for 2 class columns and 2 branches in the model (one for each column).
    '''
    model = NetFC(seed=args.seed).load_from_checkpoint(args.model)
    model.eval()
    model.cuda()


    if (os.path.splitext(args.csv)[1] == ".csv"):
        df_train = pd.read_csv(args.csv_train)
        df_test = pd.read_csv(args.csv)
    else:
        df_train = pd.read_parquet(args.csv_train)
        df_test = pd.read_parquet(args.csv)

    print('nb_classes in MultiBranchPred',args.nb_classes)
    test_ds = DatasetFC(df_test, mount_point=args.mount_point, img_column=args.img_column, class_column1=args.class_column1, class_column2=args.class_column2,nb_classes=args.nb_classes, transform=EvalTransforms(args.img_size))
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, prefetch_factor=4)

    with torch.no_grad():
        predictions1 = []
        predictions2 = []
        probs = []
        features = []
        for idx, batch in tqdm(enumerate(test_loader), total=len(test_loader)):
            X, Y1,Y2 = batch

            X = X.cuda().contiguous()

            pred1,pred2 = model(X)
            pred_prob1 = torch.nn.functional.softmax(pred1, dim=1)
            pred_prob2 = torch.nn.functional.softmax(pred2, dim=1)

            prob_concat = torch.cat([pred_prob1, pred_prob2], dim=1)
            probs.append(prob_concat.cpu().numpy())
            predictions1.append(torch.argmax(pred1, dim=1).cpu().numpy())
            predictions2.append(torch.argmax(pred2, dim=1).cpu().numpy())

    pred_column1 = args.pred_column + args.diff[0]
    pred_column2 = args.pred_column + args.diff[1]
    df_test[pred_column1] = np.concatenate(predictions1, axis=0)
    df_test[pred_column2] = np.concatenate(predictions2, axis=0)
    probs = np.concatenate(probs, axis=0)

    if not os.path.exists(args.out):
        os.makedirs(args.out)

    print('==== Classification report concat===')
    concat_class = np.concatenate([df_test[args.class_column1], df_test[args.class_column2]], axis=0)
    concat_pred = np.concatenate([df_test[pred_column1], df_test[pred_column2]], axis=0)
    print(classification_report(concat_class, concat_pred))

    ext = os.path.splitext(args.csv)[1]
    if(ext == ".csv"):
        df_test.to_csv(os.path.join(args.out, os.path.basename(args.csv).replace(".csv", "_prediction.csv")), index=False)
    else:
        df_test.to_parquet(os.path.join(args.out, os.path.basename(args.csv).replace(".parquet", "_prediction.parquet")), index=False)


    pickle.dump(probs, open(os.path.join(args.out, os.path.basename(args.csv).replace(ext, "_probs.pickle")), 'wb'))

    if len(features) > 0:
        features = np.concatenate(features, axis=0)
        pickle.dump(features, open(os.path.join(args.mount_point, args.out, os.path.basename(args.csv).replace(ext, "_prediction.pickle")), 'wb'))

def MultiPred(args):
    '''
    Function to handle prediction from a dataset with 2 different class columns.
    for example, csv file:
    Path, Patient, Label R, Label L
    img1.nii.gz, patient1, 1, 4
    img2.nii.gz, patient2, 0, 5
    img3.nii.gz, patient3, 2, 3

    The function will predict the target vector for each image and save the prediction in a new column in the csv file.
    Each probability of class is between 0 and 1. The sum is not equal to 1.
    In the example, the sum would give maximum 6.

    The target vector must be splitable in 2 parts (in the middle), one for each class column.
    idx 0 to 2 are for Label R and idx 3 to 5 are for Label L in the example.
    target vector = [0, 1, 0, 0, 1, 0] for the first row.
    '''
    model = NetTarget(seed=args.seed).load_from_checkpoint(args.model)
    model.eval()
    model.cuda()


    if (os.path.splitext(args.csv)[1] == ".csv"):
        df_train = pd.read_csv(args.csv_train)
        df_test = pd.read_csv(args.csv)
    else:
        df_train = pd.read_parquet(args.csv_train)
        df_test = pd.read_parquet(args.csv)

    test_ds = Datasetarget(df_test, mount_point=args.mount_point, img_column=args.img_column, class_column1=args.class_column1, class_column2=args.class_column2,nb_classes=args.nb_classes, transform=EvalTransforms(args.img_size))

    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, prefetch_factor=4)

    with torch.no_grad():
        predictionsR = []
        predictionsL = []
        probs_sig = []
        probs_softmax = []
        features = []

        probR,probL = [],[]
        trueR,trueL = [],[]
        predictions_model = []
        idx_batch = []
        for idx, batch in tqdm(enumerate(test_loader), total=len(test_loader)):
            X, Y = batch
            X = X.cuda().contiguous()


            if args.extract_features:
                pred, x_f = model(X)
                features.append(x_f.cpu().numpy())
            else:
                pred = model(X)
                demi_len = pred.shape[1]//2
                pred_softmaxR = torch.nn.functional.softmax(pred[:,:demi_len], dim=1) #dim 0
                pred_softmaxL = torch.nn.functional.softmax(pred[:,demi_len:], dim=1) #dim 1
                pred_sigmoid = torch.nn.functional.sigmoid(pred)

            for j in range(pred_sigmoid.shape[0]):

                noneR=0
                noneL=0

                # best_probR,idxR = torch.max(pred_sigmoid[j,:demi_len],dim=0)
                # best_probL,idxL = torch.max(pred_sigmoid[j,demi_len:],dim=0)
                best_probR,idxR = torch.max(pred_softmaxR[j,:],dim=0)
                best_probL,idxL = torch.max(pred_softmaxL[j,:],dim=0)

                idxL = idxL.item() + demi_len

                predictionsR.append(idxR.item())
                predictionsL.append(idxL)
                # Code below is used in the case where there is not a class for non impacted canines
                # if best_probR.item() > 0.7:
                #     predictionsR.append(idxR.item())
                # elif best_probR.item() <= 0.8:
                #     predictionsR.append(None)
                #     noneR=1


                # if best_probL.item() > 0.7:
                #     idxL = idxL.item() + demi_len
                #     predictionsL.append(idxL)
                # elif best_probL.item() <= 0.7 :
                #     predictionsL.append(None)
                #     noneL=1

                # if noneR == 1 and noneL == 1:
                #     #get the highest probability and replace the None
                #     # best_prob,idx = torch.max(pred_sigmoid[j,:],dim=0)
                #     if best_probR.item() > best_probL.item():
                #         idx = idxR
                #     else:
                #         idx = idxL

                #     if idx.item() < demi_len:
                #         predictionsR[-1] = idx.item()
                #     else:
                #         predictionsL[-1] = idx.item()


                #find label R and L from Y[j]
                #look for the index of the non zero value
                target_lst = Y[j].tolist()
                idx_non_zero_target = [i for i, val in enumerate(target_lst) if val > 0.0]

                if len(idx_non_zero_target) > 1:
                    #when 2 classes are given, target_vector has 2 values of 0.5 so we need to change the value to 1
                    trueR_value = [value*2 for value in target_lst[:demi_len]]
                    trueR.append(trueR_value)
                    # probR.append(pred_softmax[j,:demi_len].cpu().numpy())
                    probR.append(pred_softmaxR[j,:].cpu().numpy())

                    trueL_value = [value*2 for value in target_lst[demi_len:]]
                    trueL.append(trueL_value)
                    # probL.append(pred_softmax[j,demi_len:].cpu().numpy())
                    probL.append(pred_softmaxL[j,:].cpu().numpy())
                else:
                    if idx_non_zero_target[0] < demi_len:
                        trueR.append(target_lst[:demi_len])
                        # probR.append(pred_softmax[j,:demi_len].cpu().numpy())
                        probR.append(pred_softmaxR[j,:].cpu().numpy())
                    else:
                        trueL.append(target_lst[demi_len:])
                        # probL.append(pred_softmax[j,demi_len:].cpu().numpy())
                        probL.append(pred_softmaxL[j,:].cpu().numpy())

                probs_sig.append(pred_sigmoid[j,:].cpu().numpy())
                # probs_softmax.append(pred_softmax[j,:].cpu().numpy())
                concat_softmax = torch.cat([pred_softmaxR[j,:],pred_softmaxL[j,:]],dim=0)
                probs_softmax.append(concat_softmax.cpu().numpy())

                predictions_model.append(pred[j,:].cpu().numpy())



    print('args.class_column1',args.class_column1)
    if '_' in args.class_column1:
        predR_column = args.pred_column+'_R'
        predL_column = args.pred_column+'_L'
    else:
        predR_column = args.pred_column + ' R'
        predL_column = args.pred_column + ' L'
    df_test[predR_column] = predictionsR
    df_test[predL_column] = predictionsL
    output_dir =args.out


    df_test['Prob sigmoid'] = probs_sig
    df_test['Prob softmax'] = probs_softmax

    filename = os.path.basename(args.csv).replace('.csv', '_prediction.csv')
    df_test.to_csv(os.path.join(output_dir, filename), index=False)
    # create csv with the predictions target before sigmoid
    prob_filenameCsv = os.path.basename(args.csv).replace('.csv', '_prob.csv')
    prob_filenamePickle = os.path.basename(args.csv).replace('.csv', '_prob-pred.pickle')
    df_pred = pd.DataFrame()
    df_pred['Before Sigmoid'] = predictions_model
    df_pred.to_pickle(os.path.join(output_dir, prob_filenamePickle))
    df_pred['Sigmoid'] = probs_sig
    df_pred.to_csv(os.path.join(output_dir, prob_filenameCsv))


    ## Compute AUC and ROC AUC curve and save it
    auc_fn = "auc_evaluation.csv"
    auc_path = os.path.join(output_dir, "AUC/")
    if not os.path.exists(auc_path):
        os.makedirs(auc_path)
    auc_path = os.path.join(auc_path, auc_fn)

    roc_curve_fn = "roc_curve.png"
    roc_curve_path = os.path.join(output_dir, "AUC/")
    if not os.path.exists(roc_curve_path):
        os.makedirs(roc_curve_path)
    roc_curve_path = os.path.join(roc_curve_path, roc_curve_fn)

    auc_data_lst = []
    probR,trueR = np.array(probR), np.array(trueR)
    probL, trueL = np.array(probL), np.array(trueL)
    aucR_tot =compute_roc_auc(torch.tensor(probR), torch.tensor(trueR))
    aucL_tot =compute_roc_auc(torch.tensor(probL), torch.tensor(trueL))
    if aucR_tot>0.6:
        printGreen(f'AUC R: {aucR_tot}')
    else:
        printRed(f'AUC R: {aucR_tot}')
    if aucL_tot>0.6:
        printGreen(f'AUC L: {aucL_tot}')
    else:
        printRed(f'AUC L: {aucL_tot}')

    #Compute AUC for each class
    first_column = []
    for i in range(args.nb_classes):
        if i==demi_len:
            first_column.append('auc 1')
            auc_data_lst.append(aucR_tot)
            first_column.append(i)
        else:
            first_column.append(i)
        if i< demi_len:
            aucR =compute_roc_auc(torch.tensor(probR)[:,i], torch.tensor(trueR)[:,i])
            auc_data_lst.append(aucR)

            if i< demi_len-1:
                plt.figure(1)
                add_roc_curve(torch.tensor(probR)[:,i], torch.tensor(trueR)[:,i], i)
            else:
                right_curve_path = roc_curve_path.replace('.png','_R.png')
                plot_roc_curve_final(torch.tensor(probR)[:,i], torch.tensor(trueR)[:,i], i,right_curve_path)
        else:
            aucL =compute_roc_auc(torch.tensor(probL)[:,i-demi_len], torch.tensor(trueL)[:,i-demi_len])
            auc_data_lst.append(aucL)
            if i< args.nb_classes-1:
                plt.figure(2)
                add_roc_curve(torch.tensor(probL)[:,i-demi_len], torch.tensor(trueL)[:,i-demi_len], i)
            else:
                left_curve_path = roc_curve_path.replace('.png','_L.png')
                plot_roc_curve_final(torch.tensor(probL)[:,i-demi_len], torch.tensor(trueL)[:,i-demi_len], i,left_curve_path)

    first_column.append('auc 2')
    auc_data_lst.append(aucL_tot)

    df_auc = pd.DataFrame({'Class':first_column,"AUC": auc_data_lst})
    df_auc.to_csv(auc_path, index=False)



def NormalPred(args):
    '''
    Prediction function for a single class column.
    Sum of probabilities of each class is equal to 1 (Use Softmax activation function).
    '''

    if args.seg_column is None:
        model = Net(seed=args.seed).load_from_checkpoint(args.model)


    model.eval()
    model.cuda()

    if(os.path.splitext(args.csv)[1] == ".csv"):
        df_train = pd.read_csv(args.csv_train)
        df_test = pd.read_csv(args.csv)
    else:
        df_train = pd.read_parquet(args.csv_train)
        df_test = pd.read_parquet(args.csv)

    use_class_column = False
    if args.class_column is not None and args.class_column in df_test.columns:
        use_class_column = True

    if use_class_column:
        df_filtered_train = df_train.dropna(subset=[args.class_column])
        df_filtered_train = df_filtered_train.reset_index(drop=True)
        unique_classes = np.unique(df_filtered_train[args.class_column])

        unique_class_weights = np.array(class_weight.compute_class_weight(class_weight='balanced', classes=unique_classes, y=df_filtered_train[args.class_column]))

        # unique_classes = np.sort(np.unique(df_train[args.class_column]))
        # unique_class_weights = np.array(class_weight.compute_class_weight(class_weight='balanced', classes=unique_classes, y=df_train[args.class_column]))
        class_replace = {}
        for cn, cl in enumerate(unique_classes):
            class_replace[cl] = cn
        print(unique_classes, unique_class_weights, class_replace)

        df_test[args.class_column] = df_test[args.class_column].replace(class_replace)

        if args.seg_column is None:
            test_ds = BasicDataset(df_test, img_column=args.img_column, mount_point=args.mount_point, class_column=args.class_column, transform=EvalTransforms(args.img_size))

    else:
        test_ds = BasicDataset(df_test, img_column=args.img_column, mount_point=args.mount_point, transform=EvalTransforms(args.img_size))

    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, prefetch_factor=4)

    with torch.no_grad():

        predictions = []
        probs = []
        features = []
        for idx, X in tqdm(enumerate(test_loader), total=len(test_loader)):
            if use_class_column and args.seg_column is None:
                X, Y = X
            elif use_class_column and args.seg_column is not None:
                X0, X1, Y = X
                X = torch.cat([X0, X1], dim=1)
            X = X.cuda().contiguous()
            if args.extract_features:
                pred, x_f = model(X)
                features.append(x_f.cpu().numpy())
            else:
                pred = model(X)
                pred_prob = torch.nn.functional.softmax(pred, dim=1)

            probs.append(pred_prob.cpu().numpy())
            predictions.append(torch.argmax(pred, dim=1).cpu().numpy())

    df_test[args.pred_column] = np.concatenate(predictions, axis=0)
    probs = np.concatenate(probs, axis=0)

    if not os.path.exists(args.out):
        os.makedirs(args.out)

    if use_class_column:
        print(classification_report(df_test[args.class_column], df_test[args.pred_column]))

    ext = os.path.splitext(args.csv)[1]
    if(ext == ".csv"):
        df_test.to_csv(os.path.join(args.out, os.path.basename(args.csv).replace(".csv", "_prediction.csv")), index=False)
    else:
        df_test.to_parquet(os.path.join(args.out, os.path.basename(args.csv).replace(".parquet", "_prediction.parquet")), index=False)


    pickle.dump(probs, open(os.path.join(args.out, os.path.basename(args.csv).replace(ext, "_probs.pickle")), 'wb'))

    if len(features) > 0:
        features = np.concatenate(features, axis=0)
        pickle.dump(features, open(os.path.join(args.mount_point, args.out, os.path.basename(args.csv).replace(ext, "_prediction.pickle")), 'wb'))


def main(args):
    if not os.path.exists(os.path.dirname(args.out)):
        os.makedirs(os.path.dirname(args.out))
    if args.mode == 'CV_2pred':
        MultiPred(args)
    elif args.mode == 'CV':
        NormalPred(args)
    elif args.mode =="CV_2fclayer":
        MultiBranchPred(args)

def get_argparse():
    parser = argparse.ArgumentParser(description='Classification predict')
    parser.add_argument('--csv', type=str, help='CSV file for testing', required=True)
    parser.add_argument('--csv_train', type=str, help='CSV file to compute class replace', required=True)
    parser.add_argument('--extract_features', type=bool, help='Extract the features', default=False)
    parser.add_argument('--img_column', type=str, help='Column name in the csv file with image path', default="Path")
    parser.add_argument('--class_column', type=str, help='Column name in the csv file with classes', default="Label")
    parser.add_argument('--seg_column', type=str, help='Column name in the csv file with image segmentation path', default=None)
    parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float, help='Learning rate')
    parser.add_argument('--model', type=str, help='Model path to use for the predictions',required=True,  default='./')
    parser.add_argument('--epochs', help='Max number of epochs', type=int, default=200)
    parser.add_argument('--out', help='Output directory', type=str, default="./")
    parser.add_argument('--pred_column', help='Output column name', type=str, default="pred")
    parser.add_argument('--mount_point', help='Dataset mount directory', type=str, default="./")
    parser.add_argument('--num_workers', help='Number of workers for loading', type=int, default=4)
    parser.add_argument('--batch_size', help='Batch size', type=int, default=4)
    parser.add_argument('--base_encoder', type=str, default='efficientnet-b0', help='Type of base encoder')
    parser.add_argument('--img_size', help='Image size of the dataset', type=int, default=224)

    parser.add_argument('--seed', help='Seed for reproducibility', type=int, default=42)

    parser.add_argument('--mode', type=str, help='Mode used for the model', default='CV', choices=['CV', 'CV_2pred', 'CV_2fclayer'])
    # Use the next parameters for modes CV_2pred and CV_2fclayer (when 2 columns of predicitons are used in the csv file)
    parser.add_argument('--nb_classes', help='Number of classes', type=int, default=6)
    parser.add_argument('--class_column1', type=str, help='Column name in the csv file with classes', default="Label_R")
    parser.add_argument('--class_column2', type=str, help='Column name in the csv file with classes', default="Label_L")
    parser.add_argument('--diff', nargs='+',help='Differentiator between the 2 Label/predict columns. Ex: Label 1, Label 2 -->  1 2', default=['_R','_L'])



    return parser


if __name__ == '__main__':

    parser = get_argparse()
    args = parser.parse_args()

    if args.model =='./':
        printRed('Please provide a path to a model')
        exit()
    main(args)