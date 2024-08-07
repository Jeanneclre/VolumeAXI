import numpy as np
import argparse
import os

import json
import sys
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc, roc_auc_score,f1_score,precision_score,recall_score,accuracy_score
from sklearn.metrics import classification_report
import pandas as pd

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import itertools
import pickle

from useful_readibility import printRed, printBlue,printGreen


COLORS={
    'HEADER': '\033[95m',
    'OKBLUE': '\033[94m',
    'OKCYAN': '\033[96m',

    'OKGREEN': '\033[92m',

    'ENDBLUE': '\033[34m',
    'ENDCYAN': '\033[36m',
    'ENDC': '\033[0m',
}
# This file is used to evaluate the results of a classification or segmentation task (after the model has been trained and predictions have been made)

def plot_confusion_matrix(cm, classes,normalize=False,title='Confusion matrix',cmap=plt.cm.Blues):
    #This function prints and plots the confusion matrix. Normalization can be applied by setting `normalize=True`.
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix, avg:", np.trace(cm)/len(classes))
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap, aspect='auto')
    plt.title(title)
    plt.colorbar()
    print('len classes:',len(classes))
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.3f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.tight_layout()

    return cm


def choose_score(args,report):
    if args.eval_metric == 'F1':
      # Calculate F1 score
      weighted_f1_score = report["weighted avg"]["f1-score"]
      # Print or store F1 score
      print(COLORS["OKBLUE"], "Weighted F1 Score:", weighted_f1_score, COLORS["ENDBLUE"])
      return weighted_f1_score

    elif args.eval_metric == 'AUC':
      # Calculate AUC score
      weighted_auc_score = report["weighted avg"]["auc"]
      # Print or store AUC score
      print(COLORS["OKBLUE"], "Weighted AUC Score:", weighted_auc_score, COLORS["ENDBLUE"])
      return weighted_auc_score

    else:
      sys.exit("The value of score is not F1 or AUC. You must specify F1 or AUC.")


#####################################################################################################################################################################################
#                                                                                                                                                                                   #
#                                                                        Classification                                                                                             #
#                                                                                                                                                                                   #
#####################################################################################################################################################################################


def classification_eval(df, args, y_true_arr, y_pred_arr):
    # For the classification, evaluating a classification model, generating classification metrics, creating confusion matrix visualizations
    # It also responsible for plotting ROC curves, aggregating and reporting classification metrics in a structured format
    input_dir = os.path.dirname(args.csv)
    output_dir = os.path.join(args.mount_point, input_dir)
    output_dir= output_dir

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    class_names = pd.unique(df[args.csv_true_column])
    class_names =[int(x) for x in class_names if str(x) != 'nan']
    class_names.sort()
    print("Class names:", class_names)

    for idx, row in df.iterrows():
      y_true_arr.append(row[args.csv_true_column])
      y_pred_arr.append(row[args.csv_prediction_column])

    report = classification_report(y_true_arr, y_pred_arr, output_dict=True, zero_division=1)
    # print(json.dumps(report, indent=2))

    cnf_matrix = confusion_matrix(y_true_arr, y_pred_arr)
    np.set_printoptions(precision=3)

    # Plot non-normalized confusion matrix
    fig = plt.figure(figsize=args.figsize)

    plot_confusion_matrix(cnf_matrix, classes=class_names, title=args.title)

    fn_cf = os.path.splitext(args.out)[0] + "_confusion.png"
    confusion_filename = os.path.join(output_dir,fn_cf)
    fig.savefig(confusion_filename)

    # Plot normalized confusion matrix
    fig2 = plt.figure(figsize=args.figsize)
    cm = plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True, title=args.title + ' - normalized')

    fn =os.path.splitext(args.out)[0] + "_norm_confusion.png"
    norm_confusion_filename = os.path.join(output_dir, fn)
    fig2.savefig(norm_confusion_filename)


    probs_fn = args.csv.replace("_prediction.csv", "_probs.pickle")

    if os.path.exists(probs_fn) and os.path.splitext(probs_fn)[1] == ".pickle":

      with open(probs_fn, 'rb') as f:
        y_scores = pickle.load(f)
        y_scores = y_scores[:,:4]

      y_onehot = pd.get_dummies(y_true_arr)


      # Create an empty figure, and iteratively add new lines
      # every time we compute a new class
      plt.figure(figsize=(8, 6))
      plt.plot([0, 1], [0, 1], linestyle='--', color='gray')

      supports = []
      aucs = []
      for i in range(y_scores.shape[1]):
        y_true = y_onehot.iloc[:, i]
        y_score = y_scores[:, i]

        fpr, tpr= roc_curve(y_true, y_score)[:2]
        auc_score = roc_auc_score(y_true, y_score)
        aucs.append(auc_score)
        #add AUC value to the report
        report_key = list(report.keys())[i]
        support_class = report[report_key].pop("support")
        report[report_key]["auc"] = auc_score
        report[report_key]['accuracy'] = ''

        #moove support after auc column
        report[report_key]["support"] = int(support_class)

        supports.append(report.get(str(i), {}).get("support", 0))


        plt.plot(fpr, tpr, label=f"{y_onehot.columns[i]} (AUC={auc_score:.2f})")

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend()

    fname = os.path.splitext(args.out)[0] + "_roc.png"
    roc_filename = os.path.join(output_dir, fname)
    plt.savefig(roc_filename)
    plt.close()

    support = np.array(supports)
    auc = np.array(aucs)

    if np.sum(support) != 0:
        report["weighted avg"]["auc"] = np.average(auc, weights=support)

    else:
        report["weighted avg"]["auc"] = 0

    df_report = pd.DataFrame(report).transpose()

    df_report.loc['accuracy'] = ''

    df_report.loc['accuracy','accuracy']=report['accuracy']
    df_report.loc['accuracy','support']= df_report.loc['weighted avg','support']

    fn = os.path.splitext(args.out)[0] + "_classification_report.csv"
    report_filename = os.path.join(output_dir, fn)
    df_report.to_csv(report_filename)

    # Extraction of the score (AUC or F1)
    score = choose_score(args,report)
    return score


def ClassificationMultiLabel_eval(df, args, y_true_arr, y_pred_arr):
  '''
  function to evaluate a multi-label column classification model.
  Test file example:
  Path,            Name,             Label1,  Label2, Pred1,  Pred2
  /path/to/image1, image1,           1,       3,      1,      3
  /path/to/image2, image2,           None,       4,   None,   4
  /path/to/image3, image3,           2,       5,      2,      4
  '''
  input_dir = os.path.dirname(args.csv)
  output_dir = os.path.join(args.mount_point, input_dir)
  output_dir= output_dir

  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  if '_' in args.diff[0]:
    column1_nm = args.csv_true_column +  args.diff[0]
    column2_nm = args.csv_true_column +  args.diff[1]

    pred1_nm = args.csv_prediction_column + args.diff[0]
    pred2_nm = args.csv_prediction_column + args.diff[1]

  else:
    column1_nm = args.csv_true_column + ' ' + args.diff[0]
    column2_nm = args.csv_true_column + ' ' + args.diff[1]

    pred1_nm = args.csv_prediction_column + ' ' + args.diff[0]
    pred2_nm = args.csv_prediction_column + ' ' + args.diff[1]


  #concatenate the 2 columns to get the class names
  df_combined = pd.concat([df[column1_nm], df[column2_nm]])
  class_names = pd.unique(df_combined)

  #remove nan
  class_names = [x for x in class_names if str(x) != 'nan']
  class_names.sort()

  #make sure it's integers
  class_names = [int(x) for x in class_names]

  print("Class names:", class_names)
  # Count false predictions (case where the true label is None and the prediction is not None)
  fail_fp_R=0
  fail_fp_L=0
  # Count wrong predictions (case where the true label is not None and the prediction is None)
  fail_wp_R=0
  fail_wp_L=0

  #First we fill the y_true_arr and y_pred_arr with the values of the first column
  for idx,row in df.iterrows():
    if str(row[column1_nm]) != 'nan' and str(row[pred1_nm]) != 'nan':

      y_true_arr.append(str(int(row[column1_nm])))
      y_pred_arr.append(str(int(row[pred1_nm])))
    elif str(row[column1_nm]) !='nan' and str(row[pred1_nm]) == 'nan':
      fail_wp_R+=1
    elif str(row[column1_nm])=='nan' and str(row[pred1_nm]) != 'nan':
      fail_fp_R+=1
    else:
      pass

  #Then we fill the y_true_arr and y_pred_arr with the values of the second column
  for idx,row in df.iterrows():
    if str(row[column2_nm]) != 'nan' and str(row[pred2_nm]) != 'nan':
      y_true_arr.append(str(int(row[column2_nm])))
      y_pred_arr.append(str(int(row[pred2_nm])))
    elif str(row[column2_nm]) !='nan' and str(row[pred2_nm]) == 'nan':
      fail_wp_L+=1
    elif str(row[column2_nm])=='nan' and str(row[pred2_nm]) != 'nan':
      fail_fp_L+=1
    else:
      pass


  report = classification_report(y_true_arr, y_pred_arr, output_dict=True, zero_division=1)

  cnf_matrix = confusion_matrix(y_true_arr, y_pred_arr)

  np.set_printoptions(precision=3)

  # Plot non-normalized confusion matrix
  fig = plt.figure(figsize=args.figsize)
  plot_confusion_matrix(cnf_matrix, classes=class_names, title=args.title)

  fn_cf = os.path.splitext(args.out)[0] + "_confusion.png"
  confusion_filename = os.path.join(output_dir,fn_cf)
  #add legend with the number of failed predictions
  fig.text(0.25, 0.01, f'Predicted ghost: {fail_fp_R} ({args.diff[0]}), {fail_fp_L} ({args.diff[1]}), Missed Prediction: {fail_wp_R} ({args.diff[0]}), {fail_wp_L} ({args.diff[1]})', ha='center', va='center', color='red')
  fig.savefig(confusion_filename)

  # Plot normalized confusion matrix
  fig2 = plt.figure(figsize=args.figsize)
  cm = plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True, title=args.title + ' - normalized')

  fn =os.path.splitext(args.out)[0] + "_norm_confusion.png"
  norm_confusion_filename = os.path.join(output_dir, fn)
  print('norm_confusion_filename',norm_confusion_filename )
  fig2.text(0.25, 0.01, f'Predicted ghost: {fail_fp_R} ({args.diff[0]}), {fail_fp_L} ({args.diff[1]}), Missed Prediction: {fail_wp_R} ({args.diff[0]}), {fail_wp_L} ({args.diff[1]})', ha='center', va='center',color='red')
  fig2.savefig(norm_confusion_filename)

  # save report to csv

  df_report = pd.DataFrame(report).transpose()
  # if 'accuracy'
  if 'accuracy' in df_report.columns:
    df_report.loc['accuracy'] = ''

    df_report.loc['accuracy','accuracy']=report['accuracy']
    df_report.loc['accuracy','support']= df_report.loc['weighted avg','support']

  fn = os.path.splitext(args.out)[0] + "_classification_report.csv"
  report_filename = os.path.join(output_dir, fn)
  df_report.to_csv(report_filename)

  args.eval_metric = 'F1'
  score = choose_score(args,report)
  return score


def main(args):
    y_true_arr = []
    y_pred_arr = []

    path_to_csv = os.path.join(args.mount_point, args.csv)
    if(os.path.splitext(args.csv)[1] == ".csv"):
        df = pd.read_csv(path_to_csv)
    else:
        df = pd.read_parquet(path_to_csv)

    if args.mode == 'CV':
      score = classification_eval(df, args, y_true_arr, y_pred_arr)
      pass
    elif args.mode == 'CV_2pred':
      score = ClassificationMultiLabel_eval(df, args, y_true_arr, y_pred_arr)

    return score



def get_argparse():
  # Function to parse arguments for the evaluation script
  parser = argparse.ArgumentParser(description='Evaluate classification result', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument('--csv', type=str, help='CSV file', required=True)
  parser.add_argument('--csv_true_column', type=str, help='Which column to do the stats on, if Multi like Label L and Label R, write Label', default="class")
  parser.add_argument('--csv_prediction_column', type=str, help='csv prediction class, if Multi write common word', default='pred')

  parser.add_argument('--title', type=str, help='Title for the image', default='Confusion matrix')
  parser.add_argument('--figsize', type=str, nargs='+', help='Figure size', default=(8, 8))
  parser.add_argument('--eval_metric', type=str, help='Score you want to choose for picking the best model : F1 or AUC', default='F1', choices=['F1', 'AUC'])
  parser.add_argument('--mount_point', type=str, help='Mount point for the data', default='./')

  parser.add_argument('--out', type=str, help='Output filename for the plot', default="Final_evaluation.png")

  parser.add_argument('--mode', type=str, help='Mode of the evaluation', default='CV', choices=['CV', 'CV_2pred'])
  # For MultiLabel evaluation
  parser.add_argument('--diff',nargs='+', help='Differentiator between the 2 Label/predict columns. Ex: Label 1, Label 2 -->  1 2', default=['_R','_L'])




  return parser


if __name__ == "__main__":
  parser = get_argparse()
  args = parser.parse_args()

  main(args)



