# VolumeAXI

## Table of Contents
- [Introduction](#introduction)
- [Where to find the module?](#where-to-find-the-module)
- [Train new models](#train-new-models)
- [Experiments & Results](#experiments--results)
- [Explainability](#explainability)
- [Contribute](#contribute)
- [License](#license)

## Introduction 

**VolumeAXI** aims to develop interpretable deep learning models for the automated classification of impacted maxillary canines and assessment of dental root resorption in adjacent teeth using Cone-Beam Computed Tomography (CBCT). 
We propose to develop a 3D slicer module, called Volume Analysis, eXplainability and Interpretability (Volume-AXI), with the goal of providing users an explainable approach for classification of bone and teeth structural defects in CBCT scans gray-level images. Visualization through Gradient-weighted Class Activation Mapping (Grad-CAM) has been integrated to generate explanations of the CNN predictions, enhancing interpretability and trustworthiness for clinical adoption.

## Where to find the module?
**VolumeAXI** model has been deployed in the open-source software 3D Slicer.

It is available in the extension *Automated Dental Tools*.
Installation steps:
1. Install the last stable or nightly version of 3D Slicer (module available from version 5.6.2).
2. Use the Extension Manager to search for *Automated Dental Tools*. [How to install extensions in 3D slicer](https://slicer.readthedocs.io/en/latest/user_guide/extensions_manager.html)
3. Restart the software as requested.

Bravo! You can now look for it in the Module Selection Tool bar.

## Train New Models
### Prerequisites
Python version 3.12.2

Main packages and their versions (a Yaml file is available to recreate the environment with Conda):

>pytorch-lightning==1.9.5\
>torch==2.2.2\
>torchaudio==2.2.2\
>torchmetrics==1.3.2\
>torchvision==0.17.2\
> \
>numpy==1.26.4\
>nibabel==5.2.1\
>matplotlib.pyplot==3.8.3\
> \
>scikit-learn==1.4.2\
>simpleitk==2.3.1

### Dataset preparation
To train a new model, you need a CSV file with the path of the images and the column with the labels.

There are several network options. Two of them 'CV_2pred' and 'CV_2fclayer' need 2 columns with the label.
In the folder 'Preprocess', you will find most scripts that were used to preprocess the data.

Our method follows this pipeline:
1. Applying the mask to the CBCT
   
```bash
python3 create_CBCTmask.py --dir Path/To/The/Scans/Folder-or-File --mask Path/to/MasksToApply --label 1 2 3 --output Output/path
```

 you can add --dilatation_radius, if you need to dilate the mask with a box-shaped structure.

2. Resample to the willing size/ spacing
```bash
python3 resample.py --dir --size 224 224 224 --spacing 0.3 0.3 0.3 --out Output/path

transformations parameters:
--linear True/False to use linear interpolation
--center True/False to center the image in the space
--fit_spacing True/False to recompute the spacing according to the new image size
--iso_spacing True/False to keep the same spacing for all the images
```

3. Create the CSV file

```bash
python3 create_CSV_input.py --input_folder  --output Path/to/The/CSV/File --label_file Path/to/xlsx-or-csv --patient_column <column with the names to match the files names> --label_column Label
```

In our cases, we had some letter that we could make match with --words_list ['_L','_R'] and --side.

4. Change the label identification or just count the number of them.

```bash
python3 dataset_info.py --input Path/CSV/File --class_column Label
```

5. Split the dataset into training and testing (if willing)

```bash
python3 split_dataset.py --input Csv/File --out_dir Path/to/folder --test_size 0.2 --val_size 0.15 --class_column Label
```

2 options to split the dataset are available with `--split_option <'TT' or 'TTV'>`. 'TT' stands for Train, Test if you only need a splitting between both (used for our training).
'TTV' splits into training, testing and validation. It just another option unused here.

### Training

There are 3 options to train a model.
|Modes | Descriptions |
|------|-------------|
| 'CV' | Csv file has only one column with the labels. Use MONAI architectures.|
|'CV_2pred'| Csv file has 2 labels columns (one for each side Left and Right). Use MONAI architectures.|
|'CV_2fclayer'| Csv file has 2 labels columns. Use MONAI archiectures + 2 fully connected layers (one for each side Left and Right).|

For 'CV_2pred' and 'CV_2fclayer', you choose which column you pass to the ```--class_column``` parameter, it will be used to split the dataset in the Cross-Validation.

Some MONAI architectures are already implemented (base_encoder): DenseNet, DenseNet169, DenseNet201, DenseNet264, SEResNet50, ResNet18 and the EfficientNetBN.

```bash
python3 classification_train_v2.py --csv <csv file path> --img_column Path --class_column Label --nb_classes 8
--base_encoder DenseNet201 --lr 1e-4 --epochs 400 --out <output folder> --patience 50 --img_size 224 --mode CV_2pred

Cross-Validation parameters:
--split 5
--test_size 0.15
--val_size 0.15

Optionnal parameter:
--csv_special path/to/specialDataset to add a dataset that need other transformations to the training
```

### Predictions

```bash
python3 classification_predict.py --csv <csv file to predict> --csv_train <csv training file>
--img_column Path --class_column Label --out <output directory> --pred_column Predictions --base_encoder DenseNet201 --mode 'CV'

2 Labels columns parameters:
--nb_classes 8
--class_column1 Label_R
--class_column2 Label_L
--diff ['_R','_L'] to create the predictions column with the same differentiator than initially used.
```

If you choose the mode 'CV_2pred', it will compute the AUC for the classes at this step.
### Evaluation

```bash
python3 classification_eval_VAXI.py --csv <csv prediction file> --csv_true_column Label --csv_prediction_column Predictions
--out <path to the plot file>  --mode CV

2 Labels columns parameters:
--csv_true_column and --csv_prediction_column must have the common denominator of the columns names.
For example, if the csv has Label_R, Label_L, Predictions_R, Predictions_L, the common names are Label and Predictions.

--diff ['_R','_L']
```

### Gradcam

To use the GradCam script, you need to know the layers names of your model. You can use `retrieve_modelsData.py` to do so.
The script must be run for each class by changing `--class_index` and if you have 2 predictions columns, it must be done for each one of them.
The output includes the grey-level scan and a grey-level heatmaps. To see the results, you need to use 3D Slicer to superimpose both image and change the color of the heatmaps to *ColdToHotRainbow* (see [How to Overlay](#how-to-overlay-the-heatmap-with-the-scan))

```bash
python3 gradcam3D_monai.py --csv_test <path to the prediction file> --img_column Path --class_column Label_R --pred_column Predictions_R
--model_path <path to the .ckpt file> --out <output directory> --img_size 224 --nb_class 8 --class_index 1 --base_encoder DenseNet201
--layer_name model.features.denseblock4
```

 In the case where you have 2 fully connected layer, you need to specify a `--side` because the Gradcam function from MONAI doesn't work with multiple outputs models.


## Experiments & Results

Differents architectures have been used. The mode 'CV_2fclayer' with DenseNet201 is the one giving the best Gradcam and metrics so far.

![2FC_DenseNet201](https://github.com/user-attachments/assets/58204eb8-6345-4182-a8ef-b054eee2dfbf)
Fig1: DenseNet201 architecture with the 2 fully connected layer

![DenseNet201_architecture](https://github.com/user-attachments/assets/d653b266-bf4a-4e64-a29b-633f92b4dc07)
Fig2: DenseNet201 architecture

### Buccolingual Position Classification of Impacted Canines

* Classes: Non impacted, Buccal, Bicortical and Palatal
* **Accuracy:** 78%
* **Weight Average F1-score:** 77%

![Ext_TestingFold_CM](https://github.com/user-attachments/assets/67f842e3-ba15-4106-9e02-1f858576c2ad)
Fig3: External testing fold Confusion Matrix normalized

#### Explainability

![GradCAM_4CM](https://github.com/user-attachments/assets/d65d26eb-a9a2-42b2-aef5-af5c534793d1)
Fig4: Heatmaps Overlaid in 3D slicer

#### How to Overlay the Heatmap with the scan?

1. Load your grey-level CBCT scan and the grey-level heatmap.
First you need to change the color of the heatmap:
3. Select the *Volumes* Module.
4. Select the heatmap in *Active Volume* and the *ColdToHotRainbow* option in the *Lookup Table* (see image below)

![Tuto_ChangeColor](https://github.com/user-attachments/assets/8502ff92-5c63-4a48-9b3f-46a5d282cf38)

Time to superimpose! 
5. Click on the pins ![Screenshot from 2024-07-18 15-32-41](https://github.com/user-attachments/assets/2429708b-2780-4ca8-bd27-72473045a9a2) and then the small arrows ![Screenshot from 2024-07-18 15-33-49](https://github.com/user-attachments/assets/9c4f2275-638b-4179-931e-909cd45dd0df) in one of the view (Axial, Coronal or Sagittal).
6. Synchronize all view by clicking on the link icon ![Screenshot from 2024-07-18 15-35-24](https://github.com/user-attachments/assets/78924a58-22dc-414d-9091-925607cfe08b)
7. Select the scan file (named _original), the heatmap and change the percentage of appearance of the top one on the other (see image below)
![Overlay_Tuto](https://github.com/user-attachments/assets/5cff63a0-3143-4368-953f-0dc0047459ad)

Now, enjoy the visualization :)

## Contribute
We welcome community contributions to VolumeAXI. For those keen on enhancing this tool, please adhere to the steps below:

Fork the repository.
Create your feature branch (git checkout -b feature/YourFeature).
Commit your changes (git commit -am 'Add some feature').
Push to the branch (git push origin feature/YourFeature).
Open a pull request.
For a comprehensive understanding of our contribution process, consult our [Contribution Guidelines](path/to/contribution_guidelines.md).

## License

**VolumeAXI** is under the [APACHE 2.0](LICENSE) license.

---

**VolumeAXI Team**: For further details, inquiries, or suggestions, feel free to [contact us](mailto:juan_prieto@med.unc.edu,luciacev@umich.edu).

