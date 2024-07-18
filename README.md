# VolumeAXI

## Table of Contents
- [Introduction](#introduction)
- [Where to find the module?](#where-to-find-the-module)
- [Train new models](#train-new-models)
- [How does it work](#how-does-it-work)
- [Experiments & Results](#experiments--results)
- [Explainability](#explainability)
- [Contribute](#contribute)
- [Application](#application)
- [FAQs](#faqs)
- [License](#license)

##Introduction 

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


