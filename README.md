# seg hub

Welcome to seg hub. This is a toolbox for semantic segmentation in python. The main idea is to combine multiple ways of feature extraction with a random forest classifier (and possibly other classification algorithms such as kmeans). The idea of combining a convolutional neural network for feature extraction with a random forest classifier originates from Lucien Hinderling of the pertzlab at the University of Bern, where I did my Master Thesis.

Please note that the intention behind seg hub is not implement in fastest-possible solutions. Much rather it shall be a set of tools and ways to explore and compare different approaches to semantic segmentation.

**If you decide to use this tool or some of the code in any sort of public work, please do contact me and cite this repository. I would be excited.**

## Requirements

You can find the requirements for this package in the requirements.txt file. As usual, you can install it with

    pip install -r requirements.txt

We recommend to install the required packages into a fresh conda environment with a working installation of Python 3.

## Installation
You can install seg hub via pip using

    pip install git+https://github.com/quasar1357/seg_hub.git

After this, you can simply import the functions needed in Python (e.g. `from seghub.dino_utils import get_dinov2_feature_space`).

## Get started
You can either directly use the various functions, e.g. to import per-patch features with the DINOv2 vision transformer use

    from seghub.dino_utils import get_dinov2_patch_features

    patch_features = get_dinov2_patch_features(test_image)

Please read carefully the detailed docstrings to get more information about the functions.

## Scripts
- classif_utils.py: functions for easy application of utilities that are helpful for classification, such as PCA and K-means.
- convpaint_utils.py: functions for feature extraction using convpaint with the convolutional neural network VGG16.
- dino_utils.py: functions for feature extraction using DINOv2, a state-of-the-art vision transformer model.
- ilastik-utils.py: functions for feature extraction using classical filterbanks implemented in the popular segmentation tool "ilastik".
- rf_utils.py: wrapper functions to combine feature extraction with a random forest classifier from sklearn (train and predict models).
- util_funcs.py: various utility functions used in the other scripts.

## Notebooks
The notebook seg_hub_function_tests.ipynb shows examples for using functions from the scripts mentioned above.

## Issues
If you encounter any problems, I am grateful if you file an [issue](https://github.com/quasar1357/seg_hub/issues) along with a detailed description.
