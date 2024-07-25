import itertools
from ilastik.napari.filters import (FilterSet,
                                    Gaussian,
                                    LaplacianOfGaussian,
                                    GaussianGradientMagnitude,
                                    DifferenceOfGaussians,
                                    StructureTensorEigenvalues,
                                    HessianOfGaussianEigenvalues)
import numpy as np
from seghub.util_funcs import get_features_targets

# Define the filter set and scales
FILTER_LIST = (Gaussian,
               LaplacianOfGaussian,
               GaussianGradientMagnitude,
               DifferenceOfGaussians,
               StructureTensorEigenvalues,
               HessianOfGaussianEigenvalues)
SCALE_LIST = (0.3, 0.7, 1.0, 1.6, 3.5, 5.0, 10.0)
# Generate all combinations of FILTER_LIST and SCALE_LIST
ALL_FILTER_SCALING_COMBOS = list(itertools.product(range(len(FILTER_LIST)), range(len(SCALE_LIST))))
# Create a FilterSet with all combinations
FILTERS = tuple(FILTER_LIST[row](SCALE_LIST[col]) for row, col in sorted(ALL_FILTER_SCALING_COMBOS))
FILTER_SET = FilterSet(filters=FILTERS)

def get_ila_feature_space(image, filter_set=FILTER_SET):
    """
    Feature Extraction with Ilastik for single- or multi-channel images.
    INPUT:
        image (np.ndarray): image to predict on; shape (C, H, W) or (H, W, C) or (H, W)
        filter_set (FilterSet from ilastik.napari.filters): filter set to use for feature extraction
    OUTPUT:
        features (np.ndarray): feature map (H, W, F) with F being the number of features per pixel
    """
    # Extract features (depending on the number of channels)
    if image.ndim > 2:
        features = get_ila_features_multichannel(image)
    else:
        features = filter_set.transform(image)
    return features

def get_ila_features_multichannel(image, filter_set=FILTER_SET):
    """
    Feature Extraction with Ilastik for multichannel images.
    Concatenates the feature maps of each channel.
    INPUT:
        image (np.ndarray): image to predict on; shape (C, H, W) or (H, W, C)
        filter_set (FilterSet from ilastik.napari.filters): filter set to use for feature extraction
    OUTPUT:
        features (np.ndarray): feature map (H, W, C) with C being the number of features per pixel
    """
    # Ensure (H, W, C) - expected by Ilastik
    if len(image.shape) == 3 and image.shape[0] < 4:
        image = np.moveaxis(image, 0, -1)
    # Loop over channels, extract features and concatenate them
    for ch_idx in range(image.data.shape[2]):
        channel_feature_map = filter_set.transform(np.asarray(image[:,:,ch_idx]))
        if ch_idx == 0:
            feature_map = channel_feature_map
        else:
            feature_map = np.concatenate((feature_map, channel_feature_map), axis=2)
    return feature_map

def get_ila_features_targets(image, labels, filter_set=FILTER_SET):
    '''
    Takes an image and labels, extracts features using ilastik filter,
    and returns the features of annotated pixels and their targets.
    INPUT:
        image (np.ndarray): image. Shape (H, W, C) or (H, W)
        labels (np.ndarray): labels. Shape (H, W)
    OUTPUT:
        features_annot (np.ndarray): features of annotated pixels. Shape (n_annotated, F)
        targets (np.ndarray): targets of annotated pixels. Shape (n_annotated)
    '''
    feature_space = get_ila_feature_space(image, filter_set=filter_set)
    features_annot, targets = get_features_targets(feature_space, labels)
    return features_annot, targets
