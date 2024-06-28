import numpy as np
from napari_convpaint.conv_paint_utils import (Hookmodel, filter_image_multichannels, get_features_current_layers, predict_image)

def get_convpaint_features_targets_model(image, labels, layer_list=[0], scalings=[1,2], model_name="vgg16"):
    '''
    Extract features of annotated pixels (not entire image) from an image using ConvPaint and VGG16 as feature extractor.
    Return the extracted features together with their targets and the model used for feature extraction.
    INPUT:
        image (np.ndarray): image to extract features from. Shape (H, W) or (H, W, C)
        labels (np.ndarray): labels for the image. Shape (H, W), same dimensions as image
        layer_list (list of int): list of layer indices to use for feature extraction with vgg16
        scalings (list of int): list of scalings to use for feature extraction with vgg16
        model (str): model to use for feature extraction
    OUTPUTS:
        features_annot (np.ndarray): extracted features of annotated pixels. Shape (n_annotated, F)
        targets (np.ndarray): targets of annotated pixels. Shape (n_annotated)
        model (Hookmodel): model used for feature extraction
    '''
    if len(image.shape) == 3:
        image = np.moveaxis(image, -1, 0)
    # Define the model
    model = Hookmodel(model_name=model_name)
    # Ensure the layers are given as a list
    if isinstance(layer_list, int):
        layer_list = [layer_list]
    # Read out the layer names
    all_layers = [key for key in model.module_dict.keys()]
    layers = [all_layers[i] for i in layer_list]
    # Register the hooks for the selected layers
    model.register_hooks(selected_layers=layers)
    # Get the features and targets
    features_annot, targets = get_features_current_layers(
        model=model, image=image, annotations=labels, scalings=scalings,
        order=1, use_min_features=False, image_downsample=1)
    # Convert features from df to np.array
    features_annot = features_annot.to_numpy()
    return features_annot, targets, model

def get_convpaint_feature_space(image, layer_list=[0], scalings=[1,2], model_name="vgg16"):
    '''
    Use convpaint to extract features from an image (entire image).
    INPUT:
        image (np.ndarray): image to extract features from. Shape (H, W) or (H, W, C)
        layer_list (list of int): list of layer indices to use for feature extraction with vgg16
        scalings (list of int): list of scalings to use for feature extraction with vgg16
        model (str): model to use for feature extraction
    OUTPUTS:
        features (np.ndarray): extracted features. Shape (H, W, F)
    '''
    if len(image.shape) == 3:
        image = np.moveaxis(image, -1, 0)
    # Define the model
    model = Hookmodel(model_name=model_name)
    # Ensure the layers are given as a list
    if isinstance(layer_list, int):
        layer_list = [layer_list]
    # Read out the layer names
    all_layers = [key for key in model.module_dict.keys()]
    layers = [all_layers[i] for i in layer_list]
    # Register the hooks for the selected layers
    model.register_hooks(selected_layers=layers)
    features = filter_image_multichannels(image, model, scalings=scalings,
                                          order=1, image_downsample=1)
    # Concatenate feature space
    feature_space = np.concatenate(features, axis=1)
    # Remove the first dimension
    feature_space = feature_space[0]
    feature_space = np.moveaxis(feature_space, 0, -1)
    return feature_space