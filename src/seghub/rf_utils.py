import numpy as np
from seghub.util_funcs import test_img_labels_batch_shapes, reshape_patches_to_img, calculate_padding, get_features_targets
from seghub.classif_utils import get_pca_features
from sklearn.ensemble import RandomForestClassifier
from time import time
from skimage import filters, morphology

def train_seg_forest(image_batch, labels_batch, features_func, features_cfg={}, print_steps=False, random_state=0,
                     pcs_as_features=False, feature_smoothness=False, img_as_feature=False):
    '''
    Takes an image batch and a label batch, extracts features using the given function, and trains a random forest classifier.
    INPUT:
        image_batch (list of np.ndarrays or np.ndarray): list/batch of images. Each image has shape (H, W, C) or (H, W)
        labels_batch (list of np.ndarrays or np.ndarray): list/batch of labels. Each label has shape (H, W)
        features_func (function): function to extract features from an image
            must take an image of shape (H, W, C) or (H, W)
            and return features of shape (H, W, n_features) (feature space)
        features_cfg (dict): configuration for the feature extraction function (parametes of the function given as key-value pairs)
        print_steps (bool): whether to print progress
        random_state (int): random state for the random forest classifier
        pcs_as_features (bool or int): number of principal components to use as features (or False to turn off)
        feature_smoothness (bool or int): size of the median filter footprint for smoothening the features (or False to turn off)
        img_as_feature (bool): whether to add the original image as a feature (each channel is added as a feature)
    OUTPUT:
        random_forest (RandomForestClassifier): trained random forest classifier
    '''
    # Check if the image and label batch have the correct shapes
    test_img_labels_batch_shapes(image_batch, labels_batch)
    # Extract features and targets for the entire batch (but only of the annotated images and pixels)
    features_list = []
    targets_list = []
    i = 0
    num_labelled = sum([np.any(labels) for labels in labels_batch])
    t_start = time()
    # Iterate over the images and their labels and extract features and targets for each annotated pixel
    for image, labels in zip(image_batch, labels_batch):
        if np.all(labels == 0):
            continue
        if print_steps:
            est_t = f"{((time()-t_start)/(i))*(num_labelled-i):.1f} seconds" if i > 0 else "NA"
            print(f'Extracting features for labels {i+1}/{num_labelled} - estimated time left: {est_t}')
            i += 1
        feature_space = features_func(image, **features_cfg)
        # Get principal components as features if desired
        if pcs_as_features:
            feature_space = get_pca_features(feature_space, num_components=pcs_as_features)
        # Smoothen the features by applying a median filter if desired
        if feature_smoothness:
            feature_space = np.moveaxis(feature_space, 2, 0)
            feature_space = np.array([filters.median(f, footprint=morphology.disk(feature_smoothness)) for f in feature_space])
            feature_space = np.moveaxis(feature_space, 0, 2)
        # Add original image as feature if desired
        if img_as_feature:
            if len(image.shape) == 2:
                image = np.expand_dims(image, axis=2)
            feature_space = np.dstack((feature_space, image))
        # Get features and targets of annotated pixels
        features_annot, targets = get_features_targets(feature_space, labels)
        features_list.append(features_annot)
        targets_list.append(targets)
    features_annot = np.concatenate(features_list)
    targets = np.concatenate(targets_list)
    # Train the random forest classifier
    random_forest = RandomForestClassifier(n_estimators=100, random_state=random_state)
    random_forest.fit(features_annot, targets)
    return random_forest

def predict_seg_forest_single_image(image, random_forest, features_func, features_cfg={}, pred_per_patch=False, patch_size=(14,14),
                                    pcs_as_features=False, feature_smoothness=False, img_as_feature=False, pred_smoothness=False):
    '''
    Takes an image and a trained random forest classifier, extracts features using the given function, and predicts labels.
    INPUT:
        image (np.ndarray): image to predict on. Shape (H, W, C) or (H, W)
        random_forest (RandomForestClassifier): trained random forest classifier
        features_func (function): function to extract features from an image
            must take an image of shape (H, W, C) or (H, W)
            and return features of shape (H, W, n_features) --> pred_per_patch is False
                                      OR (H*W, n_features) --> pred_per_patch is False
                                      OR (Hp*Wp, n_features) --> pred_per_patch must be set to True
        features_cfg (dict): configuration for the feature extraction function (parametes of the function given as key-value pairs)
        pred_per_patch (int): whether to predict per patch or per pixel
            If True, features_func must return per patch features (and patch_size must be given)
            If False, features_func must return per pixel features
        patch_size (tuple of int): size of the patches
        pcs_as_features (bool or int): number of principal components to use as features (or False to turn off)
        feature_smoothness (bool or int): size of the median filter footprint for smoothening the features (or False to turn off)
        img_as_feature (bool): whether to add the original image as a feature (each channel is added as a feature)
        pred_smoothness (bool or int): size of the majority filter footprint for smoothening the prediction (or False to turn off)
    OUTPUT:
        pred_img (np.ndarray): predicted labels. Shape (H, W)
    '''
    features = features_func(image, **features_cfg)
    # Get principal components as features if desired
    if pcs_as_features:
        features = get_pca_features(features, num_components=pcs_as_features)
    # Smoothen the features by applying a median filter if desired
    if feature_smoothness:
        if len(features.shape) < 3:
            raise ValueError("Smoothening of features only possible if extracted as feature space.")
        else:
            features = np.moveaxis(features, 2, 0)
            features = np.array([filters.median(f, footprint=morphology.disk(feature_smoothness)) for f in features])
            features = np.moveaxis(features, 0, 2)
    # Add original image as feature if desired
    if img_as_feature:
        if len(features.shape) < 3:
            raise ValueError("Adding image as feature only possible if features are extracted as feature space.")
        else:
            if len(image.shape) == 2:
                image = np.expand_dims(image, axis=2)
            features = np.dstack((features, image))
    # If features are given as a feature space, we reshape them to flat features
    if len(features.shape) > 2:
        num_features = features.shape[2]
        features = np.reshape(features, (image.shape[0]*image.shape[1], num_features))
    predicted_labels = random_forest.predict(features)
    # If we have no patch resizing, the features are already per pixel
    if not pred_per_patch:
        pred_img = np.reshape(predicted_labels, image.shape[:2])
    # Otherwise, we reshape the predicted labels to the image size considering the patches
    else:
        vertical_pad, horizontal_pad = calculate_padding(image.shape, patch_size=patch_size)
        padded_img_shape = (image.shape[0] + vertical_pad, image.shape[1] + horizontal_pad)
        pred_img = reshape_patches_to_img(predicted_labels, padded_img_shape, interpolation_order=0)
        # Crop away the padding back to the original image size
        pred_img = pred_img[:image.shape[0], :image.shape[1]]
    # Smoothen the prediction by applying a majority filter if desired
    if pred_smoothness:
        pred_img = filters.rank.majority(pred_img, footprint=morphology.disk(pred_smoothness))
    return pred_img

def predict_seg_forest(img_batch, random_forest, features_func, features_cfg={}, pred_per_patch=False, patch_size=(14,14), print_steps=False,
                       pcs_as_features=False, feature_smoothness=False, img_as_feature=False, pred_smoothness=False):
    '''
    Takes an image batch and a trained random forest classifier, extracts features using the given function, and predicts labels for all images.
    INPUT:
        img_batch (list of np.ndarrays or np.ndarray): list/batch of images to predict on. Each image has shape (H, W, C) or (H, W)
        random_forest (RandomForestClassifier): trained random forest classifier
        features_func (function): function to extract features from an image
            must take an image of shape (H, W, C) or (H, W)
            and return features of shape (H * W, n_features) --> pred_per_patch is False
                                      OR (H, W, n_features) --> pred_per_patch is False
                                      OR (Hp*Wp, n_features) --> pred_per_patch is True
        features_cfg (dict): configuration for the feature extraction function (parametes of the function given as key-value pairs)
        pred_per_patch (int): whether to predict per patch or per pixel
            If True, features_func must return per patch features (and patch_size must be given)
            If False, features_func must return per pixel features
        patch_size (tuple of int): size of the patches; must be given if pred_per_patch == True
        print_steps (bool): whether to print progress
        pcs_as_features (bool or int): number of principal components to use as features (or False to turn off)
        feature_smoothness (bool or int): size of the median filter footprint for smoothening the features (or False to turn off)
        img_as_feature (bool): whether to add the original image as a feature (each channel is added as a feature)
        pred_smoothness (bool or int): size of the majority filter footprint for smoothening the prediction (or False to turn off)
    OUTPUT:
        pred_batch (np.ndarray): predicted labels. Shape (N, H, W)
    '''
    if type(img_batch) is list:
        img_batch = np.array(img_batch)
    pred_batch = np.zeros(img_batch.shape[:3], dtype=np.uint8)
    t_start = time()
    for i, image in enumerate(img_batch):
        if print_steps:
            est_t = f"{((time()-t_start)/(i))*(len(img_batch)-i):.1f} seconds" if i > 0 else "NA"
            print(f'Predicting image {i+1}/{len(img_batch)} - estimated time left: {est_t}')
        pred_batch[i] = (predict_seg_forest_single_image(image, random_forest, features_func, features_cfg, pred_per_patch=pred_per_patch, patch_size=patch_size,
                                                         pcs_as_features=pcs_as_features, feature_smoothness=feature_smoothness, img_as_feature=img_as_feature, pred_smoothness=pred_smoothness))
    return pred_batch

def selfpredict_seg_forest_single_image(image, labels, features_func, features_cfg={}, random_state=0,
                                        pcs_as_features=False, feature_smoothness=False, img_as_feature=False, pred_smoothness=False):
    '''
    Takes an image and labels, extracts features using the given function, trains a random forest classifier
    based on the labels, and predicts labels for the entire image. Extracts features only once.
    INPUT:
        image (np.ndarray): image to predict on. Shape (H, W, C) or (H, W)
        labels (np.ndarray): labels for the image. Shape (H, W), same dimensions as image
        features_func (function): function to extract features from an image
            must take an image of shape (H, W, C) or (H, W)
            and return features of shape (H, W, n_features) (feature space)
        features_cfg (dict): configuration for the feature extraction function (parametes of the function given as key-value pairs)
        random_state (int): random state for the random forest classifier
        pcs_as_features (bool or int): number of principal components to use as features (or False to turn off)
        feature_smoothness (bool or int): size of the median filter footprint for smoothening the features (or False to turn off)
        img_as_feature (bool): whether to add the original image as a feature (each channel is added as a feature)
        pred_smoothness (bool or int): size of the majority filter footprint for smoothening the prediction (or False to turn off)
    OUTPUT:
        pred_img (np.ndarray): predicted labels. Shape (H, W)
    '''
    # Extract features for the entire image (only need to do this once if selfpredicting)
    feature_space = features_func(image, **features_cfg)
    # Get principal components as features if desired
    if pcs_as_features:
        feature_space = get_pca_features(feature_space, num_components=pcs_as_features)        
    # Smoothen the features by applying a median filter if desired
    if feature_smoothness:
        feature_space = np.moveaxis(feature_space, 2, 0)
        feature_space = np.array([filters.median(f, footprint=morphology.disk(feature_smoothness)) for f in feature_space])
        feature_space = np.moveaxis(feature_space, 0, 2)
    # Add original image as feature if desired
    if img_as_feature:
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=2)
        feature_space = np.dstack((feature_space, image))
    # Get features and targets of annotated pixels
    features, targets = get_features_targets(feature_space, labels)
    # Train the random forest classifier
    random_forest = RandomForestClassifier(n_estimators=100, random_state=random_state)
    random_forest.fit(features, targets)
    # Predict the labels for all pixels in the image
    num_pix = image.shape[0]*image.shape[1]
    num_features = feature_space.shape[2]
    features_flat = np.reshape(feature_space, (num_pix, num_features))
    predicted_labels = random_forest.predict(features_flat)
    # Reshape the predicted labels to the image size (H, W)
    pred_img = np.reshape(predicted_labels, image.shape[:2])
    # Smoothen the prediction by applying a majority filter if desired
    if pred_smoothness:
        pred_img = filters.rank.majority(pred_img, footprint=morphology.disk(pred_smoothness))
    return pred_img

def segment_seg_forest(train_image_batch, labels_batch, pred_image_batch, features_func, features_cfg={}, print_steps=False, random_state=0,
                       pcs_as_features=False, feature_smoothness=False, img_as_feature=False, pred_smoothness=False):
    '''
    Takes an image batch and a label batch, extracts features using the given function, trains a random forest classifier
    based on the labels, and predicts labels for the entire image batch or on a separate image batch if given.
    INPUT:
        train_image_batch (list of np.ndarrays or np.ndarray): list/batch of images. Each image has shape (H, W, C) or (H, W)
        labels_batch (list of np.ndarrays or np.ndarray): list/batch of labels. Each label has shape (H, W)
        pred_image_batch (list of np.ndarrays or np.ndarray): list/batch of images to predict on. Each image has shape (H, W, C) or (H, W)
            if None, the train_image_batch is used for prediction
        features_func (function): function to extract features from an image
            must take an image of shape (H, W, C) or (H, W)
            and return features of shape (H * W, n_features) (feature space)
        features_cfg (dict): configuration for the feature extraction function (parametes of the function given as key-value pairs)
        print_steps (bool): whether to print progress
        random_state (int): random state for the random forest classifier
        pcs_as_features (bool or int): number of principal components to use as features (or False to turn off)
        feature_smoothness (bool or int): size of the median filter footprint for smoothening the features (or False to turn off)
        img_as_feature (bool): whether to add the original image as a feature (each channel is added as a feature)
        pred_smoothness (bool or int): size of the majority filter footprint for smoothening the prediction (or False to turn off)
    OUTPUT:
        pred_batch (np.ndarray): predicted labels. Shape (N, H, W) where N is the number of images in the batch to predict
    '''
    if pred_image_batch is None:
        pred_image_batch = train_image_batch
    rf = train_seg_forest(train_image_batch, labels_batch, features_func=features_func, features_cfg=features_cfg, print_steps=print_steps, random_state=random_state,
                          pcs_as_features=pcs_as_features, feature_smoothness=feature_smoothness, img_as_feature=img_as_feature)
    pred_batch = predict_seg_forest(pred_image_batch, rf, features_func=features_func, features_cfg=features_cfg, print_steps=print_steps,
                                    pcs_as_features=pcs_as_features, feature_smoothness=feature_smoothness, img_as_feature=img_as_feature, pred_smoothness=pred_smoothness)
    return pred_batch