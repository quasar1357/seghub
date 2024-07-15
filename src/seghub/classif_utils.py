import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from skimage import filters, morphology

def get_pca_features(features, num_components=2):
    '''
    Takes linear features or a feature space and returns the principal components of the features (normalized to 0-1).
    INPUT:
        features (np.ndarray): feature space, shape (H, W, F); or linear features, shape (H*W, F)
        num_components (int): number of principal components to extract
    OUTPUT:
        pc_normalized (np.ndarray): normalized principal components. Shape (H, W), or (H*W) accoring to input features
    '''
    if len(features.shape) > 2:
        # features = np.moveaxis(features, -1, 0)
        features_linear = features.reshape(-1, features.shape[-1])
    else:
        features_linear = features
    pca = PCA(n_components=num_components)
    pc = pca.fit_transform(features_linear)
    pc_normalized = (pc - np.min(pc)) / (np.max(pc) - np.min(pc))
    if len(features.shape) > 2:
        pc_normalized = pc_normalized.reshape(features.shape[0], features.shape[1], num_components)
    return pc_normalized

def get_kmeans_clusters(features, num_clusters=3):
    '''
    Takes linear features or a feature space and returns the cluster labels of the features.
    INPUT:
        features (np.ndarray): feature space, shape (H, W, F); or linear features, shape (H*W, F)
        num_clusters (int): number of clusters to use
    OUTPUT:
        cluster_labels (np.ndarray): cluster labels. Shape (H*W), or (H, W) accoring to input features
    '''
    if len(features.shape) > 2:
        # features = np.moveaxis(features, -1, 0)
        features_linear = features.reshape(-1, features.shape[-1])
    else:
        features_linear = features
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(features_linear)
    cluster_labels = kmeans.labels_
    if len(features.shape) > 2:
        cluster_labels = cluster_labels.reshape(features.shape[0], features.shape[1])
    return cluster_labels

def features_combinator(image, features_func_list, features_cfg_list, num_pcs_list=None, feature_smoothness_list=None, img_as_feature=False):
    '''
    Takes an image and a list of feature functions and their configurations, and returns a combined feature space.
    INPUT:
        image (np.ndarray): input image, shape (H, W, C)
        features_func_list (list): list of feature functions to apply
            each function must take an image of shape (H, W, C) or (H, W)
            and return features of shape (H, W, n_features) (feature space)
        features_cfg_list (list): list of configurations for each feature function
        num_pcs_list (list): list of number of principal components to extract for each feature
        feature_smoothness_list (list): list of smoothness parameters for each feature
    OUTPUT:
        features_combined (np.ndarray): combined feature space, shape (H, W, F)
    '''
    features_list = []
    for i, features_func in enumerate(features_func_list):
        feature_space = features_func(image, **features_cfg_list[i])
        # Get principal components if desired
        if num_pcs_list is not None and num_pcs_list[i] > 0:
            feature_space = get_pca_features(feature_space, num_components=num_pcs_list[i])
        # Smoothen feature_space if desired
        if feature_smoothness_list is not None and feature_smoothness_list[i] > 0:
            feature_space = np.moveaxis(feature_space, 2, 0)
            feature_space = np.array([filters.median(f, footprint=morphology.disk(feature_smoothness_list[i])) for f in feature_space])
            feature_space = np.moveaxis(feature_space, 0, 2)
        # Add image pixel values (of each channel) as features
        features_list.append(feature_space)
    features_combined = np.stack(features_list, axis=-1)
    if img_as_feature:
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=2)
        features_combined = np.dstack((features_combined, image))
    return features_combined