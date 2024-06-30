import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

def get_pca_features(features, num_components=3):
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

def get_kmeans_clusters(features, num_clusters=4, img_shape=None):
    '''
    Takes linear features or a feature space and returns the cluster labels of the features.
    INPUT:
        features (np.ndarray): feature space, shape (H, W, F); or linear features, shape (H*W, F)
        num_clusters (int): number of clusters to use
        img_shape (tuple): shape of the image. If given, reshapes the cluster labels to this shape
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
