import numpy as np
from sklearn.ensemble import RandomForestClassifier
from skimage import filters, morphology
from time import time
from seghub.util_funcs import test_img_labels_batch_shapes, reshape_patches_to_img, calculate_padding, get_features_targets
from seghub.classif_utils import get_pca_features, get_kmeans_clusters

class SegBox:

    def __init__(self, verbose=True):
        self.extractors = {}
        self.options = {"pcs_as_features":False, "img_as_feature":False, "pred_smoothening":False}
        self.rf = None
        self.verbose = verbose

    def __str__(self):
        out_str = 'Segmentation Box\n================\n'
        out_str += 'OPTIONS:\n'
        out_str += self.get_options_infos()
        out_str += 'FEATURE EXTRACTORS:\n'
        out_str += self.get_extractors_infos()
        return out_str

    def set_options(self, pcs_as_features=False,
                          img_as_feature=False,
                          pred_smoothening=False):
        options = {"pcs_as_features":pcs_as_features,
                   "img_as_feature":img_as_feature,
                   "pred_smoothening":pred_smoothening}
        if not all(self.options[k] == options[k] for k in options.keys()):
            self.options = {"pcs_as_features":pcs_as_features,
                            "img_as_feature":img_as_feature,
                            "pred_smoothening":pred_smoothening}
            if self.rf is not None and self.verbose:
                print('Options changed. Random Forest model has been reset.')
            self.rf = None

    def get_options_infos(self):
        out_str = ''
        for option_name in self.options:
            out_str += f'  {option_name}: {self.options[option_name]}\n'
        return out_str

    def add_extractor(self, extractor_name, extractor_func, extractor_cfg, num_pcs=False, smoothening=False, overwrite=False):
        if extractor_name in self.extractors and not overwrite:
            raise ValueError('Extractor already exists. ' + 
                             'Use overwrite=True to overwrite it.')
        self.extractors[extractor_name] = {'func': extractor_func,
                                           'cfg': extractor_cfg,
                                           'num_pcs': num_pcs,
                                           'smoothening': smoothening}
        if self.rf is not None and self.verbose:
            print('Added new feature extractor. Random Forest model has been reset.')
        self.rf = None

    def add_extractors(self, extractors_dict):
        for extractor_name in extractors_dict:
            self.add_extractor(extractor_name, extractors_dict[extractor_name]['func'], extractors_dict[extractor_name]['cfg'])

    def set_extractor_func(self, extractor_name, extractor_func):
        if not extractor_name in self.extractors:
            raise ValueError('Extractor not found.')
        self.extractors[extractor_name]['func'] = extractor_func
        if self.rf is not None and self.verbose:
            print('Changed an extractor function. Random Forest model has been reset.')
        self.rf = None

    def set_extractor_cfg(self, extractor_name, extractor_cfg):
        if not extractor_name in self.extractors:
            raise ValueError('Extractor not found.')
        self.extractors[extractor_name]['cfg'] = extractor_cfg
        if self.rf is not None and self.verbose:
            print('Changed an extractor config. Random Forest model has been reset.')
        self.rf = None

    def set_extractor_options(self, extractor_name, num_pcs=0, smoothening=0):
        if not extractor_name in self.extractors:
            raise ValueError('Extractor not found.')
        self.extractors[extractor_name]['num_pcs'] = num_pcs
        self.extractors[extractor_name]['smoothening'] = smoothening
        if self.rf is not None and self.verbose:
            print('Changed extractor options. Random Forest model has been reset.')
        self.rf = None   

    def remove_extractor(self, extractor_name):
        if not extractor_name in self.extractors:
            raise ValueError('Extractor not found.')
        del self.extractors[extractor_name]
        if self.rf is not None and self.verbose:
            print('Removed and extractor. Random Forest model has been reset.')
        self.rf = None   

    def get_extractors_infos(self):
        out_str = ''
        for extractor_name in self.extractors:
            out_str += f'{extractor_name}:\n'
            out_str += f'  Function:    {self.extractors[extractor_name]["func"].__name__}\n'
            out_str += f'  Config:      {self.extractors[extractor_name]["cfg"]}\n'
            out_str += f'  Num PCs:     {self.extractors[extractor_name]["num_pcs"]}\n'
            out_str += f'  Smoothening: {self.extractors[extractor_name]["smoothening"]}\n'
        return out_str

    def extract_features_single_extractor(self, img, extractor_name):
        if not extractor_name in self.extractors:
            raise ValueError('Extractor not found.')
        extr_func = self.extractors[extractor_name]['func']
        extr_cfg = self.extractors[extractor_name]['cfg']
        feature_space = extr_func(img, **extr_cfg)
        # Get principal components if desired
        num_pcs = self.extractors[extractor_name]['num_pcs']
        if num_pcs > 0:
            feature_space = get_pca_features(feature_space, num_pcs)
        # Smoothen features if desired
        smoothening_factor = self.extractors[extractor_name]['smoothening']
        if smoothening_factor > 0:
            feature_space = np.moveaxis(feature_space, 2, 0)
            feature_space = np.array([filters.median(f, footprint=morphology.disk(smoothening_factor)) for f in feature_space])
            feature_space = np.moveaxis(feature_space, 0, 2)
        return feature_space

    def extract_features(self, img):
        features_list = []
        for extractor_name in self.extractors:
            features_list.append(self.extract_features_single_extractor(img, extractor_name))
        features_combined = np.concatenate(features_list, axis=-1)
        if self.options["img_as_feature"]:
            if len(image.shape) == 2:
                image = np.expand_dims(image, axis=2)
            features_combined = np.dstack((features_combined, image))
        return features_combined

    def rf_train(self, img, labels, n_estimators=100, random_state=None):
        feature_space = self.extract_features(img)
        features_annot, labels = get_features_targets(feature_space, labels)
        self.rf = RandomForestClassifier(n_estimators=n_estimators,
                                                 random_state=random_state)
        self.rf.fit(features_annot, labels)

    def rf_predict(self, img):
        feature_space = self.extract_features(img)
        num_features = feature_space.shape[2]
        num_pix = img.shape[0]*img.shape[1]
        features = np.reshape(feature_space, (num_pix, num_features))
        predicted_labels = self.rf.predict(features)
        pred_img = np.reshape(predicted_labels, img.shape[:2])
        pred_smoothening_factor = self.options["pred_smoothening"]
        if pred_smoothening_factor:
            pred_img = filters.rank.majority(pred_img,
                                             footprint=morphology.disk(pred_smoothening_factor))
        return pred_img

    def rf_selfpredict(self, img, labels, n_estimators=100, random_state=None):
        # Extract features
        feature_space = self.extract_features(img)
        # Train the random forest on the annotated pixels
        features_annot, labels = get_features_targets(feature_space, labels)
        self.rf = RandomForestClassifier(n_estimators=n_estimators,
                                                 random_state=random_state)
        self.rf.fit(features_annot, labels)
        # Predict the labels for all pixels in the image
        num_features = feature_space.shape[2]
        num_pix = img.shape[0]*img.shape[1]
        features = np.reshape(feature_space, (num_pix, num_features))
        predicted_labels = self.rf.predict(features)
        pred_img = np.reshape(predicted_labels, img.shape[:2])
        pred_smoothening_factor = self.options["pred_smoothening"]
        if pred_smoothening_factor:
            pred_img = filters.rank.majority(pred_img,
                                             footprint=morphology.disk(pred_smoothening_factor))
        return pred_img

    def get_kmeans(self, img, num_clusters):
        feature_space = self.extract_features(img)
        clusters = get_kmeans_clusters(feature_space, num_clusters)
        return clusters
    

# ToDo:
#  - Batch training, prediciton and Segmentation
#  - Classifier agnostic implementation
#  - Add more classifiers