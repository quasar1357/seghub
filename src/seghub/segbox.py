from time import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from skimage import filters, morphology
import joblib
import yaml

from seghub.util_funcs import test_img_labels_batch_shapes, get_features_targets
from seghub.classif_utils import get_pca_features, get_kmeans_clusters

class SegBox:
    '''
    SegBox is a modular toolbox for image segmentation using different ways of feature extraction
    combined with fast and efficient classification alforithms such as Random Forests and KMeans.
    '''

    def __init__(self, verbose=True):
        self.options = {"PCs as features":False, "IMG as feature":False, "Smoothen preds":False}
        self.extractors = {}
        self.random_forest = None
        self.rf_settings = {"Nr. estimators":100, "Random state":None}
        self.prediction_history = []
        self.adjusted = False
        self.verbose = verbose

    def __str__(self):
        out_str = 'Segmentation Box\n================\n'
        out_str += 'OPTIONS:\n'
        out_str += self.get_options_infos()
        out_str += 'RANDOM FOREST SETTINGS:\n'
        out_str += self.get_rf_settings()
        out_str += 'FEATURE EXTRACTORS:\n'
        out_str += self.get_extractors_infos()
        return out_str

    def set_options(self, pcs_as_features=False,
                          img_as_feature=False,
                          pred_smoothening=False):
        '''
        Set general options for the SegBox,
        such as using principal components as features or smoothening predictions.
        '''
        options = {"PCs as features": pcs_as_features,
                   "IMG as feature": img_as_feature,
                   "Smoothen preds": pred_smoothening}
        if all(self.options[k] == options[k] for k in options.keys()):
            return
        self.options = {"PCs as features": pcs_as_features,
                        "IMG as feature": img_as_feature,
                        "Smoothen preds": pred_smoothening}
        if self.random_forest is not None:
            self.adjusted = True
            if self.verbose:
                print('Warning, options changed. ' +
                      'Random Forest model might not be compatible anymore.')

    def get_options_infos(self):
        '''
        Return a string with information about the options set for the SegBox.
        '''
        out_str = ''
        max_len = max(len(option_name) for option_name in self.options)
        for option_name in self.options:
            pad = (max_len-len(option_name))*" "
            out_str += f'  {option_name}:{pad} {self.options[option_name]}\n'
        return out_str

    def set_rf_settings(self, n_estimators=100, random_state=None):
        '''
        Change settings for the Random Forest model,
        such as the number of estimators and the random state.
        '''
        settings = {"Nr. estimators": n_estimators,
                    "Random state": random_state}
        if all(self.rf_settings[k] == settings[k] for k in settings.keys()):
            return
        self.rf_settings = {"Nr. estimators": n_estimators,
                            "Random state": random_state}
        if self.random_forest is not None:
            self.adjusted = True
            if self.verbose:
                print('Warning, Random Forest settings changed. ' +
                      'Random Forest model might not be compatible anymore.')

    def get_rf_settings(self):
        '''
        Return a string with information about the Random Forest settings.
        '''
        out_str = ''
        max_len = max(len(setting_name) for setting_name in self.rf_settings)
        for setting_name in self.rf_settings:
            pad = (max_len-len(setting_name))*" "
            out_str += f'  {setting_name}:{pad} {self.rf_settings[setting_name]}\n'
        return out_str

    def add_extractor(self, extractor_name, extractor_func, extractor_cfg,
                      num_pcs=False, smoothening=False, overwrite=False):
        '''
        Add an extractor to the list of extractors giving its name, function and configuration.
        '''
        if extractor_name in self.extractors and not overwrite:
            raise ValueError('Extractor already exists. ' +
                             'Use overwrite=True to overwrite it.')
        self.extractors[extractor_name] = {'func': extractor_func,
                                           'cfg': extractor_cfg,
                                           'num_pcs': num_pcs,
                                           'smoothening': smoothening}
        if self.random_forest is not None:
            self.adjusted = True
            if self.verbose:
                print('Warning, added new feature extractor. ' +
                      'Random Forest model might not be compatible anymore.')

    def add_extractors(self, extractors_dict):
        '''
        Add multiple extractors at once using a dictionary with the following structure:
        {'extractor_name1': {'func': extractor_func1, 'cfg': extractor_cfg1},
         'extractor_name2': {'func': extractor_func2, 'cfg': extractor_cfg2},
         ...}
        '''
        for extractor_name in extractors_dict:
            extr_in = extractors_dict[extractor_name]
            if not 'num_pcs' in extr_in:
                extr_in['num_pcs'] = False
            if not 'smoothening' in extr_in:
                extr_in['smoothening'] = False
            self.add_extractor(extractor_name, extr_in['func'],
                                               extr_in['cfg'],
                                               extr_in['num_pcs'],
                                               extr_in['smoothening'])

    def set_extractor_func(self, extractor_name, extractor_func):
        '''
        Change the function used by an extractor that has already been registered.
        '''
        if not extractor_name in self.extractors:
            raise ValueError('Extractor not found.')
        self.extractors[extractor_name]['func'] = extractor_func
        if self.random_forest is not None:
            self.adjusted = True
            if self.verbose:
                print('Warning, changed an extractor function. ' +
                      'Random Forest model might not be compatible anymore.')

    def set_extractor_cfg(self, extractor_name, extractor_cfg):
        '''
        Change the configuration for an extractor, such as layers of a CNN used.
        '''
        if not extractor_name in self.extractors:
            raise ValueError('Extractor not found.')
        self.extractors[extractor_name]['cfg'] = extractor_cfg
        if self.random_forest is not None:
            self.adjusted = True
            if self.verbose:
                print('Warning, changed an extractor config. ' +
                      'Random Forest model might not be compatible anymore.')

    def set_extractor_options(self, extractor_name, num_pcs=0, smoothening=0):
        '''
        Set options for an extractor, such as number of principal components or smoothening factor.
        '''
        if not extractor_name in self.extractors:
            raise ValueError('Extractor not found.')
        self.extractors[extractor_name]['num_pcs'] = num_pcs
        self.extractors[extractor_name]['smoothening'] = smoothening
        if self.random_forest is not None:
            self.adjusted = True
            if self.verbose:
                print('Warning, changed extractor options. ' +
                      'Random Forest model might not be compatible anymore.')

    def remove_extractor(self, extractor_name):
        '''
        Remove an extractor from the list of extractors.
        '''
        if not extractor_name in self.extractors:
            raise ValueError('Extractor not found.')
        del self.extractors[extractor_name]
        if self.random_forest is not None:
            self.adjusted = True
            if self.verbose:
                print('Warning, removed and extractor. ' +
                      'Random Forest model might not be compatible anymore.')

    def get_extractor_info(self, extractor_name):
        '''
        Return a string with information about a single extractor.
        '''
        if not extractor_name in self.extractors:
            raise ValueError('Extractor not found.')
        out_str = f'{extractor_name}:\n'
        out_str += f'  Function:    {self.extractors[extractor_name]["func"].__name__}\n'
        out_str += f'  Config:      {self.extractors[extractor_name]["cfg"]}\n'
        out_str += f'  Num PCs:     {self.extractors[extractor_name]["num_pcs"]}\n'
        out_str += f'  Smoothening: {self.extractors[extractor_name]["smoothening"]}\n'
        return out_str

    def get_extractors_infos(self):
        '''
        Return a string with information about all extractors.
        '''
        if len(self.extractors) == 0:
            return '  No extractors'
        out_str = ''
        for extractor_name in self.extractors:
            out_str += self.get_extractor_info(extractor_name)
        return out_str

    def extract_features_single_extractor(self, img, extractor_name):
        '''
        Extract features using a single extractor on an image.
        '''
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
            filter_footprint = morphology.disk(smoothening_factor)
            feature_space = np.array([filters.median(f, footprint=filter_footprint)
                                      for f in feature_space])
            feature_space = np.moveaxis(feature_space, 0, 2)
        return feature_space

    def extract_features(self, img):
        '''
        Extract features using the specified extractor(s) on an image.
        '''
        # features_list = []
        # for extractor_name in self.extractors:
        #     features_list.append(self.extract_features_single_extractor(img, extractor_name))
        # features_combined = np.concatenate(features_list, axis=-1)
        features_list = [self.extract_features_single_extractor(img, extractor_name)
                         for extractor_name in self.extractors
                         if self.extractors]
        if self.options["PCs as features"]:
            num_pcs = self.options["PCs as features"]
            pca_features = get_pca_features(features_combined, num_pcs)
        else:
            pca_features = []
        if self.options["IMG as feature"]:
            if len(img.shape) == 2:
                img = np.expand_dims(img, axis=2)
        else:
            img = []
        features_combined = np.concatenate([pca_features] + features_list + [img], axis=-1)
        return features_combined

    def rf_train(self, img, labels):
        '''
        Extract features using the specified extractor(s) on an image with its labels,
        and train a Random Forest model with the labels.
        '''
        feature_space = self.extract_features(img)
        features_annot, labels = get_features_targets(feature_space, labels)
        self.random_forest = RandomForestClassifier(n_estimators=self.rf_settings["Nr. estimators"],
                                         random_state=self.rf_settings["Random state"])
        self.random_forest.fit(features_annot, labels)
        self.adjusted = False

    def rf_train_batch(self, img_batch, labels_batch,
                       print_progress=False):
        '''
        Extract features using the specified extractor(s) on a batch of images using a batch of labels,
        and train a Random Forest model with the labels.
        '''
        test_img_labels_batch_shapes(img_batch, labels_batch)
        # Extract features and targets for the entire batch (but only of annotated images & pixels)
        features_list = []
        targets_list = []
        i = 0
        num_labelled = sum(np.any(labels) for labels in labels_batch)
        t_start = time()
        # Iterate over images and their labels; extract features & targets for each annotated pixel
        for img, labels in zip(img_batch, labels_batch):
            if np.all(labels == 0):
                continue
            if print_progress:
                if i == 0:
                    est_t = "NA"
                else:
                    est_t = f"{((time()-t_start)/(i))*(num_labelled-i):.1f} seconds"
                print(f'Extracting features for labels {i+1}/{num_labelled} ' +
                      f'- estimated time left: {est_t}')
                i += 1
            feature_space = self.extract_features(img)
            features_annot, targets = get_features_targets(feature_space, labels)
            features_list.append(features_annot)
            targets_list.append(targets)
        features_annot = np.concatenate(features_list)
        targets = np.concatenate(targets_list)
        # Train the random forest classifier
        self.random_forest = RandomForestClassifier(n_estimators=self.rf_settings["Nr. estimators"],
                                         random_state=self.rf_settings["Random state"])
        self.random_forest.fit(features_annot, targets)
        self.adjusted = False

    def rf_predict(self, img):
        '''
        Predict the labels for an image using the trained Random Forest model.
        '''
        if self.random_forest is None:
            raise ValueError('Random Forest model not trained.')
        if self.adjusted and self.verbose:
            print('Warning, model adjusted since training/loading. Consider retraining the model.')
        feature_space = self.extract_features(img)
        num_features = feature_space.shape[2]
        num_pix = img.shape[0]*img.shape[1]
        features = np.reshape(feature_space, (num_pix, num_features))
        predicted_labels = self.random_forest.predict(features)
        pred_img = np.reshape(predicted_labels, img.shape[:2])
        pred_smoothening_factor = self.options["Smoothen preds"]
        if pred_smoothening_factor:
            pred_img = filters.rank.majority(pred_img,
                                             footprint=morphology.disk(pred_smoothening_factor))
        self.prediction_history.append(pred_img)
        return pred_img

    def rf_predict_batch(self, img_batch,
                         print_progress=False):
        '''
        Predict the labels for a batch of images using the trained Random Forest model.
        '''
        if isinstance(img_batch, list):
            img_batch = np.array(img_batch)
        pred_batch = np.zeros(img_batch.shape[:3], dtype=np.uint8)
        t_start = time()
        for i, img in enumerate(img_batch):
            if print_progress:
                if i == 0:
                    est_t = "NA"
                else:
                    est_t = f"{((time()-t_start)/(i))*(len(img_batch)-i):.1f} seconds"
                print(f'Predicting image {i+1}/{len(img_batch)} - estimated time left: {est_t}')
            pred_batch[i] = self.rf_predict(img)
        self.prediction_history.append(pred_batch)
        return pred_batch

    def rf_segment(self, train_img_batch, labels_batch, pred_img_batch=None,
                   print_progress=False):
        '''
        Extract features using the specified extractor(s) on a batch of images using a batch of labels,
        train a Random Forest model with the labels,
        and use it to predict the labels for an entire image batch (or the training batch if not specified).
        '''
        if pred_img_batch is None:
            pred_img_batch = train_img_batch
        self.rf_train_batch(train_img_batch, labels_batch,
                            print_progress=print_progress)
        pred_batch = self.rf_predict_batch(pred_img_batch,
                                           print_progress=print_progress)
        return pred_batch

    def rf_selfpredict(self, img, labels):
        '''
        Extract features using the specified extractor(s),
        train a Random Forest model with the labels, and
        use it to predict the labels for the entire image.
        '''
        # Extract features
        feature_space = self.extract_features(img)
        # Train the random forest on the annotated pixels
        features_annot, labels = get_features_targets(feature_space, labels)
        self.random_forest = RandomForestClassifier(n_estimators=self.rf_settings["Nr. estimators"],
                                         random_state=self.rf_settings["Random state"])
        self.random_forest.fit(features_annot, labels)
        self.adjusted = False
        # Predict the labels for all pixels in the image
        num_features = feature_space.shape[2]
        num_pix = img.shape[0]*img.shape[1]
        features = np.reshape(feature_space, (num_pix, num_features))
        predicted_labels = self.random_forest.predict(features)
        pred_img = np.reshape(predicted_labels, img.shape[:2])
        pred_smoothening_factor = self.options["Smoothen preds"]
        if pred_smoothening_factor:
            pred_img = filters.rank.majority(pred_img,
                                             footprint=morphology.disk(pred_smoothening_factor))
        self.prediction_history.append(pred_img)
        return pred_img

    def rf_save(self, filename):
        '''
        Save the trained Random Forest model and its settings to files.
        '''
        if self.random_forest is None:
            raise ValueError('No trained Random Forest model.')
        joblib.dump(self.random_forest, filename+'.joblib')
        data = {"options":self.options,
                "rf settings":self.rf_settings,
                "extractors": self.extractors}
        with open(filename+'.yml', 'w') as file:
            yaml.dump(data, file, default_flow_style=False)

    def rf_load(self, filename):
        '''
        Load a trained Random Forest model and its settings from files.
        '''
        self.random_forest = joblib.load(filename+'.joblib')
        with open(filename+'.yml', 'r') as file:
            data = yaml.load(file, Loader=yaml.FullLoader)
        self.options = data["options"]
        self.rf_settings = data["rf settings"]
        self.extractors = data["extractors"]
        self.adjusted = False

    def get_kmeans(self, img, num_clusters):
        '''
        Extract features using the specified extractor(s), then cluster them using KMeans.
        '''
        feature_space = self.extract_features(img)
        clusters = get_kmeans_clusters(feature_space, num_clusters)
        self.prediction_history.append(clusters)
        return clusters


# ToDo:
#  - Use linter, add unit-tests etc.
#  - Classifier agnostic implementation
#  - Add more different classifiers
