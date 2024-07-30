from seghub.util_funcs import *
import unittest
import numpy as np

import unittest
import numpy as np

class TestReshapePatchesToImg(unittest.TestCase):
    
    def test_correct_input_1d(self):
        patches_values = np.random.rand(32)
        image_shape = (40, 20)
        patch_size = (5, 5)
        
        patch_img = reshape_patches_to_img(patches_values, image_shape, patch_size)
        
        self.assertEqual(patch_img.shape, image_shape)
    
    def test_correct_input_2d(self):
        patches_values = np.random.rand(32, 3)
        image_shape = (40, 20)
        patch_size = (5, 5)
        
        patch_img = reshape_patches_to_img(patches_values, image_shape, patch_size)
        
        self.assertEqual(patch_img.shape, (40, 20, 3))
    
    def test_mismatched_dimensions(self):
        patches_values = np.random.rand(32, 3)
        image_shape = (41, 20)  # Not a multiple of patch size (5, 5)
        patch_size = (5, 5)
        
        with self.assertRaises(ValueError):
            reshape_patches_to_img(patches_values, image_shape, patch_size)
    
    def test_interpolation_order_none(self):
        patches_values = np.random.rand(16)
        image_shape = (16, 16)
        patch_size = (4, 4)
        
        patch_img = reshape_patches_to_img(patches_values, image_shape, patch_size,
                                           interpolation_order=None)
        
        self.assertEqual(patch_img.shape, image_shape)
    
    def test_interpolation_order_nonzero(self):
        patches_values = np.random.rand(16, 3)
        image_shape = (16, 16)
        patch_size = (4, 4)
        interpolation_order = 3
        
        patch_img = reshape_patches_to_img(patches_values, image_shape, patch_size, interpolation_order=interpolation_order)
        
        self.assertEqual(patch_img.shape, (16, 16, 3))
    
    def test_invalid_patch_dimensions(self):
        patches_values = np.random.rand(100, 3, 3)  # Invalid patch dimension
        image_shape = (20, 20)
        patch_size = (5, 5)
        
        with self.assertRaises(ValueError):
            reshape_patches_to_img(patches_values, image_shape, patch_size)

class TestGetFeaturesTargets(unittest.TestCase):
    
    def test_correct_input(self):
        feature_space = np.random.rand(10, 10, 5)  # 10x10 spatial dimension, 5 features
        labels = np.random.randint(0, 3, (10, 10))  # 10x10 spatial dimensions, labels 0, 1 and 2
        
        features_annot, targets = get_features_targets(feature_space, labels)
        
        # Check if shapes are correct
        self.assertEqual(features_annot.shape[1], 5)
        self.assertEqual(features_annot.shape[0], np.sum(labels > 0))
        self.assertEqual(targets.shape[0], np.sum(labels > 0))
    
    def test_mismatched_dimensions(self):
        feature_space = np.random.rand(10, 10, 5)
        labels = np.random.randint(0, 2, (8, 10))  # Different spatial dimensions
        
        with self.assertRaises(ValueError):
            get_features_targets(feature_space, labels)
    
    def test_no_annotated_pixels(self):
        feature_space = np.random.rand(10, 10, 5)
        labels = np.zeros((10, 10))  # All labels are zero
        
        features_annot, targets = get_features_targets(feature_space, labels)
        
        # Check if the outputs are empty
        self.assertEqual(features_annot.shape[0], 0)
        self.assertEqual(targets.shape[0], 0)

    def test_all_annotated_pixels(self):
        feature_space = np.random.rand(10, 10, 5)
        labels = np.ones((10, 10))  # All labels are 1
        
        features_annot, targets = get_features_targets(feature_space, labels)
        
        # Check if all pixels are included
        self.assertEqual(features_annot.shape[0], 100)
        self.assertEqual(targets.shape[0], 100)
        self.assertTrue((targets == 1).all())

    def test_random_labels(self):
        feature_space = np.random.rand(10, 10, 5)
        labels = np.random.randint(-1, 2, (10, 10))  # Labels can be -1, 0, or 1
        
        features_annot, targets = get_features_targets(feature_space, labels)
        
        # Check if non-zero labels are included
        self.assertEqual(features_annot.shape[0], np.sum(labels > 0))
        self.assertTrue((targets > 0).all())


class TestTestImgLabelsBatchShapes(unittest.TestCase):
    def test_input(self):
        single_img_1 = np.ones((480, 640))
        single_img_2 = np.ones((640, 480))
        single_img_1_multi_rgb = np.ones((480, 640, 3))
        single_img_1_multi_ch = np.ones((480, 640, 5))
        single_label_1 = np.ones((480, 640))
        single_label_2 = np.ones((640, 480))

        # If labels batch is longer than images batch (more frames)
        img_batch = [single_img_1]
        labels_batch = 2*[single_label_1]
        self.assertRaises(ValueError, test_img_labels_batch_shapes,
                          img_batch, labels_batch)

        # If labels is shorter along first dimension than images (fewer frames)
        img_batch = 2*[single_img_1]
        labels_batch = [single_label_1]
        self.assertRaises(ValueError, test_img_labels_batch_shapes,
                          img_batch, labels_batch)

        # If images and labels have different spatial shapes (dim 1 and 2)
        img_batch = [single_img_1]
        labels_batch = [single_label_2]
        self.assertRaises(ValueError, test_img_labels_batch_shapes,
                          img_batch, labels_batch)

        # If not all images are either single-channel or multi-channel
        img_batch = [single_img_1, single_img_1_multi_rgb]
        labels_batch = 2*[single_label_1]
        self.assertRaises(ValueError, test_img_labels_batch_shapes,
                          img_batch, labels_batch)

        # If not all images have the same number of channels
        img_batch = [single_img_1_multi_rgb, single_img_1_multi_ch]
        labels_batch = 2*[single_label_1]
        self.assertRaises(ValueError, test_img_labels_batch_shapes,
                          img_batch, labels_batch)

        # If all inputs are correct;
        # including that 2 images can have different spatial shapes
        img_batch = [single_img_1, single_img_2]
        labels_batch = [single_label_1, single_label_2]
        self.assertIsNone(test_img_labels_batch_shapes(img_batch, labels_batch))