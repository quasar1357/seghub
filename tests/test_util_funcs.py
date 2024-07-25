from seghub.util_funcs import *
import unittest
import numpy as np

class TestUtilFuncs(unittest.TestCase):
    def test_test_img_labels_batch_shapes(self):
        # If labels is longer along first dimension than images (more frames)
        self.assertRaises(ValueError, test_img_labels_batch_shapes, [np.array((480, 640))], 2*[np.array((480, 640))])
        # If labels is shorter along first dimension than images (fewer frames)
        self.assertRaises(ValueError, test_img_labels_batch_shapes, 2*[np.array((480, 640))], [np.array((480, 640))])
        # If images and labels have different shapes
        self.assertRaises(ValueError, test_img_labels_batch_shapes, [np.array((480, 640))], [np.array((640, 480))])