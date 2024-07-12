import numpy as np
from skimage.transform import resize
from time import time
import warnings

def norm_for_imagenet(img_arr):
    '''
    Normalize array to ImageNet mean and standard deviation.
    INPUT:
        img_arr (np.ndarray): array to normalize; shape (H, W, C) where C=3
    OUTPUT:
        img_arr_norm (np.ndarray): normalized array; same shape as input
    '''
    if np.min(img_arr) < 0 or np.max(img_arr) > 1:
        warnings.warn('Image is not in the range [0, 1]. Are you sure you pre-processed the image correctly?')
    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
    img_arr_norm = (img_arr - mean) / std
    return img_arr_norm

def normalize_np_array(array, new_mean, new_sd, axis=(0,1)):
    '''
    Normalizes a numpy array to a new mean and standard deviation.
    '''
    current_mean, current_sd = np.mean(array, axis=axis), np.std(array, axis=axis)
    # Avoid division by zero; leads to setting channels with all the same value to the new mean
    current_sd[current_sd == 0] = 1
    new_mean, new_sd = np.array(new_mean), np.array(new_sd)
    array_norm = (array - current_mean) / current_sd
    array_norm = array_norm * new_sd + new_mean
    return array_norm

def pad_to_patch(image, vert_pos="center", hor_pos="center", pad_mode='constant', patch_size=(14,14)):
    '''
    Pads an image to the next multiple of patch size.
    The pad position can be chosen on both axis in the tuple (vert, hor),
    where vert can be "top", "center" or "bottom" and hor can be "left", "center" or "right".
    pad_mode can be chosen according to numpy pad method.
    '''
    # If image is an rgb image, run this function on each channel
    if len(image.shape) == 3:
        channel_list = np.array([pad_to_patch(image[:,:, channel], vert_pos, hor_pos, pad_mode, patch_size) for channel in range(image.shape[2])])
        rgb_padded = np.moveaxis(channel_list, 0, 2)
        return rgb_padded
    # For a greyscale image (or each separate RGB channel):
    h, w = image.shape
    ph, pw = patch_size
    # Calculate how much padding has to be done in total on each axis
    # The total pad on one axis is a patch size minus whatever remains when dividing the picture size including the extra pads by the patch size
    # The  * (h % ph != 0) term (and same with wdith) ensure that the pad is 0 if the shape is already a multiple of the patch size
    vertical_pad, horizontal_pad = calculate_padding(image.shape, patch_size=patch_size)
    # Define the paddings on each side depending on the chosen positions
    top_pad = {"top": vertical_pad,
               "center": np.ceil(vertical_pad/2),
               "bottom": 0
               }[vert_pos]
    bot_pad = vertical_pad - top_pad
    left_pad = {"left": horizontal_pad,
                "center": np.ceil(horizontal_pad/2),
                "right": 0
                }[hor_pos]
    right_pad = horizontal_pad - left_pad
    # Make sure paddings are ints
    top_pad, bot_pad, left_pad, right_pad = int(top_pad), int(bot_pad), int(left_pad), int(right_pad)
    # Pad the image using the pad sizes as calculated and the mode given as input
    image_padded = np.pad(image, ((top_pad, bot_pad), (left_pad, right_pad)), mode=pad_mode)
    return image_padded

def calculate_padding(im_shape, patch_size=(14,14)):
    '''
    Takes an image shape - (H, W, C) or (H, W) - and a patch_size and returns the padding needed to make the image a multiple of the patch size (v, h).
    '''
    h, w = im_shape[:2]
    ph, pw = patch_size
    # Calculate how much padding has to be done in total on each axis
    # The total pad on one axis is a patch size minus whatever remains when dividing the picture size including the extra pads by the patch size
    # The  * (h % ph != 0) term (and same with wdith) ensure that the pad is 0 if the shape is already a multiple of the patch size
    vertical_pad = (ph - h % ph) * (h % ph != 0)
    horizontal_pad = (pw - w % pw) * (w % pw != 0)
    return vertical_pad, horizontal_pad

def reshape_patches_to_img(patches, image_shape, patch_size=(14,14), interpolation_order=None):
    '''
    Takes linearized patches, with or without a second dimension for features,
    and reshapes them to the size of the image.
    If interpolation_order is None or 0, the patches are simply repeated.
    If interpolation_order is not None and not 0, the patches are resized to the image size.
    INPUT:
        patches (np.ndarray): linearized patches. Shape (Hp * Wp, F) or (Hp * Wp)
        image_shape (tuple of int): shape of the image to reshape to
        patch_size (tuple of int): size of the patches
        interpolation_order (int): order of interpolation for resizing the patches
    OUTPUT:
        patch_img (np.ndarray): reshaped image. Shape (H, W) or (H, W, F)
    '''
    if not (image_shape[0]%patch_size[0] == 0 and image_shape[1]%patch_size[1] == 0):
        raise ValueError('Image shape must be multiple of patch size')
    if len(patches.shape) == 1:
        patch_as_pix_shape = int(image_shape[0] / patch_size[0]), int(image_shape[1] / patch_size[1])
    elif len(patches.shape) == 2:
        patch_as_pix_shape = int(image_shape[0] / patch_size[0]), int(image_shape[1] / patch_size[1]), patches.shape[1]
    else:
        raise ValueError('Patches must have one or two dimensions')
    patch_img = np.reshape(patches, patch_as_pix_shape)
    if interpolation_order is None or interpolation_order == 0:
        # Repeat each row and each column according to the patch size to recreate the patches
        patch_img = np.repeat(patch_img, patch_size[0], axis=0)
        patch_img = np.repeat(patch_img, patch_size[1], axis=1)
    else:
        patch_img = resize(patch_img, image_shape[0:2], mode='edge', order=interpolation_order, preserve_range=True)
    return patch_img

def get_features_targets(feature_space, labels):
    '''
    Takes a feature space and labels of same dimensions and returns the features of annotated pixels and their targets.
    INPUT:
        feature_space (np.ndarray): feature space. Shape (H, W, F)
        labels (np.ndarray): labels. Shape (H, W)
    OUTPUT:
        features_annot (np.ndarray): features of annotated pixels. Shape (n_annotated, F)
        targets (np.ndarray): targets of annotated pixels. Shape (n_annotated)
    '''
    if not feature_space.shape[:2] == labels.shape:
        raise ValueError('Feature space and labels must have the same spatial dimensions')

    # Flatten the spatial dimensions (keeping the features in the last dimension) --> per pixel features
    labels_flat = labels.flatten()
    num_pix = len(labels_flat)
    num_features = feature_space.shape[2]
    features_flat = np.reshape(feature_space, (num_pix, num_features))
    # Extract only the annotated pixels
    labels_mask = labels_flat > 0
    targets = labels_flat[labels_mask]
    features_annot = features_flat[labels_mask]
    return features_annot, targets

def test_img_labels_batch_shapes(image_batch, labels_batch):
    '''
    Takes an image batch and a label batch and checks if they have the correct shapes.
    OUTPUT:
        None
    '''
    if not len(image_batch) == len(labels_batch):
        raise ValueError('Image and label batch must have the same length (each image needs its labels)')
    if not all([image.shape[:2] == labels.shape for image, labels in zip(image_batch, labels_batch)]):
        raise ValueError('Each image and its labels must have the same spatial dimensions')
    if not all([len(image.shape) == len(image_batch[0].shape) for image in image_batch]):
        raise ValueError('All images in the batch must have the same number of channels')
    is_multichannel = len(image_batch[0].shape) == 3
    if is_multichannel:
        all_same_channels = all([image.shape[2] == image_batch[0].shape[2] for image in image_batch])
        if not all_same_channels:
            raise ValueError('All images in the batch must have the same number of channels')
    return None
