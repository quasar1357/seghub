import warnings
import numpy as np
from napari_convpaint.conv_paint_utils import Hookmodel, get_features_current_layers
import torch
from torch.nn.functional import interpolate as torch_interpolate
from seghub.util_funcs import norm_for_imagenet, get_features_targets

def get_vgg16_layers(model_name="vgg16"):
    '''Get the layers of a Hookmodel model (notably vgg16).'''
    model = Hookmodel(model_name=model_name)
    return model.module_dict

def get_vgg16_feature_space(image, layer_list=[0], scalings=[1,2],
                            model_name="vgg16", rgb_if_possible=True):
    '''
    Use vgg16 to extract features from an image (entire image); returns them in input image shape.
    If the image has 3 channels and RGB is chosen, extract features as usual.
    Otherwise extract features for each channel (triplicated for the 3 color channels)
    and concatenate them.
    INPUT:
        image (np.ndarray): image to extract features from. Shape (H, W) or (H, W, C)
            Expects the image to be in the range [0, 1]; will normalize to ImageNet stats
        layer_list (list of int): list of layer indices to use for feature extraction with vgg16
        scalings (list of int): list of scalings to use for feature extraction with vgg16
        model (str): model to use for feature extraction
        rgb_if_possible (bool): whether to treat a 3-channel image as RGB or not
    OUTPUTS:
        features (np.ndarray): extracted features. Shape (H, W, F)
    '''
    # We take (H, W, C) but for VGG16 need [C, H, W]
    if len(image.shape) == 3:
        image = np.moveaxis(image, -1, 0)
    # Define the model
    model = Hookmodel(model_name=model_name)
    # Ensure the layers are a list
    if isinstance(layer_list, int):
        layer_list = [layer_list]
    # Read out the layer names
    all_layers = list(model.module_dict.keys())
    layers = [all_layers[i] for i in layer_list]
    # Register the hooks for the selected layers
    model.register_hooks(selected_layers=layers)
    features = filter_image_multichannels_rgb(image, model, scalings=scalings,
                                          order=1, image_downsample=1,
                                          rgb_if_possible=rgb_if_possible)
    # Concatenate feature space (the function above returns a list)
    feature_space = np.concatenate(features, axis=1)
    # Remove first dimension (the function above assumes time-frames,
    # of which we only use a single one)
    feature_space = feature_space[0]
    # Move the feature dimension to the last axis to return (H, W, F) (like the input image shape)
    feature_space = np.moveaxis(feature_space, 0, -1)
    return feature_space

def get_vgg16_features_targets(image, labels, layer_list=[0], scalings=[1,2],
                               model_name="vgg16", rgb_if_possible=True):
    '''
    Takes an image and extracts features using a vgg16 model;
    returns the features of annotated pixels and their targets.
    If the image has 3 channels and RGB is chosen, extract features as usual.
    Otherwise extract features for each channel (triplicated for the 3 color channels)
    and concatenate them.
    INPUT:
        image (np.ndarray): image to extract features from. Shape (H, W) or (H, W, C)
        labels (np.ndarray): labels. Shape (H, W)
        layer_list (list of int): list of layer indices to use for feature extraction with vgg16
        scalings (list of int): list of scalings to use for feature extraction with vgg16
        model (str): model to use for feature extraction
        rgb_if_possible (bool): whether to treat a 3-channel image as RGB or not
    OUTPUT:
        features_annot (np.ndarray): features of annotated pixels. Shape (n_annotated, F)
        targets (np.ndarray): targets of annotated pixels. Shape (n_annotated)
    '''
    feature_space = get_vgg16_feature_space(image, layer_list=layer_list, scalings=scalings,
                                            model_name=model_name, rgb_if_possible=rgb_if_possible)
    features_annot, targets = get_features_targets(feature_space, labels)
    return features_annot, targets

def get_vgg16_pixel_features(image, layer_list=[0], scalings=[1,2],
                             model_name="vgg16", rgb_if_possible=True):
    '''
    Takes an image and extracts features using a vgg16 model;
    returns linear per-pixel features.
    If the image has 3 channels and RGB is chosen, extract features as usual.
    Otherwise extract features for each channel (triplicated for the 3 color channels)
    and concatenate them.
    INPUT:
        image (np.ndarray): image to extract features from. Shape (H, W) or (H, W, C)
        layer_list (list of int): list of layer indices to use for feature extraction with vgg16
        scalings (list of int): list of scalings to use for feature extraction with vgg16
        model (str): model to use for feature extraction
        rgb_if_possible (bool): whether to treat a 3-channel image as RGB or not
    OUTPUT:
        features (np.ndarray): extracted features. Shape (H*W, F)
                               where F is the number of features extracted
    '''
    feature_space = get_vgg16_feature_space(image, layer_list=layer_list, scalings=scalings,
                                            model_name=model_name, rgb_if_possible=rgb_if_possible)
    # Flatten the spatial dimensions (keeping features in last dimension) --> per pixel features
    num_pix = image.shape[0] * image.shape[1]
    num_features = feature_space.shape[2]
    pixel_features_flat = np.reshape(feature_space, (num_pix, num_features))
    return pixel_features_flat

### NOTE: THIS FUNCTION WAS COPIED FROM CONVPAINT AND ADJUSTED TO USE CORRECT RGB FEATURE EXTRACTION

def filter_image_multichannels_rgb(image, hookmodel, scalings=[1], order=0,
                              image_downsample=1, rgb_if_possible=True):
    """Recover the outputs of chosen layers of a pytorch model. Layers and model are
    specified in the hookmodel object. If image has multiple channels, each channel
    is processed separately.
    
    Parameters
    ----------
    image : np.ndarray
        2d Image to filter
    hookmodel : Hookmodel
        Model to extract features from
    scalings : list of ints, optional
        Downsampling factors, by default None
    order : int, optional
        Interpolation order for low scale resizing,
        by default 0
    image_downsample : int, optional
        Downsample image by this factor before extracting features, by default 1

    Returns
    -------
    all_scales : list of np.ndarray
        List of filtered images. The number of images is C x Sum_i(F_i x S)
        where C is the number of channels, F_i is the number of filters of the ith layer
        and S the number of scaling factors.

    """
    input_channels = hookmodel.named_modules[0][1].in_channels
    image = np.asarray(image, dtype=np.float32)

    if image.ndim == 2:
        # print("Image is grayscale")
        image = image[::image_downsample, ::image_downsample]
        image = np.ones((input_channels, image.shape[0], image.shape[1]), dtype=np.float32) * image
        image_series = [image]
    elif rgb_if_possible and image.ndim == 3 and image.shape[0] == input_channels:
        # print("Using RGB")
        image = image[:, ::image_downsample, ::image_downsample]
        image_series = [image]
    elif image.ndim == 3:
        # print("NOT using RGB")
        if rgb_if_possible:
            raise Warning("RGB_if_possible is on, " +
                          "but image has different number of channels than model. NOT using RGB.")
        image = image[:, ::image_downsample, ::image_downsample]
        # Create "fake" RGB images by triplicating the channels
        image_series = [np.ones((input_channels, im.shape[0], im.shape[1]), dtype=np.float32) * im
                        for im in image]
    else:
        raise ValueError("Incompatible image shape:", image.shape)

    int_mode = 'bilinear' if order > 0 else 'nearest'
    align_corners = False if order > 0 else None

    all_scales = []
    with torch.no_grad():
        for input_image in image_series:
            # Normalize the image to ImageNet stats; make sure to keep the original dtype
            old_type = input_image.dtype
            # norm_for_imagenet expects (H, W, C) --> move channels and then back
            input_image = np.moveaxis(norm_for_imagenet(np.moveaxis(input_image, 0, -1)), -1, 0)
            input_image = input_image.astype(old_type)
            # Loop and concatenate the outputs of the scalings
            for scaling in scalings:
                im_tot = input_image[:, ::scaling, ::scaling]
                im_torch = torch.tensor(im_tot[np.newaxis, ::])
                hookmodel.outputs = []
                try:
                    _ = hookmodel(im_torch)
                except AssertionError:
                    pass
                except Exception as ex:
                    raise ex
                # If the output is not the same size as the input (esp. later layers), resize it
                for output_layer in hookmodel.outputs:
                    if input_image.shape[1:3] != output_layer.shape[2:4]:
                        output_layer = torch_interpolate(output_layer,
                                                         size=input_image.shape[1:3],
                                                         mode=int_mode,
                                                         align_corners=align_corners)
                        # Alternative way of resizing...
                        # out_shape = (1, out_np.shape[1],
                        #              input_image.shape[1], input_image.shape[2])
                        # out_np = skimage.transform.resize(
                        #     input_image=out_np,
                        #     output_shape=out_shape,
                        #     preserve_range=True, order=order)

                    out_np = output_layer.cpu().detach().numpy()
                    all_scales.append(out_np)

    return all_scales
