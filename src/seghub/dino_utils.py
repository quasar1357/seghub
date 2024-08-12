import numpy as np
import torch
from torchvision.transforms import ToTensor
import timm
from seghub.util_funcs import (norm_for_imagenet,
                               pad_to_patch, reshape_patches_to_img, calculate_padding,
                               get_features_targets)
from seghub.classif_utils import get_pca_features
# from huggingface_hub import hf_hub_download

loaded_dinov2_models = {}

def preprocess_for_dinov2(image):
    '''
    Preprocesses an image for DINOv2 feature extraction.
    INPUT:
        image (np.ndarray): image. Shape (H, W, C)
            Expects the image to be in the range [0, 1]
    OUTPUT:
        image_batch (torch.Tensor): preprocessed image. Shape (1, C, H, W)
    '''
    # Normalize the image to ImageNet stats
    image = norm_for_imagenet(image)
    # Convert to tensor and add batch dimension
    image_tensor = ToTensor()(image).float()
    image_batch = image_tensor.unsqueeze(0)
    return image_batch

def extract_features_rgb(image, dinov2_model='s_r'):
    '''
    Takes an RGB image and extracts features using a DINOv2 model.
    INPUT:
        image (np.ndarray): RGB image. Shape (H, W, C) where C=3
            Expects H and W as multiples of patch size (=14x14 for original DINOv2, 16x16 for UNI)
            Expects the image to be in the range [0, 1]; will normalize to ImageNet stats
        dinov2_model (str): model to use for feature extraction.
            Options: 's', 'b', 'l', 'g', 's_r', 'b_r', 'l_r', 'g_r' (r = registers)
                     'uni' = https://github.com/mahmoodlab/UNI
    OUTPUT:
        features (np.ndarray): extracted features. Shape (Hp*Wp, F)
                               where F is the number of features extracted,
                               and Hp and Wp are patches per height and width, respectively
    '''
    if dinov2_model == 'uni':
        return extract_uni_features_rgb(image)

    # Preprocess image
    image_batch = preprocess_for_dinov2(image)
    # Define and load model
    models = {'s': 'dinov2_vits14',
            'b': 'dinov2_vitb14',
            'l': 'dinov2_vitl14',
            'g': 'dinov2_vitg14',
            's_r': 'dinov2_vits14_reg',
            'b_r': 'dinov2_vitb14_reg',
            'l_r': 'dinov2_vitl14_reg',
            'g_r': 'dinov2_vitg14_reg'}
    dinov2_name = models[dinov2_model]
    if dinov2_name not in loaded_dinov2_models:
        try:
            loaded_dinov2_models[dinov2_name] = torch.hub.load('facebookresearch/dinov2',
                                                               dinov2_name, pretrained=True,
                                                               verbose=False)
        # The force_reload might be necessary for models with registers
        except RuntimeError:
            loaded_dinov2_models[dinov2_name] = torch.hub.load('facebookresearch/dinov2',
                                                               dinov2_name, pretrained=True,
                                                               verbose=False, force_reload=True)
    model = loaded_dinov2_models[dinov2_name]
    model.eval()
    # Extract features
    with torch.no_grad():
        features_dict = model.forward_features(image_batch)
        features = features_dict['x_norm_patchtokens']
    # Convert to numpy array
    features = features.numpy()
    # Remove batch dimension
    features = features[0]
    return features

def extract_uni_features_rgb(image):
    '''
    Takes an RGB image and extracts features using the UNI model (for pathology).
    NOTE: User must be logged in to huggingface and have access to the UNI model use:
            from huggingface_hub import login; login(hf_token)
          where hf_token is your personal huggingface token found at
            https://huggingface.co/settings/tokens
    INPUT:
        image (np.ndarray): RGB image. Shape (H, W, C) where C=3
            Expects H and W to be multiples of patch size (=16x16)
            Expects the image to be in the range [0, 1]; will normalize to ImageNet stats
    OUTPUT:
        features (np.ndarray): extracted features. Shape (Hp*Wp, F)
                               where F is the number of features extracted,
                               and Hp and Wp are patches per height and width, respectively
    '''
    # Preprocess image
    image_batch = preprocess_for_dinov2(image)
    # Load model
    if 'uni' in loaded_dinov2_models:
        model = loaded_dinov2_models['uni']
    else:
        model = timm.create_model("hf-hub:MahmoodLab/uni",
                                  pretrained=True, init_values=1e-5, dynamic_img_size=True)
        # To download the model weights to a specified checkpoint location
        # model = timm.create_model(
        #     "vit_large_patch16_224", img_size=224, patch_size=16,
        #     init_values=1e-5, num_classes=0, dynamic_img_size=True
        # )
        # try:
        #     model_file = hf_hub_download("MahmoodLab/UNI", filename="pytorch_model.bin", force_download=True)
        #     model.load_state_dict(torch.load(model_file, map_location="cpu"), strict=True)
        # # The force_download might be necessary if the model is not found in the cache
        # except RuntimeError:
        #     model_file = hf_hub_download("MahmoodLab/UNI", filename="pytorch_model.bin", force_download=True)
        #     model.load_state_dict(torch.load(model_file, map_location="cpu"), strict=True)
        loaded_dinov2_models['uni'] = model
    model.eval()
    # Make sure image is on same device as model
    device = next(model.parameters())[0].device
    image_batch = image_batch.to(device)
    # Extract features
    with torch.inference_mode():
        patch_embeddings = model.patch_embed(image_batch)
    # Remove batch dimension and convert to numpy array
    features = patch_embeddings.squeeze().numpy()
    # Linearize (to be consistent with original DINOv2 output)
    features = features.reshape(-1, features.shape[2])
    return features

def extract_features_multichannel(image, dinov2_model='s_r'):
    '''
    Takes an image with multiple channels and extracts features using a DINOv2 model.
    Treats each channel as an RGB image (copying it for R, G and B), extracts features separately,
    and concatenates them.
    INPUT:
        image (np.ndarray): image with multiple channels. Shape (H, W, C) or (H, W)
            Expects H and W as multiples of patch size (=14x14 for original DINOv2, 16x16 for UNI)
            Expects the image to be in the range [0, 1]; will normalize to ImageNet stats
        dinov2_model (str): model to use for feature extraction.
            Options: 's', 'b', 'l', 'g', 's_r', 'b_r', 'l_r', 'g_r' (r = registers)
    OUTPUT:
        features (np.ndarray): extracted features. Shape (H*W, F*C)
                               where F is the number of features extracted per channel
    '''
    # If image is single channel, stack it to 3 dimensions (the last one being the single channel)
    if len(image.shape) == 2:
        image = np.stack((image,), axis=-1)
    # Extract features for each channel and concatenate them as separate features
    features_list = []
    for channel in range(image.shape[2]):
        # Use channel as r, g and b channels for feature extraction
        image_channel = image[:,:,channel]
        img_channel = np.stack((image_channel,)*3, axis=-1)
        channel_features = extract_features_rgb(img_channel, dinov2_model)
        features_list.append(channel_features)
    features = np.concatenate(features_list, axis=1)
    return features

def get_dinov2_patch_features(image, dinov2_model='s_r',
                              rgb_if_possible=True,
                              pc_as_features=False):
    '''
    Takes an image and extracts features using a DINOv2 model.
    If the image has 3 channels and RGB is chosen, extract features as usual.
    Otherwise extract features for each channel (triplicated for the 3 color channels)
    and concatenate them.
    INPUT:
        image (np.ndarray): image. Shape (H, W, C) or (H, W)
            Expects the image to be in the range [0, 1]; will normalize to ImageNet stats
        dinov2_model (str): model to use for feature extraction.
            Options: 's', 'b', 'l', 'g', 's_r', 'b_r', 'l_r', 'g_r' (r = registers)
        rgb_if_possible (bool): whether to treat a 3-channel image as RGB or not
    OUTPUT:
        features (np.ndarray): extracted features. Shape (Hp*Wp, F)
                               where F is the number of features extracted
                               and Hp and Wp are the number of patches per height and width
    '''
    patch_size = (14,14) if dinov2_model != 'uni' else (16,16)
    padded_image = pad_to_patch(image, "bottom", "right", patch_size=patch_size)
    # If the image has 3 channels and RGB is chosen, extract features as usual
    if len(padded_image.shape) == 3 and padded_image.shape[2] == 3 and rgb_if_possible:
        dinov2_features = extract_features_rgb(padded_image, dinov2_model)
    # If the image does not have 3 channels and/or RGB is not chosen,
    # extract features for each channel and concatenate them
    else:
        dinov2_features = extract_features_multichannel(padded_image, dinov2_model)
    # Apply PCA if chosen
    if pc_as_features:
        dinov2_features = get_pca_features(dinov2_features, num_components=pc_as_features)
    return dinov2_features

def get_dinov2_feature_space(image, dinov2_model='s_r',
                             rgb_if_possible=True,
                             pc_as_features=False,
                             interpolate_features=False):
    '''
    Takes an image, extracts features using a DINOv2 model,
    and reshapes them to input image shape/size.
    If the image has 3 channels and RGB is chosen, extract features as usual.
    Otherwise extract features for each channel (triplicated for the 3 color channels)
    and concatenate them.
    INPUT:
        image (np.ndarray): image. Shape (H, W, C) or (H, W)
            Expects the image to be in the range [0, 1]; will normalize to ImageNet stats
        dinov2_model (str): model to use for feature extraction.
            Options: 's', 'b', 'l', 'g', 's_r', 'b_r', 'l_r', 'g_r' (r = registers)
        rgb_if_possible (bool): whether to treat a 3-channel image as RGB or not
    OUTPUT:
        features (np.ndarray): extracted features. Shape (H, W, F)
                               where F is the number of features extracted
    '''
    patch_features_flat = get_dinov2_patch_features(image, dinov2_model=dinov2_model,
                                                    rgb_if_possible=rgb_if_possible,
                                                    pc_as_features=pc_as_features)
    # Recreate an image-sized feature space from the features
    patch_size = (14,14) if dinov2_model != 'uni' else (16,16)
    vertical_pad, horizontal_pad = calculate_padding(image.shape[:2], patch_size=patch_size)
    padded_img_shape = (image.shape[0] + vertical_pad, image.shape[1] + horizontal_pad)
    feature_space = reshape_patches_to_img(patch_features_flat, padded_img_shape,
                                           patch_size=patch_size,
                                           interpolation_order=interpolate_features)
    feature_space_recrop = feature_space[:image.shape[0], :image.shape[1]]
    return feature_space_recrop

def get_dinov2_pixel_features(image, dinov2_model='s_r',
                              rgb_if_possible=True,
                              pc_as_features=False,
                              interpolate_features=False):
    '''
    Takes an image, extracts features using a DINOv2 model,
    and reshapes them to per-pixel level.
    If the image has 3 channels and RGB is chosen, extract features as usual.
    Otherwise extract features for each channel (triplicated for the 3 color channels)
    and concatenate them.
    INPUT:
        image (np.ndarray): image. Shape (H, W, C) or (H, W)
            Expects the image to be in the range [0, 1]; will normalize to ImageNet stats
        dinov2_model (str): model to use for feature extraction.
            Options: 's', 'b', 'l', 'g', 's_r', 'b_r', 'l_r', 'g_r' (r = registers)
        rgb_if_possible (bool): whether to treat a 3-channel image as RGB or not
    OUTPUT:
        features (np.ndarray): extracted features. Shape (H*W, F)
                               where F is the number of features extracted
    '''
    feature_space = get_dinov2_feature_space(image, dinov2_model=dinov2_model,
                                             rgb_if_possible=rgb_if_possible,
                                             pc_as_features=pc_as_features,
                                             interpolate_features=interpolate_features)
    # Flatten the spatial dimensions (keeping the features in the last dimension)
    # --> per pixel features
    num_pix = image.shape[0] * image.shape[1]
    num_features = feature_space.shape[2]
    pixel_features_flat = np.reshape(feature_space, (num_pix, num_features))
    return pixel_features_flat

def get_dinov2_features_targets(image, labels, dinov2_model='s_r',
                                rgb_if_possible=True,
                                pc_as_features=False,
                                interpolate_features=False):
    '''
    Takes an image and labels, extracts features using a DINOv2 model,
    and returns the features of annotated pixels and their targets.
    If the image has 3 channels and RGB is chosen, extract features as usual.
    Otherwise extract features for each channel (triplicated for the 3 color channels)
    and concatenate them.
    INPUT:
        image (np.ndarray): image. Shape (H, W, C) or (H, W)
            Expects the image to be in the range [0, 1]; will normalize to ImageNet stats
        labels (np.ndarray): labels. Shape (H, W)
        dinov2_model (str): model to use for feature extraction.
            Options: 's', 'b', 'l', 'g', 's_r', 'b_r', 'l_r', 'g_r' (r = registers)
        rgb_if_possible (bool): whether to treat a 3-channel image as RGB or not
        interpolate_features (int): order of interpolation for reshaping features to image size
    OUTPUT:
        features_annot (np.ndarray): features of annotated pixels. Shape (n_annotated, F)
        targets (np.ndarray): targets of annotated pixels. Shape (n_annotated)
    '''
    feature_space = get_dinov2_feature_space(image, dinov2_model=dinov2_model,
                                             rgb_if_possible=rgb_if_possible,
                                             pc_as_features=pc_as_features,
                                             interpolate_features=interpolate_features)
    features_annot, targets = get_features_targets(feature_space, labels)
    return features_annot, targets
