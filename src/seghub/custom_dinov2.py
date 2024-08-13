from seghub import dino_utils
import numpy as np
import torch
import dinov2.models.vision_transformer as vits
from seghub.util_funcs import pad_to_patch, reshape_patches_to_img, calculate_padding


def load_model(teacher_pth_path, device="cuda", load_weights=True):
    # Read in and process the teacher model state_dict
    state_dict = torch.load(teacher_pth_path, map_location="cpu")["teacher"]
    state_dict = {key: val for key, val in state_dict.items() if "backbone" in key}
    state_dict = {key.replace("backbone.", ""): val for key, val in state_dict.items()}
    model = guess_model(state_dict)
    # Allows to load an empty model with the same architecture as the teacher
    if load_weights:
        model.load_state_dict(state_dict, strict=True)
    model.eval()
    model = model.to(device)
    return model

def guess_model(state_dict):
    patch_w = state_dict['patch_embed.proj.weight'].shape[-1]
    global_crops_size = int(np.sqrt(state_dict['pos_embed'].shape[1]-1)) * patch_w
    embed_size = state_dict['pos_embed'].shape[-1]
    embed_arch_dict = {384: "vit_small", 768: "vit_base", 1024: "vit_large", 1536: "vit_giant2"}
    arch = embed_arch_dict[embed_size]
    vit_kwargs = dict(
        patch_size=patch_w,
        img_size=global_crops_size, # needs to be adjusted to the global_crop_size value
        init_values=1.0,
        # ffn_layer="mlp",
        block_chunks=4,
        # num_register_tokens=0,
        # interpolate_antialias=False,
        # interpolate_offset=0.1,
    )
    model = vits.__dict__[arch](**vit_kwargs)
    return model

def extract_features(image, model, device="cuda"):
    image_batch = dino_utils.preprocess_for_dinov2(image)
    image_batch = image_batch.to(device)
    with torch.no_grad():
        features_dict = model.forward_features(image_batch)
        features = features_dict['x_norm_patchtokens']
    # Convert to numpy array
    features = features.cpu().numpy()
    # Remove batch dimension
    features = features[0]
    return features

def get_custom_dinov2_feature_space(image, teacher_pth_path,
                                    device="cuda", interpolate_features=False,
                                    load_weights=True):
    model = load_model(teacher_pth_path, device, load_weights=load_weights)
    patch_w = model.patch_size
    patch_size=(patch_w, patch_w)
    padded_image = pad_to_patch(image, "bottom", "right", patch_size=patch_size)
    patch_features_flat = extract_features(padded_image, model, device)
    # Recreate an image-sized feature space from the features
    vertical_pad, horizontal_pad = calculate_padding(image.shape[:2], patch_size=patch_size)
    padded_img_shape = (image.shape[0] + vertical_pad, image.shape[1] + horizontal_pad)
    feature_space = reshape_patches_to_img(patch_features_flat, padded_img_shape,
                                           patch_size=patch_size,
                                           interpolation_order=interpolate_features)
    feature_space_recrop = feature_space[:image.shape[0], :image.shape[1]]
    return feature_space_recrop
