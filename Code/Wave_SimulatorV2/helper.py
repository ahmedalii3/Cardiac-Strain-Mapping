import numpy as np
import cv2
from scipy.ndimage import distance_transform_edt, grey_closing, binary_closing, gaussian_filter
import json
from typing import Union,Any
import os


def load_image(path,isMask=False):
        # Load the image array from the .npy file
        array = np.load(path)
        if isMask:
             array = array[1]
             array = np.where(array > 1, 1, 0)
        else:
             array = array[0]
        
        # Extract the first image if the array has extra dimensions
        if len(array.shape) > 3 or (len(array.shape) == 3 and array.shape[0] not in [1, 3]):
            image_array = array[0]  # Adjust indexing as needed for your specific data
        else:
            image_array = array

        # Ensure the array is in uint8 format (0-255 range) for proper visualization
        if image_array.dtype != np.uint8:
            image_array = cv2.normalize(image_array, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Handle grayscale or RGB format
        if len(image_array.shape) == 2:  # Grayscale
            image = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)  # Convert to RGB for consistency
        elif len(image_array.shape) == 3 and image_array.shape[2] in [3, 4]:  # RGB or RGBA
            image = image_array[:, :, :3]  # Discard alpha if present
        else:
            raise ValueError("Unexpected image shape, unable to load the image properly.")

        return image

def dilate_mask(mask):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    value = 0.9
    dilated_mask = mask
    process_mask = mask
    for i in range(10):            
        old_process_mask = process_mask
        process_mask = cv2.dilate(process_mask, kernel)

        # Identify the newly added pixels
        added_region = (process_mask - old_process_mask).astype(np.float64)

        # Update the dilated image
        dilated_mask = dilated_mask + added_region * value
        value -= 0.1
    dilated_mask = gaussian_filter(dilated_mask, sigma=2)
    return dilated_mask




def dilate_mask_fade(mask, decay_distance=20, binary_closing_size=5, grey_closing_size=5):
    """
    Create a smooth fading mask based on distance from the cleaned binary mask boundary,
    applying binary closing first, and grayscale closing only outside the mask.

    Args:
        mask (np.ndarray): 2D binary mask (1 inside, 0 outside).
        decay_distance (float): Distance (in pixels) over which the mask fades to zero.
        binary_closing_size (int): Size for binary morphological closing (first step).
        grey_closing_size (int): Size for grayscale morphological closing (only on fading region).

    Returns:
        smooth_mask (np.ndarray): Smooth fading mask (values 1 inside, fades to 0 outside).
    """

    # Step 1: Ensure binary mask
    binary_mask = (mask > 0).astype(np.uint8)

    # Step 2: Apply binary closing first to smooth the mask
    binary_structure = np.ones((binary_closing_size, binary_closing_size))
    cleaned_binary_mask = binary_closing(binary_mask, structure=binary_structure).astype(np.uint8)

    # Step 3: Compute distance transform outside the cleaned mask
    distance_outside = distance_transform_edt(1 - cleaned_binary_mask)

    # Step 4: Normalize distances to [1, 0] range for fading
    smooth_mask = np.clip(1 - distance_outside / decay_distance, 0, 1)

    # Step 5: Apply grayscale closing only outside the mask
#    mask_inside = cleaned_binary_mask == 1
#    mask_outside = cleaned_binary_mask == 0

    smooth_mask_final = grey_closing(smooth_mask, structure=np.ones((grey_closing_size, grey_closing_size)))

    # Step 6: Combine: inside stays 1, outside is smoothed fading
    #smooth_mask_final = smooth_mask.copy()
    #smooth_mask_final[mask_outside] = smooth_mask_outside[mask_outside]
    #smooth_mask_final[mask_inside] = 1  # Enforce inside = 1 exactly

    return smooth_mask_final




def dilate_mask_fade_cosine(mask, decay_distance=20, pre_blur_sigma=1.0, post_blur_sigma=1.0):
    """
    Create a smooth fading mask using a cosine decay, based on distance from the mask boundary.

    Args:
        mask (np.ndarray): 2D binary mask (1 inside, 0 outside).
        decay_distance (float): Distance (in pixels) over which the mask fades to zero.
        pre_blur_sigma (float): Gaussian blur before distance transform.
        post_blur_sigma (float): Gaussian blur after fading.

    Returns:
        smooth_mask (np.ndarray): Smooth fading mask (values 1 inside, smoothly fades to 0 outside).
    """
    binary_mask = (mask > 0).astype(np.uint8)
    binary_mask_blurred = gaussian_filter(binary_mask.astype(float), sigma=pre_blur_sigma)
    binary_mask_blurred = (binary_mask_blurred > 0.5).astype(np.uint8)
    
    distance_outside = distance_transform_edt(1 - binary_mask_blurred)

    # Cosine fade
    fade = 0.5 * (1 + np.cos(np.pi * np.clip(distance_outside / decay_distance, 0, 1)))

    smooth_mask = fade * (distance_outside <= decay_distance)
    smooth_mask[binary_mask_blurred == 1] = 1  # Inside original mask stays 1

    # Final optional smoothing
    smooth_mask = gaussian_filter(smooth_mask, sigma=post_blur_sigma)

    return smooth_mask

def dilate_mask_fade_smooth(mask, decay_distance=20, pre_blur_sigma=1.0, post_blur_sigma=1.0):
    """
    Create a smooth fading mask based on distance from the binary mask boundary, 
    avoiding artifacts.

    Args:
        mask (np.ndarray): 2D binary mask (1 inside, 0 outside).
        decay_distance (float): Distance (in pixels) over which the mask fades to zero.
        pre_blur_sigma (float): Amount of Gaussian blur before distance transform (default 1.0).
        post_blur_sigma (float): Amount of Gaussian blur after fading map (default 1.0).

    Returns:
        smooth_mask (np.ndarray): Smooth fading mask (values 1 inside, fades to 0 outside).
    """

    # Ensure binary mask
    binary_mask = (mask > 0).astype(np.uint8)

    # Step 1: Smooth the mask before distance to soften sharp boundaries
    binary_mask_blurred = gaussian_filter(binary_mask.astype(float), sigma=pre_blur_sigma)

    # Step 2: Threshold again slightly (make it a bit more conservative)
    binary_mask_blurred = (binary_mask_blurred > 0.5).astype(np.uint8)

    # Step 3: Compute distance outside the mask
    distance_outside = distance_transform_edt(1 - binary_mask_blurred)

    # Step 4: Normalize distances
    smooth_mask = np.clip(1 - distance_outside / decay_distance, 0, 1)

    # Step 5: Final smoothing (optional but helps remove rays)
    smooth_mask = gaussian_filter(smooth_mask, sigma=post_blur_sigma)

    return smooth_mask
def save_if_not_exists(file_paths):
    """Check if any of the files exist"""
    for path in file_paths:
        if os.path.exists(path + '.npy'):
            return False
    return True


def numpy_to_serializable(obj: Union[np.ndarray, Any]) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64, np.int32, np.int64)):
        return obj.item()
    elif isinstance(obj, dict):
        return {k: numpy_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [numpy_to_serializable(v) for v in obj]
    else:
        return obj

def save_json_array(array: np.ndarray, filepath: str) -> None:
    with open(filepath, 'w') as f:
        json.dump(numpy_to_serializable(array), f, indent=4)



def numpy_to_serializable(obj: Union[np.ndarray, Any]) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64, np.int32, np.int64)):
        return obj.item()
    elif isinstance(obj, dict):
        return {k: numpy_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [numpy_to_serializable(v) for v in obj]
    else:
        return obj

def save_json_array(array: np.ndarray, filepath: str) -> None:
    with open(filepath, 'w') as f:
        json.dump(numpy_to_serializable(array), f, indent=4)



def numpy_to_serializable(obj: Union[np.ndarray, Any]) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64, np.int32, np.int64)):
        return obj.item()
    elif isinstance(obj, dict):
        return {k: numpy_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [numpy_to_serializable(v) for v in obj]
    else:
        return obj

def save_json_array(array: np.ndarray, filepath: str) -> None:
    with open(filepath, 'w') as f:
        json.dump(numpy_to_serializable(array), f, indent=4)
