import os
from typing import List
import numpy as np
import cv2
from scipy.ndimage import (
    distance_transform_edt,
    grey_closing,
    binary_closing,
    gaussian_filter
)


def load_image(path: str, is_mask: bool = False) -> np.ndarray:
    """
    Load and preprocess an image from a .npy file.

    Args:
        path (str): Path to the .npy file containing the image data
        is_mask (bool, optional): Whether the loaded data is a mask. Defaults to False.

    Returns:
        np.ndarray: Processed image in RGB format (uint8)

    Raises:
        ValueError: If the image shape is unexpected
        FileNotFoundError: If the .npy file doesn't exist
    """
    try:
        array = np.load(path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find file: {path}")

    # Extract mask or image data
    array = array[1] if is_mask else array[0]
    if is_mask:
        array = np.where(array > 1, 1, 0)

    # Handle multi-dimensional arrays
    if len(array.shape) > 3 or (len(array.shape) == 3 and array.shape[0] not in [1, 3]):
        image_array = array[0]
    else:
        image_array = array

    # Normalize to uint8
    if image_array.dtype != np.uint8:
        image_array = cv2.normalize(image_array, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Convert to RGB
    if len(image_array.shape) == 2:
        return cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
    elif len(image_array.shape) == 3 and image_array.shape[2] in [3, 4]:
        return image_array[:, :, :3]
    
    raise ValueError(f"Unexpected image shape: {image_array.shape}")


def dilate_mask(mask: np.ndarray, iterations: int = 10, kernel_size: int = 7) -> np.ndarray:
    """
    Create a gradually fading dilated mask using morphological operations.

    Args:
        mask (np.ndarray): Input binary mask
        iterations (int, optional): Number of dilation iterations. Defaults to 10.
        kernel_size (int, optional): Size of the elliptical kernel. Defaults to 7.

    Returns:
        np.ndarray: Smoothly faded dilated mask
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    dilated_mask = mask.copy()
    process_mask = mask.copy()
    
    for i in range(iterations):
        old_process_mask = process_mask.copy()
        process_mask = cv2.dilate(process_mask, kernel)
        
        # Update dilated mask with fading value
        value = 0.9 - (i * 0.1)
        added_region = (process_mask - old_process_mask).astype(np.float64)
        dilated_mask = dilated_mask + added_region * value

    return gaussian_filter(dilated_mask, sigma=2)


def dilate_mask_fade(
    mask: np.ndarray,
    decay_distance: float = 20,
    binary_closing_size: int = 5,
    grey_closing_size: int = 5
) -> np.ndarray:
    """
    Create a smooth fading mask using distance transform and morphological operations.

    Args:
        mask (np.ndarray): Input binary mask
        decay_distance (float, optional): Distance over which the mask fades. Defaults to 20.
        binary_closing_size (int, optional): Size for binary closing. Defaults to 5.
        grey_closing_size (int, optional): Size for grayscale closing. Defaults to 5.

    Returns:
        np.ndarray: Smooth fading mask
    """
    binary_mask = (mask > 0).astype(np.uint8)
    binary_structure = np.ones((binary_closing_size, binary_closing_size))
    
    # Clean and smooth the mask
    cleaned_mask = binary_closing(binary_mask, structure=binary_structure).astype(np.uint8)
    distance_outside = distance_transform_edt(1 - cleaned_mask)
    
    # Create smooth transition
    smooth_mask = np.clip(1 - distance_outside / decay_distance, 0, 1)
    return grey_closing(smooth_mask, structure=np.ones((grey_closing_size, grey_closing_size)))


def dilate_mask_fade_cosine(
    mask: np.ndarray,
    decay_distance: float = 20,
    pre_blur_sigma: float = 1.0,
    post_blur_sigma: float = 1.0
) -> np.ndarray:
    """
    Create a smooth fading mask using cosine decay.

    Args:
        mask (np.ndarray): Input binary mask
        decay_distance (float, optional): Distance over which the mask fades. Defaults to 20.
        pre_blur_sigma (float, optional): Blur before distance transform. Defaults to 1.0.
        post_blur_sigma (float, optional): Blur after fading. Defaults to 1.0.

    Returns:
        np.ndarray: Smooth fading mask with cosine decay
    """
    binary_mask = (mask > 0).astype(np.uint8)
    binary_mask_blurred = gaussian_filter(binary_mask.astype(float), sigma=pre_blur_sigma)
    binary_mask_blurred = (binary_mask_blurred > 0.5).astype(np.uint8)
    
    distance_outside = distance_transform_edt(1 - binary_mask_blurred)
    fade = 0.5 * (1 + np.cos(np.pi * np.clip(distance_outside / decay_distance, 0, 1)))
    
    smooth_mask = fade * (distance_outside <= decay_distance)
    smooth_mask[binary_mask_blurred == 1] = 1
    
    return gaussian_filter(smooth_mask, sigma=post_blur_sigma)


def dilate_mask_fade_smooth(
    mask: np.ndarray,
    decay_distance: float = 20,
    pre_blur_sigma: float = 1.0,
    post_blur_sigma: float = 1.0
) -> np.ndarray:
    """
    Create a smooth fading mask with pre and post blurring.

    Args:
        mask (np.ndarray): Input binary mask
        decay_distance (float, optional): Distance over which the mask fades. Defaults to 20.
        pre_blur_sigma (float, optional): Blur before distance transform. Defaults to 1.0.
        post_blur_sigma (float, optional): Blur after fading. Defaults to 1.0.

    Returns:
        np.ndarray: Smooth fading mask
    """
    binary_mask = (mask > 0).astype(np.uint8)
    binary_mask_blurred = gaussian_filter(binary_mask.astype(float), sigma=pre_blur_sigma)
    binary_mask_blurred = (binary_mask_blurred > 0.5).astype(np.uint8)
    
    distance_outside = distance_transform_edt(1 - binary_mask_blurred)
    smooth_mask = np.clip(1 - distance_outside / decay_distance, 0, 1)
    
    return gaussian_filter(smooth_mask, sigma=post_blur_sigma)


def save_if_not_exists(file_paths: List[str]) -> bool:
    """
    Check if any of the specified files exist.

    Args:
        file_paths (List[str]): List of file paths to check

    Returns:
        bool: True if none of the files exist, False otherwise
    """
    return all(not os.path.exists(f"{path}.npy") for path in file_paths)