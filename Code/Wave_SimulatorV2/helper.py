import os
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
import json
from typing import Any, Union
import matplotlib.pyplot as plt

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
    value = 0.7
    dilated_mask = mask
    process_mask = mask
<<<<<<< Updated upstream
<<<<<<< Updated upstream
    for i in range(8):            
=======
    for i in range(5):        
>>>>>>> Stashed changes
=======
    for i in range(5):        
>>>>>>> Stashed changes
        old_process_mask = process_mask
        process_mask = cv2.dilate(process_mask, kernel)

        # Identify the newly added pixels
        added_region = (process_mask - old_process_mask).astype(np.float64)

        # Update the dilated image
        dilated_mask = dilated_mask + added_region * value
        value *= 0.6
    dilated_mask = gaussian_filter(dilated_mask, sigma=2)
    plt.imshow(dilated_mask)
    plt.title('Dilated Mask')
    plt.axis('off')
    plt.show()
    return dilated_mask

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
