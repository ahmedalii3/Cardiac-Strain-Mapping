import numpy as np
import SimpleITK as sitk
import scipy.interpolate as spi
from scipy.ndimage import zoom
import os

def monotonic_zoom_interpolate(image_np, resize_factor):
    result = image_np.copy()
    for axis, factor in enumerate(resize_factor[::-1]):
        new_length = int(result.shape[axis] * factor)
        x_old = np.arange(result.shape[axis])
        x_new = np.linspace(0, result.shape[axis] - 1, new_length)
        if len(x_old) > 1:
            pchip_interp = spi.PchipInterpolator(x_old, result.take(indices=x_old, axis=axis), axis=axis)
            result = pchip_interp(x_new)
        else:
            result = np.repeat(result, new_length, axis=axis)
    return result

def preprocess_series_in_memory(series_data, target_resolution):
    if 'PixelData' in series_data:
        print("PixelData is in series_data")
        
        # Load the DICOM pixel data into a numpy array
        image_np = series_data.pixel_array[np.newaxis, ...]
        print(f"Original image shape: {image_np.shape}")
        # dicom_data_3d = series_data.pixel_array  # shape: (1, 256, 216)


        # Convert the numpy array to a SimpleITK image
        image = sitk.GetImageFromArray(image_np)
        
        # Get the current pixel spacing from the DICOM metadata
        current_spacing = np.array([float(series_data.PixelSpacing[0]), float(series_data.PixelSpacing[1]), float(series_data.SliceThickness)])
        print(f"Current spacing: {current_spacing}")

        # Calculate the resize factor based on the target resolution and current spacing
        resize_factor = np.array([current_spacing[0] / target_resolution[0], 
                                  current_spacing[1] / target_resolution[1], 
                                  1.0])
        print(f"Resize factor: {resize_factor}")

        # Calculate the new image shape after resizing
        current_size = np.array(image.GetSize())
        print(f"Current size (image.GetSize()): {current_size}")
        
        new_real_shape = current_size * resize_factor
        print(f"New real shape (before rounding): {new_real_shape}")

        # Round the new shape to integer values
        new_shape = np.round(new_real_shape).astype(int)
        print(f"New shape (rounded): {new_shape}")

        # Calculate the real resize factor to be applied
        real_resize_factor = new_shape / current_size
        print(f"Real resize factor: {real_resize_factor}")

        # Resample the image using the zoom interpolation
        image_resampled_np = monotonic_zoom_interpolate(image_np, real_resize_factor)
        print(f"Image resampled, shape after resampling: {image_resampled_np.shape}")
        # plot the resampled image

        # Convert the resampled numpy array back to a SimpleITK image
        image_resampled = sitk.GetImageFromArray(image_resampled_np)

        # Set the new spacing for the resampled image
        new_spacing = np.array([target_resolution[0], target_resolution[1], current_spacing[2]])
        image_resampled.SetSpacing(new_spacing)
        print(f"New spacing set: {new_spacing}")

        return image_resampled
    else:
        raise ValueError("No PixelData found in the provided DICOM data.")


def normalize_series_in_memory(image):
    image_np = sitk.GetArrayFromImage(image)
    print(f"Normalizing image with shape: {image_np.shape}")
    
    # Normalize the image by subtracting the mean and dividing by the standard deviation
    mean = image_np.mean()
    std = image_np.std()
    print(f"Image mean: {mean}, Image std: {std}")
    
    image_np = (image_np - mean) / std
    image = sitk.GetImageFromArray(image_np)
    return image

def pad_series_in_memory(image, target_resolution):
    constant_val = int(sitk.GetArrayFromImage(image).min())
    print(f"Padding image with constant value: {constant_val}")

    current_size = np.array(image.GetSize())
    print(f"Current image size (before padding): {current_size}")

    # Calculate padding for left, right, top, and bottom
    padding_left_right = target_resolution[0] - current_size[0]
    padding_top_bottom = target_resolution[1] - current_size[1]
    padding_left = int(padding_left_right // 2)
    padding_right = int(padding_left_right - padding_left)
    padding_top = int(padding_top_bottom // 2)
    padding_bottom = int(padding_top_bottom - padding_top)

    print(f"Padding values: left={padding_left}, right={padding_right}, top={padding_top}, bottom={padding_bottom}")

    # Perform padding
    transformed = sitk.ConstantPad(image, (padding_left, padding_top, 0), 
                                   (padding_right, padding_bottom, 0), constant_val)
    print("Image padded.")
    return transformed
