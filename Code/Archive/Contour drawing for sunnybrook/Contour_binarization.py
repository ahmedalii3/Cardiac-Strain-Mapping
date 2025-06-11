import os
import pydicom
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.path import Path
def load_contour_points(contour_path):
    """
    Load contour points from a text file.

    Args:
    contour_path (str): Path to the contour file.

    Returns:
    tuple: Two numpy arrays representing x and y coordinates.
    """
    # Load contour points from text file assuming "x y" format per line
    contour_points = np.loadtxt(contour_path)
    print(contour_points)
    # Separate x and y coordinates
    x_points = contour_points[:, 0]
    y_points = contour_points[:, 1]
    x_points = np.append(x_points,x_points[0])
    y_points = np.append(y_points,y_points[0])
    return x_points-1, y_points-1

def dicom_image_size(dicom_paths):
    """
    Get the size of the DICOM images.

    Args:
    dicom_paths (list): List of paths to the DICOM files.

    Returns:
    tuple: Width and height of the DICOM images.
    """
    # Load the first DICOM image
    dicom_image = pydicom.dcmread(dicom_paths[0])
    
    # Get the shape of the image
    image_height, image_width = dicom_image.pixel_array.shape
    
    return image_width, image_height

def create_binary_mask_from_contour(x_points, y_points, image_width, image_height):
    """
    Create a binary mask from contour points.

    Args:
    x_points (numpy.ndarray): Array of x coordinates.
    y_points (numpy.ndarray): Array of y coordinates.
    image_width (int): Width of the image.
    image_height (int): Height of the image.

    Returns:
    numpy.ndarray: Binary mask with the contour points.
    """
    # Create a grid of coordinates
    x, y = np.meshgrid(np.arange(image_width), np.arange(image_height))
    x, y = x.flatten(), y.flatten()
    points = np.vstack((x, y)).T
    
    # Create a path from the contour points
    path = Path(np.column_stack((x_points, y_points)))
    
    # Create a binary mask
    mask = path.contains_points(points)
    mask = mask.reshape((image_height, image_width))
    
    return mask

def view_mask_on_image(image, mask, title):
    """
    View the binary mask overlaid on the image.

    Args:
    image (numpy.ndarray): Input image.
    mask (numpy.ndarray): Binary mask.
    title (str): Title of the plot.
    """
    # Create a copy of the image
    image_with_mask = np.copy(image)
    
    # Set mask values on the image
    image_with_mask[mask] = 0
    
    # Display the image with mask
    plt.figure(figsize=(8, 8))
    plt.imshow(image_with_mask, cmap='grey')
    plt.title(title)
    plt.axis('off')
    plt.show()

#try to load the image and the contour
dicom_path = '/Users/ahmed_ali/Documents/GitHub/GP-2025-Strain/Data/SunnyBrook/SCD_IMAGES_01/SCD0000301/CINESAX_5/IM-0003-0060.dcm'
contour_path = '/Users/ahmed_ali/Documents/GitHub/GP-2025-Strain/Data/SunnyBrook/SCD_ManualContours/SC-HF-I-04/contours-manual/IRCCI-expert/IM-0001-0060-icontour-manual.txt'
dicom_image = pydicom.dcmread(dicom_path)
image_data = dicom_image.pixel_array
x_points, y_points = load_contour_points(contour_path)
image_width, image_height = dicom_image_size([dicom_path])
mask = create_binary_mask_from_contour(x_points, y_points, image_width, image_height)
view_mask_on_image(image_data, mask, 'Contour Points on Image')
