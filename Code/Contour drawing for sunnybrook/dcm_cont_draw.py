import os
import pydicom
import numpy as np
import matplotlib.pyplot as plt

def load_dicom_image(dicom_path):
    """
    Load a DICOM image from the given path.

    Args:
    dicom_path (str): Path to the DICOM file.

    Returns:
    numpy.ndarray: 2D array representing the image.
    """
    # Read the DICOM file
    dicom_image = pydicom.dcmread(dicom_path)
    
    # Extract pixel array from the DICOM file
    image_data = dicom_image.pixel_array
    
    return image_data

def adjust_brightness(image, brightness_factor):
    """
    Adjust the brightness of the image.

    Args:
    image (numpy.ndarray): 2D array representing the image.
    brightness_factor (float): Factor to adjust brightness. 
                               (>1 increases brightness, <1 decreases brightness)

    Returns:
    numpy.ndarray: Brightness adjusted image.
    """
    # Adjust the brightness by multiplying the image data by the brightness factor
    adjusted_image = image * brightness_factor
    
    # Clip values to the valid range (0 to maximum value for image data type)
    adjusted_image = np.clip(adjusted_image, 0, np.max(image))
    
    return adjusted_image

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
    return x_points, y_points

def plot_dicom_with_contours(dicom_image, inner_x, inner_y, outer_x, outer_y):
    """
    Plot the DICOM image with the overlayed inner and outer contours.

    Args:
    dicom_image (numpy.ndarray): 2D array representing the DICOM image.
    inner_x (numpy.ndarray): X coordinates of the inner contour.
    inner_y (numpy.ndarray): Y coordinates of the inner contour.
    outer_x (numpy.ndarray): X coordinates of the outer contour.
    outer_y (numpy.ndarray): Y coordinates of the outer contour.
    """
    plt.figure(figsize=(8, 8))
    plt.imshow(dicom_image, cmap='gray')
    
    # Overlay the inner contour in red
    plt.plot(inner_x - 1,inner_y - 1, 'r-', linewidth=2, label='Inner Contour')
    
    # Overlay the outer contour in blue
    plt.plot(outer_x , outer_y , 'b-', linewidth=2, label='Outer Contour')
    
    plt.title('DICOM Image with Inner and Outer Contour Overlay')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.gca().invert_yaxis()  # Invert y-axis to match image coordinates
    plt.legend()
    plt.show()

# Define the paths to the DICOM and contour files
dicom_path = '/Users/ahmed_ali/Documents/GitHub/GP-2025-Strain/Data/SunnyBrook/SCD_IMAGES_01/SCD0000301/CINESAX_5/IM-0003-0060.dcm'  # Update this path to your DICOM file
inner_contour_path = '/Users/ahmed_ali/Documents/GitHub/GP-2025-Strain/Data/SunnyBrook/SCD_ManualContours/SC-HF-I-04/contours-manual/IRCCI-expert/IM-0001-0060-p1contour-manual.txt'
outer_contour_path = '/Users/ahmed_ali/Documents/GitHub/GP-2025-Strain/Data/SunnyBrook/SCD_ManualContours/SC-HF-I-04/contours-manual/IRCCI-expert/IM-0001-0060-ocontour-manual.txt'

# Load the DICOM image
dicom_image = load_dicom_image(dicom_path)

# Adjust brightness (change brightness_factor as needed)
brightness_factor = 1.5  # Increase brightness by 50%
dicom_image = adjust_brightness(dicom_image, brightness_factor)

# Load the inner contour points
inner_x, inner_y = load_contour_points(inner_contour_path)

# Load the outer contour points
outer_x, outer_y = load_contour_points(outer_contour_path)

# Plot the DICOM image with the inner and outer contour overlay
plot_dicom_with_contours(dicom_image, inner_x, inner_y, outer_x, outer_y)
