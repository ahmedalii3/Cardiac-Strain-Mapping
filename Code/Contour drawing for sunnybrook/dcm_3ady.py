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

    # Separate x and y coordinates
    x_points = contour_points[:, 0]
    y_points = contour_points[:, 1]

    return x_points, y_points

def plot_dicom_with_contour(dicom_image, x_points, y_points):
    """
    Plot the DICOM image with the overlayed contour.

    Args:
    dicom_image (numpy.ndarray): 2D array representing the DICOM image.
    x_points (numpy.ndarray): X coordinates of the contour.
    y_points (numpy.ndarray): Y coordinates of the contour.
    """
    plt.figure(figsize=(8, 8))
    plt.imshow(dicom_image, cmap='gray')
    plt.plot(x_points, y_points, 'r-', linewidth=2)  # Overlay the contour in red
    plt.title('DICOM Image with Contour Overlay')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.gca().invert_yaxis()  # Invert y-axis to match image coordinates
    plt.show()

# Define the paths to the DICOM and contour files
dicom_path = 'Data/SunnyBrook/SCD_IMAGES_01/SCD0000101/CINESAX_300/IM-0003-0059.dcm'  # Update this path to your DICOM file
contour_path = 'Data/SunnyBrook/SCD_ManualContours/SC-HF-I-01/contours-manual/IRCCI-expert/IM-0001-0059-icontour-manual.txt'  # Update this path to your contour file

# Load the DICOM image
dicom_image = load_dicom_image(dicom_path)

# Load the contour points
x_points, y_points = load_contour_points(contour_path)

# Plot the DICOM image with the contour overlay
plot_dicom_with_contour(dicom_image, x_points, y_points)
