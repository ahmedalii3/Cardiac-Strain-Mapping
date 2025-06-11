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
    dicom_image = pydicom.dcmread(dicom_path)
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
    contour_points = np.loadtxt(contour_path)
    x_points = contour_points[:, 0]
    y_points = contour_points[:, 1]
    return x_points, y_points

def rotate_points(x_points, y_points, angle_degrees):
    """
    Rotate the contour points around their center by a given angle.

    Args:
    x_points (numpy.ndarray): X coordinates of the contour.
    y_points (numpy.ndarray): Y coordinates of the contour.
    angle_degrees (float): The rotation angle in degrees.

    Returns:
    tuple: Two numpy arrays of the rotated x and y coordinates.
    """
    # Calculate the center of the contour points
    cx = np.mean(x_points)
    cy = np.mean(y_points)

    # Convert the angle to radians
    angle_radians = np.radians(angle_degrees)

    # Perform the rotation
    x_rotated = cx + (x_points - cx) * np.cos(angle_radians) - (y_points - cy) * np.sin(angle_radians)
    y_rotated = cy + (x_points - cx) * np.sin(angle_radians) + (y_points - cy) * np.cos(angle_radians)

    return x_rotated, y_rotated

def plot_dicom_with_contours(dicom_image, inner_contour_points, outer_contour_points):
    """
    Plot the DICOM image with the overlayed inner and outer contours.

    Args:
    dicom_image (numpy.ndarray): 2D array representing the DICOM image.
    inner_contour_points (tuple): x and y coordinates of the inner contour.
    outer_contour_points (tuple): x and y coordinates of the outer contour.
    """
    plt.figure(figsize=(8, 8))
    plt.imshow(dicom_image, cmap='gray')

    # Plot inner contour in red
    plt.plot(inner_contour_points[0], inner_contour_points[1], 'r-', linewidth=2, label='Inner Contour (Endocardium)')

    # Plot outer contour in blue
    plt.plot(outer_contour_points[0], outer_contour_points[1], 'b-', linewidth=2, label='Outer Contour (Epicardium)')

    plt.title('DICOM Image with Inner and Outer Contours')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend(loc='upper right')
    plt.gca().invert_yaxis()  # Invert y-axis to match image coordinates
    plt.show()

# Define the paths to the DICOM and contour files
dicom_path = 'Data/SunnyBrook/SCD_IMAGES_01/SCD0000101/CINESAX_300/IM-0003-0048.dcm'
inner_contour_path = 'Data/SunnyBrook/SCD_ManualContours/SC-HF-I-01/contours-manual/IRCCI-expert/IM-0001-0048-icontour-manual.txt'
outer_contour_path = 'Data/SunnyBrook/SCD_ManualContours/SC-HF-I-01/contours-manual/IRCCI-expert/IM-0001-0059-ocontour-manual.txt'

# Load the DICOM image
dicom_image = load_dicom_image(dicom_path)

# Load the inner and outer contour points
inner_x_points, inner_y_points = load_contour_points(inner_contour_path)
outer_x_points, outer_y_points = load_contour_points(outer_contour_path)

# Rotate the contours (you can change the angle)
rotation_angle = 0  # Rotate by 45 degrees
inner_x_rotated, inner_y_rotated = rotate_points(inner_x_points, inner_y_points, rotation_angle)
outer_x_rotated, outer_y_rotated = rotate_points(outer_x_points, outer_y_points, rotation_angle)

# Plot the DICOM image with the rotated contours
plot_dicom_with_contours(dicom_image, (inner_x_rotated, inner_y_rotated), (outer_x_rotated, outer_y_rotated))
