
import numpy as np
import matplotlib.pyplot as plt
import pydicom

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

def mirror_points_around_line(x_points, y_points, line_angle_degrees):
    """
    Mirror the contour points around a line passing through the center at a given angle.

    Args:
    x_points (numpy.ndarray): X coordinates of the contour.
    y_points (numpy.ndarray): Y coordinates of the contour.
    line_angle_degrees (float): The angle of the line in degrees (135 degrees in this case).

    Returns:
    tuple: Two numpy arrays of the mirrored x and y coordinates.
    """
    # Calculate the center of the contour points
    cx = np.mean(x_points)
    cy = np.mean(y_points)

    # Translate points to make the center (cx, cy) the origin
    x_translated = x_points - cx
    y_translated = y_points - cy

    # Convert the angle to radians
    angle_radians = np.radians(line_angle_degrees)

    # Rotate the points to align the line with the x-axis
    cos_angle = np.cos(angle_radians)
    sin_angle = np.sin(angle_radians)

    x_rotated = x_translated * cos_angle + y_translated * sin_angle
    y_rotated = -x_translated * sin_angle + y_translated * cos_angle

    # Mirror the points across the x-axis (y = 0 after rotation)
    y_mirrored = -y_rotated

    # Rotate the points back to the original angle
    x_mirrored_rotated_back = x_rotated * cos_angle - y_mirrored * sin_angle
    y_mirrored_rotated_back = x_rotated * sin_angle + y_mirrored * cos_angle

    # Translate the points back to the original center
    x_mirrored = x_mirrored_rotated_back + cx
    y_mirrored = y_mirrored_rotated_back + cy

    return x_mirrored, y_mirrored

def plot_dicom_with_contours(dicom_image, original_contour_points, mirrored_contour_points):
    """
    Plot the DICOM image with the original and mirrored contours.

    Args:
    dicom_image (numpy.ndarray): 2D array representing the DICOM image.
    original_contour_points (tuple): x and y coordinates of the original contour.
    mirrored_contour_points (tuple): x and y coordinates of the mirrored contour.
    """
    plt.figure(figsize=(8, 8))
    plt.imshow(dicom_image, cmap='gray')

    # Plot original contour in red
    plt.plot(original_contour_points[0], original_contour_points[1], 'r-', linewidth=2, label='Original Contour')

    # Plot mirrored contour in blue
    plt.plot(mirrored_contour_points[0], mirrored_contour_points[1], 'b--', linewidth=2, label='Mirrored Contour')

    plt.title('DICOM Image with Original and Mirrored Contours')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend(loc='upper right')
    plt.gca().invert_yaxis()  # Invert y-axis to match image coordinates
    plt.show()

# Define the paths to the DICOM and contour files
dicom_path = 'Data/SunnyBrook/SCD_IMAGES_01/SCD0000101/CINESAX_300/IM-0003-0048.dcm'
contour_path = 'Data/SunnyBrook/SCD_ManualContours/SC-HF-I-01/contours-manual/IRCCI-expert/IM-0001-0048-icontour-manual.txt'

# Load the DICOM image
dicom_image = load_dicom_image(dicom_path)

# Load the contour points
x_points, y_points = load_contour_points(contour_path)

# Mirror the contours around the 135-degree line
line_angle = 135  # Line makes 135 degrees with the x-axis
x_mirrored, y_mirrored = mirror_points_around_line(x_points, y_points, line_angle)

# Plot the DICOM image with the original and mirrored contours
plot_dicom_with_contours(dicom_image, (x_points, y_points), (x_mirrored, y_mirrored))
