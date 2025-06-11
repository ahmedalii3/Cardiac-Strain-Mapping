import os
import pydicom
import numpy as np
import matplotlib.pyplot as plt

# Define contour colors for each type
contour_colors = {
    'icontour': 'red',  # Red for "icontour"
    'ocontour': 'green',  # Green for "ocontour"
    'papillary muscles': 'blue'  # Blue for any other contour types
}


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


def load_contour(contour_path):
    """Load contour from the text file."""
    contour = []
    with open(contour_path, 'r') as f:
        for line in f:
            x, y = line.strip().split()
            contour.append([float(x), float(y)])

    return np.array(contour)


def display_dicom_with_contours(dicom_image, contours, dicom_slice, output_dir):
    """Display the original DICOM image and the DICOM image with all contours."""
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))

    # Display the original image
    axs[0].imshow(dicom_image, cmap='gray')
    axs[0].set_title(f'Slice {dicom_slice} - Original')
    axs[0].axis('off')

    # Display the image with contours
    axs[1].imshow(dicom_image, cmap='gray')
    for contour_file, contour in contours.items():
        if 'icontour' in contour_file:
            color = contour_colors['icontour']
        elif 'ocontour' in contour_file:
            color = contour_colors['ocontour']
        else:
            color = contour_colors['papillary muscles']

        axs[1].plot(contour[:, 0] - 1, contour[:, 1] - 1, color=color, label=contour_file)

    axs[1].set_title(f'Slice {dicom_slice} - With Contours')
    axs[1].legend(loc='upper right')  # Show contour legend
    axs[1].axis('off')

    # plt.show()

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save the plot
    plt.savefig(os.path.join(output_dir, f'Slice_{dicom_slice}.png'))
    plt.close()

def process_patient_contours(image_dir, contour_dir, output_dir):
    """Automate the process of drawing contours over corresponding DICOM images."""
    # Loop over all DICOM files
    for dicom_file in os.listdir(image_dir):
        if dicom_file.endswith('.dcm'):
            # Extract the last four digits from the DICOM filename (e.g., "0020" from "IM-0003-0020")
            dicom_id = dicom_file.split('-')[-1].split('.')[0]

            # Dictionary to hold contours of different types (e.g., 'icontour', 'ocontour')
            contours = {}
            for contour_file in os.listdir(contour_dir):
                # Extract the last four digits from the contour filename (e.g., "0048" from "IM-0001-0048-icontour-manual.txt")
                contour_id = contour_file.split('-')[-3]
                if dicom_id == contour_id and contour_file.endswith('.txt'):
                    contour_path = os.path.join(contour_dir, contour_file)
                    contours[contour_file] = load_contour(contour_path)

            # If corresponding contours were found, load and display the image with the contours
            if contours:
                image_path = os.path.join(image_dir, dicom_file)
                dicom_image = load_dicom_image(image_path)

                # Adjust brightness (change brightness_factor as needed)
                brightness_factor = 1.5  # Increase brightness by 50%
                dicom_image = adjust_brightness(dicom_image, brightness_factor)

                display_dicom_with_contours(dicom_image, contours, dicom_id, output_dir)

# Paths to the patient images and contour files
image_dir = '../../Data/SunnyBrook/SCD0004501/CINESAX_1100/'  # Path to the DICOM images
contour_dir = '../../Data/SunnyBrook/SCD_ManualContours/SC-N-40/contours-manual/IRCCI-expert/'  # Path to the contour files
output_dir = 'Patient 45/Patient 45 Contour drawn (with x - 1 , y - 1 )/'  # Output directory for the images with contours
# Process and display each image with its corresponding contours
process_patient_contours(image_dir, contour_dir, output_dir)