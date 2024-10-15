import os
import pydicom
import numpy as np
import matplotlib.pyplot as plt

# Define contour colors for each type
contour_colors = {
    'icontour': 'red',  # Red for "icontour"
    'ocontour': 'green',  # Green for "ocontour"
    'other': 'blue'  # Blue for any other contour types
}

def load_dicom_image(image_path):
    """Load and return the DICOM image."""
    ds = pydicom.dcmread(image_path)
    return ds.pixel_array

def load_contour(contour_path):
    """Load contour from the text file."""
    contour = []
    with open(contour_path, 'r') as f:
        for line in f:
            x, y = line.strip().split()
            contour.append([float(x), float(y)])
    return np.array(contour)

def display_dicom_with_contours(image, contours, dicom_slice):
    """Display the original DICOM image and the DICOM image with all contours."""
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))

    # Display the original image
    axs[0].imshow(image, cmap='gray')
    axs[0].set_title(f'Slice {dicom_slice} - Original')
    axs[0].axis('off')

    # Display the image with contours
    axs[1].imshow(image, cmap='gray')
    for contour_file, contour in contours.items():
        if 'icontour' in contour_file:
            color = contour_colors['icontour']
        elif 'ocontour' in contour_file:
            color = contour_colors['ocontour']
        else:
            color = contour_colors['other']
        axs[1].plot(contour[:, 0], contour[:, 1], color=color, label=contour_file)
    axs[1].set_title(f'Slice {dicom_slice} - With Contours')
    axs[1].legend(loc='upper right')  # Show contour legend
    axs[1].axis('off')

    plt.show()

def process_patient_contours(image_dir, contour_dir):
    """Automate the process of drawing contours over corresponding DICOM images."""
    # Loop over all DICOM files
    for dicom_file in os.listdir(image_dir):
        if dicom_file.endswith('.dcm'):
            # Extract the last four digits from the DICOM filename (e.g., "0020" from "IM-0003-0020")
            dicom_id = dicom_file.split('-')[-1].split('.')[0]

            # Dictionary to hold contours of different types (e.g., 'icontour', 'ocontour')
            contours = {}
            for contour_file in os.listdir(contour_dir):
                if dicom_id in contour_file and contour_file.endswith('.txt'):
                    contour_path = os.path.join(contour_dir, contour_file)
                    contours[contour_file] = load_contour(contour_path)

            # If corresponding contours were found, load and display the image with the contours
            if contours:
                image_path = os.path.join(image_dir, dicom_file)
                image = load_dicom_image(image_path)
                display_dicom_with_contours(image, contours, dicom_id)

# Paths to the patient images and contour files
image_dir = '../../Data/SunnyBrook dataset/SCD0000301/CINESAX_5/'  # Path to the DICOM images
contour_dir = '../../Data/SunnyBrook dataset/SCD_ManualContours/SC-HF-I-04/contours-manual/IRCCI-expert/'  # Path to the contour files

# Process and display each image with its corresponding contours
process_patient_contours(image_dir, contour_dir)