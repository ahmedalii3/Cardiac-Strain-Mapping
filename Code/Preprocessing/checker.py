import logging
import os
import SimpleITK as sitk

logging.getLogger().setLevel(logging.INFO) # Allow logging.infoing info level logs
os.chdir(os.path.dirname(__file__)) #change working directory to current directory

class Checker():
    def __init__(self) -> None:
        self.standerdized_data_path = "../../Data/ACDC/database/train_standardized"

    def check_unified_resolution(self):
        """
        Check if all images in the output directory have unified resolution.
        """
        output_path = self.standerdized_data_path
        resolutions = set()  # To store unique resolutions

        for root, dirs, files in os.walk(output_path):
        
            for file in files:
                # logging.info(file)
                if file.endswith(".nii.gz"):  # Check for NIfTI files
                    nifti_file_path = os.path.join(root, file)
                    try:
                        image = sitk.ReadImage(nifti_file_path)
                        spacing = image.GetSpacing()  # Get the spacing (resolution)
                        resolutions.add(spacing)  # Add to the set of unique resolutions

                    except Exception as e:
                        logging.info(f"Failed to read {nifti_file_path}: {str(e)}")

        # Check the number of unique resolutions
        if len(resolutions) == 1:
            logging.info("All images have a unified resolution.")
        else:
            logging.info("There are multiple resolutions in the dataset:")
            for res in resolutions:
                logging.info(res)

    def check_image_value(self):

        """
        Check the minimum and maximum pixel values of the images in the output directory.
        """
        output_path = self.standerdized_data_path
        for root, dirs, files in os.walk(output_path):
            for file in files:
                if file.endswith(".nii.gz"):  # Check for NIfTI files
                    nifti_file_path = os.path.join(root, file)
                    try:
                        image = sitk.ReadImage(nifti_file_path)
                        image_np = sitk.GetArrayFromImage(image)
                        logging.info(f"Min: {image_np.min()}, Max: {image_np.max()} for: {nifti_file_path}")
                    except Exception as e:
                        logging.info(f"Failed to read {nifti_file_path}: {str(e)}")
    def check_dimension(self):
        """
        Check if all images in the output directory have a resolution of 512x512.
        """
        output_path = self.standerdized_data_path
        for root, dirs, files in os.walk(output_path):
            for file in files:
                if file.endswith(".nii.gz"):  # Check for NIfTI files
                    nifti_file_path = os.path.join(root, file)
                    try:
                        image = sitk.ReadImage(nifti_file_path)
                        size = image.GetSize()  # Get the size (resolution)
                        if size[0] == 512 or size[1] == 512:
                            logging.info(f"Resolution is 512x512 for: {nifti_file_path}")
                        else:
                            logging.info(f"Resolution lower 512x512 for: {nifti_file_path}")
                    except Exception as e:
                        logging.info(f"Failed to read {nifti_file_path}: {str(e)}")