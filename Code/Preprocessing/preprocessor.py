import SimpleITK as sitk
import os
import numpy as np
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
import scipy.interpolate as spi
import logging
from checker import Checker

logging.getLogger().setLevel(logging.INFO) # Allow logging.infoing info level logs
os.chdir(os.path.dirname(__file__)) #change working directory to current directory

class Preprocessor:
    def __init__(self):
        self.original_data_path = "../../Data/ACDC/training"
        self.standerdized_data_path = "../../Data/ACDC/train_standardized"
        self.numpy_data_path = "../../Data/ACDC/train_numpy"
        self.search_term = "4d"
        self.target_spacing = np.array([1.0,1.0,1.0])
        self.target_padding = np.array([512,512,1.0])

    def clone_directory_structure(self, source_dir: str, target_dir: str) -> None:
        """
        Clone the directory structure from src to dst.
        """
        source_dir = os.path.abspath(source_dir)
        target_dir = os.path.abspath(target_dir)
        for root, dirs, files in os.walk(source_dir):
            # Create the corresponding directory in the destination
            dst_dir = os.path.join(target_dir, os.path.relpath(root, source_dir))  # Make dst_dir relative to the absolute src
            logging.info(f"Creating directory: {dst_dir}")
            os.makedirs(dst_dir, exist_ok=True)  # Create the directory if it doesn't exist

        

    def monotonic_zoom_interpolate(self, img: np.ndarray, zoom_factor: float) -> np.ndarray:
        """
        Apply monotonic zoom interpolation to a given image.
        """        
        result = img.copy()

        for axis, factor in enumerate(zoom_factor[::-1]):
            new_length = int(result.shape[axis] * factor)
            x_old = np.arange(result.shape[axis])
            x_new = np.linspace(0, result.shape[axis] - 1, new_length)
            
            # Perform monotonic interpolation
            pchip_interp = spi.PchipInterpolator(x_old, result.take(indices=x_old, axis=axis), axis=axis)
            result = pchip_interp(x_new)
        return result
    
    def process_single_slice_dicom(self, dicom_file, target_resolution, output_dir):
        """
        Load and standardize the spacing of a single-slice DICOM file.
        """
         # Load the DICOM file
        image = sitk.ReadImage(dicom_file)
        current_spacing = np.array(image.GetSpacing())

        resize_factor = np.array([current_spacing[0] / target_resolution[0], 
                                current_spacing[1] / target_resolution[1], 
                                1.0])

        new_real_shape = np.array(image.GetSize()) * resize_factor

        new_shape = np.round(new_real_shape).astype(int)
        
        real_resize_factor = new_shape / np.array(image.GetSize())
        
        image_np = sitk.GetArrayFromImage(image)

        # image_resampled_np = zoom(image_np, real_resize_factor[::-1], order=1)
        image_resampled_np = self.monotonic_zoom_interpolate(image_np, real_resize_factor)

        image_resampled = sitk.GetImageFromArray(image_resampled_np)

        new_spacing = np.array([target_resolution[0], target_resolution[1], current_spacing[2]])

        image_resampled.SetSpacing(new_spacing)
        
        output_file = os.path.join(output_dir, os.path.basename(dicom_file))
        output_file = os.path.abspath(output_file)
        sitk.WriteImage(image_resampled, output_file)

        logging.info(f"Saved: {output_file}")
        logging.info(f"Original shape: {image_np.shape}, Resampled shape: {image_resampled_np.shape}")
        logging.info(f"Original spacing: {current_spacing}, New spacing: {image_resampled.GetSpacing()}")

    def change_mask_values(self, mask_np):
        """
        Change the values of the mask to 0 and 1 to remove the Right Ventricle.
        """
        mask_np[mask_np == 1] = 0
        mask_np[mask_np == 3] = 1

        return mask_np

    def process_mask_dicom(self, dicom_file, target_resolution, output_dir):
        # Load the DICOM file and change the values of the mask
        image = sitk.ReadImage(dicom_file)

        current_spacing = np.array(image.GetSpacing())

        resize_factor = np.array([current_spacing[0] / target_resolution[0], 
                                current_spacing[1] / target_resolution[1], 
                                1.0])

        new_real_shape = np.array(image.GetSize()) * resize_factor
        new_shape = np.round(new_real_shape).astype(int)        
        real_resize_factor = new_shape / np.array(image.GetSize())    

        image_np = sitk.GetArrayFromImage(image)
        image_resampled_np = zoom(image_np, real_resize_factor[::-1], order=0)
        image_resampled = sitk.GetImageFromArray(image_resampled_np)
        new_spacing = np.array([target_resolution[0], target_resolution[1], current_spacing[2]])
        image_resampled.SetSpacing(new_spacing)

        output_file = os.path.join(output_dir, os.path.basename(dicom_file))
        output_file = os.path.abspath(output_file)
        sitk.WriteImage(image_resampled, output_file)

        logging.info(f"Saved: {output_file}")
        logging.info(f"Original shape: {image_np.shape}, Resampled shape: {image_resampled_np.shape}")
        logging.info(f"Original spacing: {current_spacing}, New spacing: {image_resampled.GetSpacing()}")

    def loop_and_standardize(self):
        """
        Loop over the dataset, clone the folder structure, and standardize the resolution of single-slice data.
        """
        dataset_path = self.original_data_path
        output_path = self.standerdized_data_path
        search_term = self.search_term
        target_resolution = self.target_spacing

        logging.info("clone directory")
        self.clone_directory_structure(dataset_path, output_path)
        logging.info("directory cloned")
        
        for root, dirs, files in os.walk(dataset_path):
            # Ignore directories with '4d' in their name
            if search_term not in os.path.basename(root):
                if "=" not in os.path.basename(root):
                    for file in files:
                        if search_term not in file:
                            if "gt" not in file:
                                if file.endswith(".nii.gz"):  # Assuming DICOM files are used
                                    dicom_file_path = os.path.join(root, file)
                                    logging.info(f"Processing: {dicom_file_path}")
                                
                                    # Process and save in the corresponding output directory
                                    output_dir = root.replace(dataset_path, output_path, 1)
                                    self.process_single_slice_dicom(dicom_file_path, target_resolution, output_dir)                           
                            elif "gt" in file:
                                if file.endswith(".nii.gz"):
                                    dicom_file_path = os.path.join(root, file)
                                    logging.info(f"Processing: {dicom_file_path}")
                                    
                                    # Process and save in the corresponding output directory
                                    output_dir = root.replace(dataset_path, output_path, 1)
                                    self.process_mask_dicom(dicom_file_path, target_resolution, output_dir)


    def normalize_file_by_file(self, dicom_file, output_dir):
        """
        load and normalize single slice dicome file by z-score normalization
        """
        image = sitk.ReadImage(dicom_file)
        image_np = sitk.GetArrayFromImage(image)
        mean = image_np.mean()
        std = image_np.std()
        image_np = (image_np - mean) / std
        image = sitk.GetImageFromArray(image_np)
        output_file = os.path.join(output_dir, os.path.basename(dicom_file))
        sitk.WriteImage(image, output_file)
        logging.info(f"Saved: {output_dir}")
        logging.info(f"Mean: {mean}, Std: {std}")

    def loop_and_normalize(self):
        """ 
        Loop over the dataset, clone the folder structure, and standardize the resolution of single-slice data.
        """
        dataset_path = output_path = self.standerdized_data_path
        search_term = self.search_term

        for root, dirs, files in os.walk(dataset_path):
            # Ignore directories with '4d' in their name
            if search_term not in os.path.basename(root):
                if "gt" not in os.path.basename(root):
                    for file in files:
                        if search_term not in file:
                            if "gt" not in file:
                                if file.endswith(".nii.gz"):  # Assuming DICOM files are used
                                    dicom_file_path = os.path.join(root, file)
                                    logging.info(f"Processing: {dicom_file_path}")
                                    
                                    # Process and save in the corresponding output directory
                                    output_dir = root.replace(dataset_path, output_path, 1)
                                    self.normalize_file_by_file(dicom_file_path, output_dir)

   
    def process_dimension_by_padding(self,dicom_file, target_resolution, output_dir):
        """
        Load and standardize the resolution of a single-slice DICOM file using padding.
        """
        image = sitk.ReadImage(dicom_file)
        constant_val = int(sitk.GetArrayFromImage(image).min())

        current_size = np.array(image.GetSize())
        padding_left_right = target_resolution[0] - current_size[0]
        padding_top_bottom = target_resolution[1] - current_size[1]
        padding_left = int(padding_left_right // 2)
        padding_right = int(padding_left_right - padding_left)
        padding_top = int(padding_top_bottom // 2)
        padding_bottom = int(padding_top_bottom - padding_top)

        transformed = sitk.ConstantPad(image,(padding_left,padding_top,0),(padding_right,padding_bottom,0),constant_val)
        
        output_file = os.path.join(output_dir, os.path.basename(dicom_file))
        output_file = os.path.abspath(output_file)
        sitk.WriteImage(transformed, output_file)

    def loop_and_pad(self):
        """
        Loop over the dataset, clone the folder structure, and standardize the resolution of single-slice data.
        """
        
        dataset_path = output_path = self.standerdized_data_path
        search_term = self.search_term
        target_resolution = self.target_padding
        

        for root, dirs, files in os.walk(dataset_path):
            # Ignore directories with '4d' in their name
            if search_term not in os.path.basename(root):
                for file in files:
                    if search_term not in file:
                        if file.endswith(".nii.gz"):  # Assuming DICOM files are used
                            dicom_file_path = os.path.join(root, file)
                            logging.info(f"Processing: {dicom_file_path}")
                            
                            # Process and save in the corresponding output directory
                            output_dir = root.replace(dataset_path, output_path, 1)
                            self.process_dimension_by_padding(dicom_file_path, target_resolution, output_dir)

    def create_two_channel_numpy_array(self, patient_root, output_path):
        patient_root = os.path.abspath(patient_root)
        output_path = os.path.abspath(output_path)
        self.clone_directory_structure(patient_root, output_path)
        file_list = os.listdir(patient_root)
        # create dictionary for each file with its mask
        file_dict = {}
        mask_dict = {}
        frames_list = []
        for file in file_list:
            if file.endswith(".nii.gz"):
                
                if "gt" in file:
                    name_of_frame = file.split("_")[1]
                    frames_list.append(name_of_frame)
                    # print("proccesing file: ", file)    
                    mask_dict[name_of_frame] = file
                else:
                    name_of_frame = file.split("_")[1]
                    name_of_frame = name_of_frame.split(".")[0]                    
                    file_dict[name_of_frame] = file                
        if file_list[0].endswith(".nii.gz"):
            
        # create a 2 channel numpy array from the mask and the image
            for frame in frames_list:
                image = sitk.ReadImage(os.path.join(patient_root,file_dict[frame]))
                mask = sitk.ReadImage(os.path.join(patient_root,mask_dict[frame]))
                
                image_np = sitk.GetArrayFromImage(image)
                mask_np = sitk.GetArrayFromImage(mask)

                mask_np = self.change_mask_values(mask_np)
                
                for slice in range(image_np.shape[0]):
                    numpy_array = np.stack((image_np[slice], mask_np[slice]), axis=0)
                    output_file = os.path.join(output_path, os.path.basename(patient_root),  os.path.basename(patient_root) + f"_{frame}_slice_{slice}_ACDC.npy")
                    output_file = os.path.abspath(output_file)
                    if not os.path.exists(os.path.join(output_path, os.path.basename(patient_root))):
                        os.makedirs(os.path.join(output_path, os.path.basename(patient_root)))
                    np.save(output_file, numpy_array)        


    def create_numpy(self):
        dataset_path = self.standerdized_data_path
        output_path = self.numpy_data_path

        for root, dirs, files in os.walk(dataset_path):
            # pass each directory to the function
            self.create_two_channel_numpy_array(root,output_path)



preprocessor = Preprocessor()
checker = Checker()

logging.info("Standardizing resolution...")
preprocessor.loop_and_standardize()
checker.check_unified_resolution()

logging.info("Normalizing images...")
preprocessor.loop_and_normalize()
checker.check_image_value()

logging.info("Padding images...")
preprocessor.loop_and_pad()
checker.check_dimension()

logging.info("Creating numpy arrays...")
preprocessor.create_numpy()
