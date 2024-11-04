import numpy as np
import matplotlib.pyplot as plt

def load_nifti_manual(filename):
    with open(filename, 'rb') as f:
        header = f.read(348)  # NIFTI header size is 348 bytes
        print(header)
        # print(len(header))
        # Extract the data type code (stored at bytes 70-71 in the header)
        dtype_code = np.frombuffer(header[70:72], dtype=np.int16)[0]
        print(dtype_code)
        print(dtype_code)
        # Map NIfTI data type codes to numpy dtypes
        dtype_map = {
            2: np.uint8,     # DT_UINT8
            4: np.int16,     # DT_INT16
            8: np.int32,     # DT_INT32
            16: np.float32,  # DT_FLOAT32
            64: np.float64,  # DT_FLOAT64
        }
        
        dtype = dtype_map.get(dtype_code, np.float32)  # Default to float32 if unknown
        
        # Extract dimensions from the header (stored as int16)
        dims = np.frombuffer(header[40:56], dtype=np.int16)
        nx, ny, nz = int(dims[1]), int(dims[2]), int(dims[3])  # Cast dimensions to int
        
        # Read the image data (after the header)
        f.seek(352)  # Data starts after the 352nd byte
        img_data = np.frombuffer(f.read(), dtype=dtype)
        print("imag_data shape =", img_data.shape)
        # Check if the total size matches expected dimensions
        expected_elements = nx * ny * nz
        actual_elements = img_data.size
        
        if actual_elements != expected_elements:
            raise ValueError(f"Mismatch in data size. Expected {expected_elements} elements, but got {actual_elements}.")
        
        img_data = img_data.reshape((nz, ny, nx))  # Reshape to 3D array (Z, Y, X)

    return img_data

# Load the .nii file manually
filename = 'Data/ACDC/database/training/patient001/patient001_frame01.nii/CMD03Gate1.nii'
img = load_nifti_manual(filename)
print(img.shape)
# Display a slice of the image (middle slice)
middle_slice = img[img.shape[0] // 2, :, :]  # Choose the middle slice

# Plot the image using matplotlib
plt.imshow(middle_slice, cmap='gray')
plt.colorbar()
plt.show()


def convert_any_neg_to_zero(output_path):
    """
    Convert any negative pixel values to zero in the images in the output directory.
    """
    for root, dirs, files in os.walk(output_path):
        for file in files:
            if file.endswith(".nii"):  # Check for NIfTI files
                nifti_file_path = os.path.join(root, file)
                try:
                    image = sitk.ReadImage(nifti_file_path)
                    image_np = sitk.GetArrayFromImage(image)
                    image_np[image_np < 0] = 0
                    image = sitk.GetImageFromArray(image_np)
                    sitk.WriteImage(image, nifti_file_path)
                    print(f"Converted negative values to zero for: {nifti_file_path}")
                except Exception as e:
                    print(f"Failed to read {nifti_file_path}: {str(e)}")
output_path = "/Users/ahmed_ali/Documents/GitHub/GP-2025-Strain/Data/ACDC/database/train_standardized"  # Path where standardized images are saved
convert_any_neg_to_zero(output_path)

def normalize_file_by_file(dicom_file, output_dir):
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
    print(f"Saved: {output_dir}")
    print(f"Mean: {mean}, Std: {std}")
    

def loop_and_normalize(dataset_path, output_path, search_term="4d"):
    """
    Loop over the dataset, clone the folder structure, and standardize the resolution of single-slice data.
    """
    # Clone the directory structure
    # clone_directory_structure(dataset_path, output_path)

    for root, dirs, files in os.walk(dataset_path):
        # Ignore directories with '4d' in their name
        if search_term not in os.path.basename(root):
            for file in files:
                if search_term not in file:
                    if file.endswith(".nii"):  # Assuming DICOM files are used
                        dicom_file_path = os.path.join(root, file)
                        print(f"Processing: {dicom_file_path}")
                        
                        # Process and save in the corresponding output directory
                        output_dir = root.replace(dataset_path, output_path, 1)
                        normalize_file_by_file(dicom_file_path, output_dir)

# Example usage:
dataset_path = "/Users/ahmed_ali/Documents/GitHub/GP-2025-Strain/Data/ACDC/database/train_standardized"  # Original dataset path
output_path = "/Users/ahmed_ali/Documents/GitHub/GP-2025-Strain/Data/ACDC/database/train_standardized"  # New path for standardized data
loop_and_normalize(dataset_path, output_path)
# check for the min and max value of the image
def check_image_values(output_path):
    """
    Check the minimum and maximum pixel values of the images in the output directory.
    """
    for root, dirs, files in os.walk(output_path):
        for file in files:
            if file.endswith(".nii"):  # Check for NIfTI files
                nifti_file_path = os.path.join(root, file)
                try:
                    image = sitk.ReadImage(nifti_file_path)
                    image_np = sitk.GetArrayFromImage(image)
                    print(f"Min: {image_np.min()}, Max: {image_np.max()} for: {nifti_file_path}")
                except Exception as e:
                    print(f"Failed to read {nifti_file_path}: {str(e)}")

output_path = "/Users/ahmed_ali/Documents/GitHub/GP-2025-Strain/Data/ACDC/database/train_standardized"  # Path where standardized images are saved
check_image_values(output_path)