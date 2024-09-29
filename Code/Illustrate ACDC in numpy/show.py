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

# Display a slice of the image (middle slice)
middle_slice = img[img.shape[0] // 2, :, :]  # Choose the middle slice

# Plot the image using matplotlib
plt.imshow(middle_slice, cmap='gray')
plt.colorbar()
plt.show()
