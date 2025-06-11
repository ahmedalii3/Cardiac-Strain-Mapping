import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def load_nifti_manual(filename):
    with open(filename, 'rb') as f:
        header = f.read(348)  # NIFTI header size is 348 bytes
        
        # Extract the data type code (stored at bytes 70-71 in the header)
        dtype_code = np.frombuffer(header[70:72], dtype=np.int16)[0]
        
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
        nx, ny, nz, nt = int(dims[1]), int(dims[2]), int(dims[3]), int(dims[4])  # Added time dimension
        
        # Read the image data (after the header)
        f.seek(352)  # Data starts after the 352nd byte
        img_data = np.frombuffer(f.read(), dtype=dtype)
        
        # Check if the total size matches expected dimensions
        expected_elements = nx * ny * nz * nt
        actual_elements = img_data.size
        
        if actual_elements != expected_elements:
            raise ValueError(f"Mismatch in data size. Expected {expected_elements} elements, but got {actual_elements}.")
        
        img_data = img_data.reshape((nt, nz, ny, nx))  # Reshape to 4D array (T, Z, Y, X)

    return img_data

def plot_4d_slices(img_data):
    nt, nz, ny, nx = img_data.shape
    
    fig, ax = plt.subplots()
    
    # Initialize the plot with the first time slice
    img_plot = ax.imshow(img_data[0, nz // 2, :, :], cmap='gray')
    ax.set_title("Time = 0")
    plt.colorbar(img_plot, ax=ax)
    
    def update(frame):
        img_plot.set_data(img_data[frame, nz // 2, :, :])
        ax.set_title(f"Time = {frame}")
        return img_plot,

    ani = FuncAnimation(fig, update, frames=nt, blit=True, repeat=True)
    plt.show()

# Load the .nii file manually
filename = 'Data/ACDC/database/testing/patient101/patient101_4d.nii'
img_data = load_nifti_manual(filename)

# Plot the 4D data (slice over time)
plot_4d_slices(img_data)
