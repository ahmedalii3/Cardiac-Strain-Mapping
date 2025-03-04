import numpy as np
import matplotlib.pyplot as plt
from displ_strain_conversion import strain_validation
# Example input arrays (assuming proper shapes)
# frame_displ_x = np.zeros((100, 100, 10))  # Example dimensions
frame_displ_x = np.load("/Users/osama/GP-2025-Strain/Data/ACDC/Simulated_data_localized/Displacements_Loc/patient002_frame12_slice_1_ACDC_#1_x.npy")
frame_displ_y = np.load("/Users/osama/GP-2025-Strain/Data/ACDC/Simulated_data_localized/Displacements_Loc/patient002_frame12_slice_1_ACDC_#1_y.npy")  # Example dimensions
# frame_displ_x = np.load("/Users/osama/GP-2025-Strain/Code/Wave/Saved_test/Displacements/patient020_frame11_slice_0_ACDC_#1_x.npy")
# frame_displ_y = np.load("/Users/osama/GP-2025-Strain/Code/Wave/Saved_test/Displacements/patient020_frame11_slice_0_ACDC_#1_y.npy")  # Example dimensions
delta_x = 1.0
delta_y = 1.0
strain_ep_peak = 0.1
cine_no_frames = 1

# Call the function
new_displ_x, new_displ_y, max_displ, min_displ, max_ep , initial, last= strain_validation(
    frame_displ_x, frame_displ_y, delta_x, delta_y, strain_ep_peak
)
print(max_ep)
#plot initial and last strain


# # Create a figure with 2x2 subplots with increased figure size
fig, axes = plt.subplots(2, 2, figsize=(16, 14))
# Increase spacing between subplots
plt.subplots_adjust(hspace=0.5, wspace=0.4)

# Plot initial strain map
im1 = axes[0,0].imshow(initial)
axes[0,0].set_title('Initial Strain', pad=20, fontsize=12)
plt.colorbar(im1, ax=axes[0,0], pad=0.1)

# Plot last strain map
im2 = axes[0,1].imshow(last)
axes[0,1].set_title('Last Strain', pad=20, fontsize=12)
plt.colorbar(im2, ax=axes[0,1], pad=0.1)

# Plot initial histogram
axes[1,0].hist(initial.flatten(), bins=50, color='blue', alpha=0.7)
axes[1,0].set_title('Initial Strain Histogram', pad=20, fontsize=12)
axes[1,0].set_xlabel('Strain Value')
axes[1,0].set_ylabel('Frequency')

# Plot last histogram
axes[1,1].hist(last.flatten(), bins=50, color='red', alpha=0.7)
axes[1,1].set_title('Last Strain Histogram', pad=20, fontsize=12)
axes[1,1].set_xlabel('Strain Value')
axes[1,1].set_ylabel('Frequency')

# Adjust layout with padding
plt.tight_layout(pad=3.0)
plt.show()

