import numpy as np
from displ_strain_conversion import calculate_strain_and_displacement
# Example input arrays (assuming proper shapes)
# frame_displ_x = np.zeros((100, 100, 10))  # Example dimensions
frame_displ_x = np.load("/Users/osama/GP-2025-Strain/Data/ACDC/Simulated_data_localized/Displacements_Loc/patient002_frame12_slice_1_ACDC_#1_x.npy")
frame_displ_y = np.load("/Users/osama/GP-2025-Strain/Data/ACDC/Simulated_data_localized/Displacements_Loc/patient002_frame12_slice_1_ACDC_#1_y.npy")  # Example dimensions
delta_x = 1.0
delta_y = 1.0
strain_ep_peak = 0.25
cine_no_frames = 1

# Call the function
new_displ_x, new_displ_y, max_displ, min_displ, max_ep = calculate_strain_and_displacement(
    frame_displ_x, frame_displ_y, delta_x, delta_y, strain_ep_peak, cine_no_frames
)
print(max_ep)