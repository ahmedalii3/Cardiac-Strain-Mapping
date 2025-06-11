# import numpy as np
# import matplotlib.pyplot as plt

# x_disp = np.load('/Users/ahmed_ali/Downloads/generatedData/Cropped_Disp/patient008_frame13_slice_9_ACDC_#55_x.npy')
# y_disp = np.load('/Users/ahmed_ali/Downloads/generatedData/Cropped_Disp/patient008_frame13_slice_9_ACDC_#55_y.npy')
# frame1 = np.load('/Users/ahmed_ali/Downloads/generatedData/Frames/patient008_frame13_slice_9_ACDC_#55_1.npy')
# frame2 = np.load('/Users/ahmed_ali/Downloads/generatedData/Frames/patient008_frame13_slice_9_ACDC_#55_2.npy')

# def limit_strain_range(displacement_x, displacement_y, stretch, strain_upper_bound, 
#                      reduction_factor=0.99, amplification_factor=1.01, max_iterations=1000, tolerance=1e-6):
#     """
#     Convert displacement maps to strain tensors using Eulerian strain formulation.
#     Iteratively adjust displacements until all strain values are within the specified bounds:
#     - Reduce displacements if strain exceeds upper bound
#     - Amplify displacements if strain is below lower bound
    
#     Parameters:
#     -----------
#     displacement_x : numpy.ndarray
#         Displacement field in x-direction
#     displacement_y : numpy.ndarray
#         Displacement field in y-direction
#     strain_lower_bound : float
#         Minimum desired strain value
#     strain_upper_bound : float
#         Maximum allowable strain value
#     reduction_factor : float, optional
#         Factor by which to reduce displacements each iteration (default: 0.99)
#     amplification_factor : float, optional
#         Factor by which to amplify displacements each iteration (default: 1.01)
#     max_iterations : int, optional
#         Maximum number of iterations to perform (default: 1000)
#     tolerance : float, optional
#         Convergence tolerance (default: 1e-6)
    
#     Returns:
#     --------
#     tuple
#         (adjusted_displacement_x, adjusted_displacement_y, 
#          initial_strain_tensor, final_strain_tensor, max_initial_strain, max_final_strain)
#     """
#     # Ensure input arrays have the same shape
#     if displacement_x.shape != displacement_y.shape:
#         raise ValueError("Displacement maps must have the same shape")
#     if stretch:
#         strain_lower_bound = 0.01
#     else:
#         strain_lower_bound = 0
    
#     # Make copies of the input arrays to avoid modifying the originals
#     dx = displacement_x.copy()
#     dy = displacement_y.copy()

#     # Create gradient operators for calculating spatial derivatives
#     y_size, x_size = dx.shape
    
#     # Calculate initial strain tensor
#     # Calculate displacement gradients using central differences
#     dudx_initial = np.zeros_like(dx)
#     dudy_initial = np.zeros_like(dx)
#     dvdx_initial = np.zeros_like(dx)
#     dvdy_initial = np.zeros_like(dx)
    
#     # Interior points (central differences)
#     dudx_initial[1:-1, 1:-1] = (dx[1:-1, 2:] - dx[1:-1, :-2]) / 2
#     dudy_initial[1:-1, 1:-1] = (dx[2:, 1:-1] - dx[:-2, 1:-1]) / 2
#     dvdx_initial[1:-1, 1:-1] = (dy[1:-1, 2:] - dy[1:-1, :-2]) / 2
#     dvdy_initial[1:-1, 1:-1] = (dy[2:, 1:-1] - dy[:-2, 1:-1]) / 2
    
#     # Edges (forward/backward differences)
#     # Left edge
#     dudx_initial[:, 0] = dx[:, 1] - dx[:, 0]
#     dvdx_initial[:, 0] = dy[:, 1] - dy[:, 0]
#     # Right edge
#     dudx_initial[:, -1] = dx[:, -1] - dx[:, -2]
#     dvdx_initial[:, -1] = dy[:, -1] - dy[:, -2]
#     # Top edge
#     dudy_initial[0, :] = dx[1, :] - dx[0, :]
#     dvdy_initial[0, :] = dy[1, :] - dy[0, :]
#     # Bottom edge
#     dudy_initial[-1, :] = dx[-1, :] - dx[-2, :]
#     dvdy_initial[-1, :] = dy[-1, :] - dy[-2, :]
    
#     # Calculate Eulerian strain tensor components
#     # E = 1/2 * (∇u + ∇u^T + ∇u^T∇u)
#     E_xx_initial = 0.5 * (2*dudx_initial - dudx_initial**2 - dvdx_initial**2)
#     E_yy_initial = 0.5 * (2*dvdy_initial - dudy_initial**2 - dvdy_initial**2)
#     E_xy_initial = 0.5 * (dudy_initial + dvdx_initial - dudx_initial*dudy_initial - dvdx_initial*dvdy_initial)
    
#     # Calculate principal strains
#     avg_normal_strain_initial = (E_xx_initial + E_yy_initial) / 2
#     diff_normal_strain_initial = (E_xx_initial - E_yy_initial) / 2
#     radius_initial = np.sqrt(diff_normal_strain_initial**2 + E_xy_initial**2)
    
#     E1_initial = avg_normal_strain_initial + radius_initial  # Maximum principal strain
#     E2_initial = avg_normal_strain_initial - radius_initial  # Minimum principal strain

#     E3_initial = 1 / ((1 + E1_initial) * (1 + E2_initial)) - 1
    
#     # Find maximum and minimum absolute strain values
#     max_initial_strain = max(np.max(np.abs(E1_initial)), np.max(np.abs(E2_initial)))
#     min_initial_strain = min(np.min(np.abs(E1_initial)), np.min(np.abs(E2_initial)))
    
#     # Store initial strain tensor
#     initial_strain_tensor = {
#         'E_xx': E_xx_initial,
#         'E_yy': E_yy_initial,
#         'E_xy': E_xy_initial,
#         'E1': E1_initial,
#         'E2': E2_initial,
#         'E3': E3_initial,
#         'min_abs_strain': min_initial_strain,
#         'max_abs_strain': max_initial_strain
#     }
    
#     # If initial strain is already within bounds, no need to iterate
#     if (max_initial_strain <= strain_upper_bound) and (min_initial_strain >= strain_lower_bound):
#         return dx, dy, initial_strain_tensor, initial_strain_tensor, max_initial_strain, max_initial_strain, min_initial_strain, min_initial_strain
    
#     # Otherwise, proceed with iterative adjustment
#     iterations = 0
#     max_strain = max_initial_strain
#     min_strain = min_initial_strain
#     prev_max_strain = float('inf')
#     prev_min_strain = 0
    
#     # Initialize strain tensor components for the loop
#     E_xx = E_xx_initial.copy()
#     E_yy = E_yy_initial.copy()
#     E_xy = E_xy_initial.copy()
#     E1 = E1_initial.copy()
#     E2 = E2_initial.copy()
    
#     while ((max_strain > strain_upper_bound) or (min_strain < strain_lower_bound)) and (iterations < max_iterations):
#         # Determine whether to reduce or amplify displacements
#         if max_strain > strain_upper_bound:
#             # Reduce displacements if above upper bound
#             adjustment_factor = reduction_factor
#         elif min_strain < strain_lower_bound:
#             # Amplify displacements if below lower bound
#             adjustment_factor = amplification_factor
#         else:
#             # This shouldn't happen due to the while condition, but just in case
#             break
        
#         # Apply adjustment
#         dx *= adjustment_factor
#         dy *= adjustment_factor
        
#         # Recalculate displacement gradients
#         dudx = np.zeros_like(dx)
#         dudy = np.zeros_like(dx)
#         dvdx = np.zeros_like(dx)
#         dvdy = np.zeros_like(dx)
        
#         # Interior points (central differences)
#         dudx[1:-1, 1:-1] = (dx[1:-1, 2:] - dx[1:-1, :-2]) / 2
#         dudy[1:-1, 1:-1] = (dx[2:, 1:-1] - dx[:-2, 1:-1]) / 2
#         dvdx[1:-1, 1:-1] = (dy[1:-1, 2:] - dy[1:-1, :-2]) / 2
#         dvdy[1:-1, 1:-1] = (dy[2:, 1:-1] - dy[:-2, 1:-1]) / 2
        
#         # Edges (forward/backward differences)
#         # Left edge
#         dudx[:, 0] = dx[:, 1] - dx[:, 0]
#         dvdx[:, 0] = dy[:, 1] - dy[:, 0]
#         # Right edge
#         dudx[:, -1] = dx[:, -1] - dx[:, -2]
#         dvdx[:, -1] = dy[:, -1] - dy[:, -2]
#         # Top edge
#         dudy[0, :] = dx[1, :] - dx[0, :]
#         dvdy[0, :] = dy[1, :] - dy[0, :]
#         # Bottom edge
#         dudy[-1, :] = dx[-1, :] - dx[-2, :]
#         dvdy[-1, :] = dy[-1, :] - dy[-2, :]
        
#         # Calculate Eulerian strain tensor components
#         # E = 1/2 * (∇u + ∇u^T + ∇u^T∇u)
#         E_xx = 0.5 * (2*dudx - dudx**2 - dvdx**2)
#         E_yy = 0.5 * (2*dvdy - dudy**2 - dvdy**2)
#         E_xy = 0.5 * (dudy + dvdx - dudx*dudy - dvdx*dvdy)
        
#         # Calculate principal strains
#         avg_normal_strain = (E_xx + E_yy) / 2
#         diff_normal_strain = (E_xx - E_yy) / 2
#         radius = np.sqrt(diff_normal_strain**2 + E_xy**2)
        
#         E1 = avg_normal_strain + radius  # Maximum principal strain
#         E2 = avg_normal_strain - radius  # Minimum principal strain
#         E3 = 1 / ((1 + E1) * (1 + E2)) - 1
        
#         # Find maximum and minimum absolute strain values
#         max_strain = max(np.max(np.abs(E1)), np.max(np.abs(E2)))
#         min_strain = min(np.min(np.abs(E1)), np.min(np.abs(E2)))
        
#         # Check convergence
#         if (abs(max_strain - prev_max_strain) < tolerance and 
#             abs(min_strain - prev_min_strain) < tolerance):
#             break
        
#         prev_max_strain = max_strain
#         prev_min_strain = min_strain
#         iterations += 1
    
#     # Prepare final strain tensor
#     final_strain_tensor = {
#         'E_xx': E_xx,
#         'E_yy': E_yy,
#         'E_xy': E_xy,
#         'E1': E1,
#         'E2': E2,
#         'E3': E3,
#         'min_abs_strain': min_strain,
#         'max_abs_strain': max_strain
#     }
    
#     return dx, dy, initial_strain_tensor, final_strain_tensor, max_initial_strain, max_strain, min_initial_strain, min_strain


# dx, dy, initial_strain_tensor, final_strain_tensor, max_initial_strain, max_strain, min_initial_strain, min_strain = limit_strain_range(x_disp, y_disp, False, 1)
# fig, ax = plt.subplots(1, 3, figsize=(15, 5))
# im1 = ax[0].imshow(initial_strain_tensor['E1'], cmap='jet', vmin=-0.4, vmax=0.4)
# ax[0].set_title('E1 Strain')
# fig.colorbar(im1, ax=ax[0])

# im2 = ax[1].imshow(initial_strain_tensor['E2'], cmap='jet', vmin=-0.4, vmax=0.4)
# ax[1].set_title('E2 Strain')
# fig.colorbar(im2, ax=ax[1])
# # magnitude = np.sqrt(x_disp**2 + y_disp**2)
# im3 = ax[2].imshow(frame1, cmap='jet')
# ax[2].set_title('Displacement')
# fig.colorbar(im3, ax=ax[2])
# plt.show()


# from scipy.ndimage import distance_transform_edt, grey_closing, binary_closing, gaussian_filter
# def dilate_mask_fade(mask, decay_distance=20, binary_closing_size=5, grey_closing_size=5):
#     """
#     Create a smooth fading mask based on distance from the cleaned binary mask boundary,
#     applying binary closing first, and grayscale closing only outside the mask.

#     Args:
#         mask (np.ndarray): 2D binary mask (1 inside, 0 outside).
#         decay_distance (float): Distance (in pixels) over which the mask fades to zero.
#         binary_closing_size (int): Size for binary morphological closing (first step).
#         grey_closing_size (int): Size for grayscale morphological closing (only on fading region).

#     Returns:
#         smooth_mask (np.ndarray): Smooth fading mask (values 1 inside, fades to 0 outside).
#     """

#     # Step 1: Ensure binary mask
#     binary_mask = (mask > 0).astype(np.uint8)

#     # Step 2: Apply binary closing first to smooth the mask
#     binary_structure = np.ones((binary_closing_size, binary_closing_size))
#     cleaned_binary_mask = binary_closing(binary_mask, structure=binary_structure).astype(np.uint8)

#     # Step 3: Compute distance transform outside the cleaned mask
#     distance_outside = distance_transform_edt(1 - cleaned_binary_mask)

#     # Step 4: Normalize distances to [1, 0] range for fading
#     smooth_mask = np.clip(1 - distance_outside / decay_distance, 0, 1)

#     # Step 5: Apply grayscale closing only outside the mask
# #    mask_inside = cleaned_binary_mask == 1
# #    mask_outside = cleaned_binary_mask == 0

#     smooth_mask_final = grey_closing(smooth_mask, structure=np.ones((grey_closing_size, grey_closing_size)))

#     # Step 6: Combine: inside stays 1, outside is smoothed fading
#     #smooth_mask_final = smooth_mask.copy()
#     #smooth_mask_final[mask_outside] = smooth_mask_outside[mask_outside]
#     #smooth_mask_final[mask_inside] = 1  # Enforce inside = 1 exactly
#     # Get mask of values strictly between 0 and 1
#     mask_between = (smooth_mask_final > 0) 

#     # Copy smooth_mask_final
#     # fade = smooth_mask_final.copy()

#     # Apply logistic function only where values are in (0, 1)
#     # fade[mask_between] = (1 + 10*np.exp(-10* (1 - 0.5))) / (1 + 10*np.exp(-10 * (fade[mask_between] - 0.5)))
#     fade = (1 - np.exp(1.2 * smooth_mask_final)) / (1 - np.exp(1.2))

#     # Explicitly set inside and outside
#     # fade[smooth_mask_final >= 1] = 1
#     fade[smooth_mask_final <= 0] = 0

# # Return final mask
#     return fade
    
# import numpy as np
# import matplotlib.pyplot as plt
# mask = np.load("/Users/ahmed_ali/Documents/GitHub/GP-2025-Strain/Data/ACDC/database/train_numpy/patient009/patient009_frame01_slice_3_ACDC.npy")
# # print mask uniqque values
# print(np.unique(mask[1]))
# mask[1]=np.where(mask[1] == 2, 1, 0)
# dilated_mask = dilate_mask_fade(mask[1])
# plt.imshow(dilated_mask)
# plt.colorbar()
# plt.show()

import numpy as np
import matplotlib.pyplot as plt
y_train = np.load('/Users/ahmed_ali/Documents/GitHub/GP-2025-Strain/Code/FrameWork/Loaded_data/y_train.npy')
mask = y_train[0,...,2]
plt.imshow(mask)
plt.colorbar()
plt.show()