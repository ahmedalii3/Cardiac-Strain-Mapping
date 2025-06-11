from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
def limit_strain_range(displacement_x, displacement_y, strain_upper_bound, stretch = False,
                     reduction_factor=0.99, amplification_factor=1.01, max_iterations=1000, tolerance=1e-6):
        """
        Convert displacement maps to strain tensors using Eulerian strain formulation.
        Iteratively adjust displacements until all strain values are within the specified bounds:
        - Reduce displacements if strain exceeds upper bound
        - Amplify displacements if strain is below lower bound

        Parameters:
        -----------
        displacement_x : numpy.ndarray
            Displacement field in x-direction
        displacement_y : numpy.ndarray
            Displacement field in y-direction
        strain_lower_bound : float
            Minimum desired strain value
        strain_upper_bound : float
            Maximum allowable strain value
        reduction_factor : float, optional
            Factor by which to reduce displacements each iteration (default: 0.99)
        amplification_factor : float, optional
            Factor by which to amplify displacements each iteration (default: 1.01)
        max_iterations : int, optional
            Maximum number of iterations to perform (default: 1000)
        tolerance : float, optional
            Convergence tolerance (default: 1e-6)

        Returns:
        --------
        tuple
            (adjusted_displacement_x, adjusted_displacement_y,
            initial_strain_tensor, final_strain_tensor, max_initial_strain, max_final_strain)
        """
        # Ensure input arrays have the same shape
        if displacement_x.shape != displacement_y.shape:
            raise ValueError("Displacement maps must have the same shape")
        if stretch:
            strain_lower_bound = 0.01
        else:
            strain_lower_bound = 0

        # Make copies of the input arrays to avoid modifying the originals
        dx = displacement_x.copy()
        dy = displacement_y.copy()

        # Create gradient operators for calculating spatial derivatives
        y_size, x_size = dx.shape

        # Calculate initial strain tensor
        # Calculate displacement gradients using central differences
        dudx_initial = np.zeros_like(dx)
        dudy_initial = np.zeros_like(dx)
        dvdx_initial = np.zeros_like(dx)
        dvdy_initial = np.zeros_like(dx)

        # Interior points (central differences)
        dudx_initial[1:-1, 1:-1] = (dx[1:-1, 2:] - dx[1:-1, :-2]) / 2
        dudy_initial[1:-1, 1:-1] = (dx[2:, 1:-1] - dx[:-2, 1:-1]) / 2
        dvdx_initial[1:-1, 1:-1] = (dy[1:-1, 2:] - dy[1:-1, :-2]) / 2
        dvdy_initial[1:-1, 1:-1] = (dy[2:, 1:-1] - dy[:-2, 1:-1]) / 2

        # Edges (forward/backward differences)
        # Left edge
        dudx_initial[:, 0] = dx[:, 1] - dx[:, 0]
        dvdx_initial[:, 0] = dy[:, 1] - dy[:, 0]
        # Right edge
        dudx_initial[:, -1] = dx[:, -1] - dx[:, -2]
        dvdx_initial[:, -1] = dy[:, -1] - dy[:, -2]
        # Top edge
        dudy_initial[0, :] = dx[1, :] - dx[0, :]
        dvdy_initial[0, :] = dy[1, :] - dy[0, :]
        # Bottom edge
        dudy_initial[-1, :] = dx[-1, :] - dx[-2, :]
        dvdy_initial[-1, :] = dy[-1, :] - dy[-2, :]

        # Calculate Eulerian strain tensor components
        # E = 1/2 * (∇u + ∇u^T + ∇u^T∇u)
        E_xx_initial = 0.5 * (2*dudx_initial + dudx_initial**2 + dvdx_initial**2)
        E_yy_initial = 0.5 * (2*dvdy_initial + dudy_initial**2 + dvdy_initial**2)
        E_xy_initial = 0.5 * (dudy_initial + dvdx_initial + dudx_initial*dudy_initial + dvdx_initial*dvdy_initial)
        E_yx_initial = E_xy_initial

        # Calculate principal strains
        avg_normal_strain_initial = (E_xx_initial + E_yy_initial) / 2
        diff_normal_strain_initial = (E_xx_initial - E_yy_initial) / 2
        radius_initial = np.sqrt(diff_normal_strain_initial**2 + E_xy_initial**2)


        E1_initial = avg_normal_strain_initial + radius_initial  # Maximum principal strain
        E2_initial = avg_normal_strain_initial - radius_initial  # Minimum principal strain

        # KHZ 250318: Corrected the calculation of principal strains
        E_xx_initial = 0.5 * (2*dudx_initial - dudx_initial**2 - dvdx_initial**2)
        E_yy_initial = 0.5 * (2*dvdy_initial - dudy_initial**2 - dvdy_initial**2)
        E_xy_initial = 0.5 * (dudy_initial + dvdx_initial - dudx_initial*dudy_initial - dvdx_initial*dvdy_initial)

        E1_initial = (E_xx_initial + E_yy_initial) / 2 + np.sqrt(((E_xx_initial - E_yy_initial) / 2) ** 2 + ((E_xy_initial + E_yx_initial) / 2) ** 2)
        E2_initial = (E_xx_initial + E_yy_initial) / 2 - np.sqrt(((E_xx_initial - E_yy_initial) / 2) ** 2 + ((E_xy_initial + E_yx_initial) / 2) ** 2)
        # KHZ 250318: Corrected the calculation of principal strains


        # Find maximum and minimum absolute strain values
        max_initial_strain = max(np.max(np.abs(E1_initial)), np.max(np.abs(E2_initial)))
        min_initial_strain = min(np.min(np.abs(E1_initial)), np.min(np.abs(E2_initial)))

        # Store initial strain tensor
        initial_strain_tensor = {
            'E_xx': E_xx_initial,
            'E_yy': E_yy_initial,
            'E_xy': E_xy_initial,
            'E1': E1_initial,
            'E2': E2_initial,
            'min_abs_strain': min_initial_strain,
            'max_abs_strain': max_initial_strain
        }

        # If initial strain is already within bounds, no need to iterate
        if (max_initial_strain <= strain_upper_bound) and (min_initial_strain >= strain_lower_bound):
            return dx, dy, initial_strain_tensor, initial_strain_tensor, max_initial_strain, max_initial_strain, min_initial_strain, min_initial_strain

        # Otherwise, proceed with iterative adjustment
        iterations = 0
        max_strain = max_initial_strain
        min_strain = min_initial_strain
        prev_max_strain = float('inf')
        prev_min_strain = 0

        # Initialize strain tensor components for the loop
        E_xx = E_xx_initial.copy()
        E_yy = E_yy_initial.copy()
        E_xy = E_xy_initial.copy()
        E1 = E1_initial.copy()
        E2 = E2_initial.copy()

        while ((max_strain > strain_upper_bound) or (min_strain < strain_lower_bound)) and (iterations < max_iterations):
            # Determine whether to reduce or amplify displacements
            if max_strain > strain_upper_bound:
                # Reduce displacements if above upper bound
                adjustment_factor = reduction_factor
            elif min_strain < strain_lower_bound:
                # Amplify displacements if below lower bound
                adjustment_factor = amplification_factor
            else:
                # This shouldn't happen due to the while condition, but just in case
                break

            # Apply adjustment
            dx *= adjustment_factor
            dy *= adjustment_factor

            # Recalculate displacement gradients
            dudx = np.zeros_like(dx)
            dudy = np.zeros_like(dx)
            dvdx = np.zeros_like(dx)
            dvdy = np.zeros_like(dx)

            # Interior points (central differences)
            dudx[1:-1, 1:-1] = (dx[1:-1, 2:] - dx[1:-1, :-2]) / 2
            dudy[1:-1, 1:-1] = (dx[2:, 1:-1] - dx[:-2, 1:-1]) / 2
            dvdx[1:-1, 1:-1] = (dy[1:-1, 2:] - dy[1:-1, :-2]) / 2
            dvdy[1:-1, 1:-1] = (dy[2:, 1:-1] - dy[:-2, 1:-1]) / 2

            # Edges (forward/backward differences)
            # Left edge
            dudx[:, 0] = dx[:, 1] - dx[:, 0]
            dvdx[:, 0] = dy[:, 1] - dy[:, 0]
            # Right edge
            dudx[:, -1] = dx[:, -1] - dx[:, -2]
            dvdx[:, -1] = dy[:, -1] - dy[:, -2]
            # Top edge
            dudy[0, :] = dx[1, :] - dx[0, :]
            dvdy[0, :] = dy[1, :] - dy[0, :]
            # Bottom edge
            dudy[-1, :] = dx[-1, :] - dx[-2, :]
            dvdy[-1, :] = dy[-1, :] - dy[-2, :]

            # Calculate Eulerian strain tensor components
            # E = 1/2 * (∇u + ∇u^T + ∇u^T∇u)
            E_xx = 0.5 * (2*dudx + dudx**2 + dvdx**2)
            E_yy = 0.5 * (2*dvdy + dudy**2 + dvdy**2)
            E_xy = 0.5 * (dudy + dvdx + dudx*dudy + dvdx*dvdy)

            # Calculate principal strains
            avg_normal_strain = (E_xx + E_yy) / 2
            diff_normal_strain = (E_xx - E_yy) / 2
            radius = np.sqrt(diff_normal_strain**2 + E_xy**2)

            E1 = avg_normal_strain + radius  # Maximum principal strain
            E2 = avg_normal_strain - radius  # Minimum principal strain

            # Find maximum and minimum absolute strain values
            max_strain = max(np.max(np.abs(E1)), np.max(np.abs(E2)))
            min_strain = min(np.min(np.abs(E1)), np.min(np.abs(E2)))

            # Check convergence
            if (abs(max_strain - prev_max_strain) < tolerance and
                abs(min_strain - prev_min_strain) < tolerance):
                break

            prev_max_strain = max_strain
            prev_min_strain = min_strain
            iterations += 1

        # Prepare final strain tensor
        final_strain_tensor = {
            'E_xx': E_xx,
            'E_yy': E_yy,
            'E_xy': E_xy,
            'E1': E1,
            'E2': E2,
            'min_abs_strain': min_strain,
            'max_abs_strain': max_strain
        }

        return dx, dy, initial_strain_tensor, final_strain_tensor, max_initial_strain, max_strain, min_initial_strain, min_strain




def create_interactive_plots(data, sample_idx, MODEL_TESTING_PATH):
    """
    Create interactive plots with core images, strain analysis, and strain overlays.

    Parameters:
    -----------
    data : dict
        Dictionary containing:
        - 'moving': Moving images (numpy array).
        - 'fixed': Fixed images (numpy array).
        - 'warped': Warped images (numpy array).
        - 'displacements': Displacement fields (numpy array).
    sample_idx : int, optional
        Index of the sample to plot (default: 0).

    Returns:
    --------
    None
        Displays the plots.
    """
    # Extract data for the selected sample
    moving = data['moving']
    fixed = data['fixed']
    warped = data['warped']
    disp = data['displacements']

    # Calculate strain using the displacement fields
    result = limit_strain_range(disp[..., 0], disp[..., 1], strain_upper_bound=1, stretch=False)
    dx, dy, initial_strain_tensor, final_strain_tensor, max_initial_strain, max_strain, min_initial_strain, min_strain = result
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    im1 = ax[0].imshow(final_strain_tensor['E1'], cmap='jet', vmin=-0.4, vmax=0.4)
    ax[0].set_title('E1 Strain')
    fig.colorbar(im1, ax=ax[0])

    im2 = ax[1].imshow(final_strain_tensor['E2'], cmap='jet', vmin=-0.4, vmax=0.4)
    ax[1].set_title('E2 Strain')
    fig.colorbar(im2, ax=ax[1])

    im3 = ax[2].imshow(disp[...,0]+disp[...,1], cmap='jet')
    ax[2].set_title('Displacement')
    fig.colorbar(im3, ax=ax[2])
    plt.show()
    # Create a figure with 3 rows and 3 columns
    fig, axes = plt.subplots(3, 5, figsize=(40, 21), constrained_layout=True)
    fig.suptitle(f"Sample {sample_idx} Analysis", fontsize=20, y=1.02)

    # --- First Row: Core Images ---
    images = [moving, fixed, warped]
    titles = ["Moving Image", "Fixed Image", "Warped Image"]

    Current_Row=0

    for i, (img, title) in enumerate(zip(images, titles)):
        axes[Current_Row, i].imshow(img, cmap='gray')
        axes[Current_Row, i].set_title(title, fontsize=16)
        axes[Current_Row, i].axis('off')

    # Create RGB image: R and G from warped, B from fixed
    warped_norm = (warped - warped.min()) / (np.ptp(warped))
    fixed_norm = (fixed - fixed.min()) / (np.ptp(fixed))
    moving_norm = (moving - moving.min()) / (np.ptp(moving))

    rgb_wrpd_fxd = np.stack([
        warped_norm,      # Red channel
        fixed_norm,      # Green channel
        fixed_norm        # Blue channel
    ], axis=-1)

    axes[Current_Row, 3].imshow(rgb_wrpd_fxd)
    axes[Current_Row, 3].set_title("Warped (Red) over Fixed (RGB)", fontsize=20)
    axes[Current_Row, 3].axis('off')

    rgb_mvg_fxd = np.stack([
        moving_norm,      # Red channel
        fixed_norm,      # Green channel
        fixed_norm        # Blue channel
    ], axis=-1)

    axes[Current_Row, 4].imshow(rgb_mvg_fxd)
    axes[Current_Row, 4].set_title("Moving (Red) over Fixed (RGB)", fontsize=20)
    axes[Current_Row, 4].axis('off')


    # --- Second Row: Strain Analysis (Heatmaps) ---
    Current_Row=2
    # Auto-adjust color limits for E1 and E2 strains
    strain_min = min(np.min(final_strain_tensor['E1']), np.min(final_strain_tensor['E2']))
    strain_max = max(np.max(final_strain_tensor['E1']), np.max(final_strain_tensor['E2']))
    abs_max = max(abs(strain_min), abs(strain_max))
    vmin, vmax = -abs_max, abs_max  # Symmetric colormap
    vmin, vmax = -0.5, 0.5  # Symmetric colormap

    strain_images = [final_strain_tensor['E1'], final_strain_tensor['E2']]
    strain_titles = ["Final E1 Strain", "Final E2 Strain"]


    for i, (strain_img, title) in enumerate(zip(strain_images, strain_titles)):
        im = axes[Current_Row, i].imshow(strain_img[150:350,150:350], cmap='jet', vmin=vmin, vmax=vmax)
        axes[Current_Row, i].set_title(title, fontsize=16)
        axes[Current_Row, i].axis('off')
        add_colorbar(fig, axes[Current_Row, i], im, label="Strain (unitless)")

    # Warped Difference Image (Use Signed Differences)
    diff = fixed - warped
    im6 = axes[Current_Row, 2].imshow(diff, cmap='bwr', vmin=-np.max(np.abs(diff)), vmax=np.max(np.abs(diff)))
    axes[Current_Row, 2].set_title("Warped Difference", fontsize=16)
    axes[Current_Row, 2].axis('off')
    add_colorbar(fig, axes[Current_Row, 2], im6, label="Intensity Difference")

    axes[Current_Row, 3].axis('off')
    axes[Current_Row, 4].axis('off')



    # --- Third Row: Strain Overlays on Fixed Image ---
    Current_Row=1
    overlay_titles = ["E1 Strain Overlay", "E2 Strain Overlay"]

    for i, (strain_img, title) in enumerate(zip(strain_images, overlay_titles)):
        # Display fixed image in grayscale
        axes[Current_Row, i].imshow(fixed, cmap='gray', alpha=0.95)
        # Overlay strain with semi-transparency
        im_overlay = axes[Current_Row, i].imshow(strain_img, cmap='jet', alpha=0.5, vmin=vmin, vmax=vmax)
        axes[Current_Row, i].set_title(title, fontsize=16)
        axes[Current_Row, i].axis('off')
        add_colorbar(fig, axes[Current_Row, i], im_overlay, label="Strain (unitless)")

    # Compute local absolute error
    error_map = np.abs(fixed_norm - warped_norm)

    im = axes[Current_Row, 3].imshow(error_map, cmap='hot')
    axes[Current_Row, 3].set_title("F-W Local Registration Error Heatmap", fontsize=16)
    axes[Current_Row, 3].axis('off')
    add_colorbar(fig, axes[Current_Row, 3], im, label="Absolute Intensity Difference")

    error_map = np.abs(fixed_norm - moving_norm)
    im = axes[Current_Row, 4].imshow(error_map, cmap='hot')
    axes[Current_Row, 4].set_title("F-M Local Registration Error Heatmap", fontsize=16)
    axes[Current_Row, 4].axis('off')
    add_colorbar(fig, axes[Current_Row, 4], im, label="Absolute Intensity Difference")



    axes[Current_Row, 2].axis('off')


    plt.show()


def add_colorbar(fig, ax, im, label=""):
    """
    Add a standardized colorbar to a plot axis.

    Parameters:
    -----------
    fig : matplotlib.figure.Figure
        The figure containing the plot.
    ax : matplotlib.axes.Axes
        The axis to which the colorbar will be added.
    im : matplotlib.image.AxesImage
        The image for which the colorbar is created.
    label : str, optional
        Label for the colorbar.

    Returns:
    --------
    None
        Adds a colorbar to the specified axis.
    """
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.set_ylabel(label, fontsize=12)
    cbar.ax.tick_params(labelsize=10)

def apply_displacement( image, x_displacement, y_displacement):
        # Prepare meshgrid for remap
        height, width = image.shape
        x, y = np.meshgrid(np.arange(width), np.arange(height))

        # Apply displacement (scale the displacements for more visible effect)
        x_new = (x + x_displacement).astype(np.float32)
        y_new = (y + y_displacement).astype(np.float32)
        # convert image tensor to numpy
        # image = image.numpy()
        

        # Warp the image using remap for both x and y displacements
        displaced_image = cv2.remap(image, x_new, y_new, interpolation=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT)
        return displaced_image

frame1 = np.load("/Users/ahmed_ali/Documents/GitHub/GP-2025-Strain/Code/Wave_SimulatorV2/generatedData/Frames/patient080_frame01_slice_3_ACDC_#3_1.npy")
frame2 = np.load("/Users/ahmed_ali/Documents/GitHub/GP-2025-Strain/Code/Wave_SimulatorV2/generatedData/Frames/patient080_frame01_slice_3_ACDC_#3_1.npy")

# frame1_e = np.expand_dims(frame1, axis=-1)
displacement_x = np.load("/Users/ahmed_ali/Documents/GitHub/GP-2025-Strain/Code/Wave_SimulatorV2/generatedData/Displacements/patient080_frame01_slice_3_ACDC_#3_x.npy")
displacement_y =  np.load("/Users/ahmed_ali/Documents/GitHub/GP-2025-Strain/Code/Wave_SimulatorV2/generatedData/Displacements/patient080_frame01_slice_3_ACDC_#3_y.npy")
frame1 = frame1[..., 0]
frame2 = frame2[..., 0]
warped = apply_displacement(frame1, displacement_x, displacement_y)
# image1 = image1[..., 0]
# image2 = image2[..., 0]
# warped = warped[..., 0]
print(frame1.shape)
print(displacement_x.shape)
print(warped.shape)
create_interactive_plots({
    'moving': frame1,
    'fixed': frame2,
    'warped': warped,
    'displacements': np.stack([displacement_x, displacement_y], axis=-1)
}, 0, None)