import numpy as np
import matplotlib.pyplot as plt 
def limit_strain_range(displacement_x, displacement_y, stretch, strain_upper_bound, 
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
    E_xx_initial = 0.5 * (2*dudx_initial - dudx_initial**2 - dvdx_initial**2)
    E_yy_initial = 0.5 * (2*dvdy_initial - dudy_initial**2 - dvdy_initial**2)
    E_xy_initial = 0.5 * (dudy_initial + dvdx_initial - dudx_initial*dudy_initial - dvdx_initial*dvdy_initial)
    
    # Calculate principal strains
    avg_normal_strain_initial = (E_xx_initial + E_yy_initial) / 2
    diff_normal_strain_initial = (E_xx_initial - E_yy_initial) / 2
    radius_initial = np.sqrt(diff_normal_strain_initial**2 + E_xy_initial**2)
    
    E1_initial = avg_normal_strain_initial + radius_initial  # Maximum principal strain
    E2_initial = avg_normal_strain_initial - radius_initial  # Minimum principal strain

    E3_initial = 1 / ((1 + E1_initial) * (1 + E2_initial)) - 1
    
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
        'E3': E3_initial,
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
        E_xx = 0.5 * (2*dudx - dudx**2 - dvdx**2)
        E_yy = 0.5 * (2*dvdy - dudy**2 - dvdy**2)
        E_xy = 0.5 * (dudy + dvdx - dudx*dudy - dvdx*dvdy)
        
        # Calculate principal strains
        avg_normal_strain = (E_xx + E_yy) / 2
        diff_normal_strain = (E_xx - E_yy) / 2
        radius = np.sqrt(diff_normal_strain**2 + E_xy**2)
        
        E1 = avg_normal_strain + radius  # Maximum principal strain
        E2 = avg_normal_strain - radius  # Minimum principal strain
        E3 = 1 / ((1 + E1) * (1 + E2)) - 1
        
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
        'E3': E3,
        'min_abs_strain': min_strain,
        'max_abs_strain': max_strain
    }
    
    return dx, dy, initial_strain_tensor, final_strain_tensor, max_initial_strain, max_strain, min_initial_strain, min_strain

def plot_strain_results(initial_strain_tensor, final_strain_tensor, min_initial_strain, max_initial_strain, 
                        min_final_strain, max_final_strain, strain_lower_bound=None, strain_upper_bound=None):
    """
    Plot strain maps and histograms from the strain tensors, including minimum strain information.
    
    Parameters:
    -----------
    initial_strain_tensor : dict
        Dictionary containing initial strain tensor components
    final_strain_tensor : dict
        Dictionary containing final strain tensor components
    min_initial_strain : float
        Minimum initial strain value
    max_initial_strain : float
        Maximum initial strain value
    min_final_strain : float
        Minimum final strain value
    max_final_strain : float
        Maximum final strain value
    strain_lower_bound : float, optional
        Lower strain bound (for reference lines)
    strain_upper_bound : float, optional
        Upper strain bound (for reference lines)
    """
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # Increase spacing between subplots
    plt.subplots_adjust(hspace=0.5, wspace=0.4)
    
    # Get the principal strain maps from each tensor
    initial_strain_map = initial_strain_tensor['E1']
    final_strain_map = final_strain_tensor['E1']
    
    # Create a common color scale for comparison
    vmin = min(np.min(initial_strain_map), np.min(final_strain_map))
    vmax = max(np.max(initial_strain_map), np.max(final_strain_map))
    # Define custom colormap with zero as green

    # Plot initial strain map
    im1 = axes[0, 0].imshow(initial_strain_map, vmin=vmin, vmax=vmax, cmap='viridis')
    # axes[0, 0].set_title(f'Initial Strain (Range: {min_initial_strain:.4f} to {max_initial_strain:.4f})', 
    #                      pad=20, fontsize=12)
    plt.colorbar(im1, ax=axes[0, 0], pad=0.1)
    
    # Plot final strain map
    im2 = axes[0, 1].imshow(final_strain_map, vmin=vmin, vmax=vmax, cmap='viridis')
    # axes[0, 1].set_title(f'Final Strain (Range: {min_final_strain:.4f} to {max_final_strain:.4f})', 
    #                      pad=20, fontsize=12)
    plt.colorbar(im2, ax=axes[0, 1], pad=0.1)
    
    # Plot initial histogram
    n, bins, patches = axes[1, 0].hist(np.abs(initial_strain_map.flatten()), bins=50, color='blue', alpha=0.7)
    axes[1, 0].set_title('Initial Strain Histogram ', pad=20, fontsize=12)
    axes[1, 0].set_xlabel('Strain Value')
    axes[1, 0].set_ylabel('Frequency')
    
    if strain_upper_bound is not None:
        axes[1, 0].axvline(x=strain_upper_bound, color='purple', linestyle='-', 
                           label=f'Upper Bound: {strain_upper_bound:.4f}')
    axes[1, 0].legend()
    
    # Plot final histogram
    n, bins, patches = axes[1, 1].hist(np.abs(final_strain_map.flatten()), bins=50, color='green', alpha=0.7)
    axes[1, 1].set_title('Final Strain Histogram', pad=20, fontsize=12)
    axes[1, 1].set_xlabel('Strain Value')
    axes[1, 1].set_ylabel('Frequency')
    if strain_upper_bound is not None:
        axes[1, 1].axvline(x=strain_upper_bound, color='purple', linestyle='-', 
                           label=f'Upper Bound: {strain_upper_bound:.4f}')
    
    
    ## Set common x-axis limits for histograms after 0.0(Background)##
    
    # axes[1,1].set_xlim(0.01,0.5)
    # axes[1,1].set_ylim(0,1000)
    # axes[1,0].set_xlim(0.01,0.5)
    # axes[1,0].set_ylim(0,1000)
    
    axes[1, 1].legend()
    
    # Adjust layout with padding
    plt.tight_layout(pad=3.0)
    plt.show()


# Example usage
if __name__ == "__main__":
    
    # Set strain peak limit
    strain_peak = 0.1

    # displacement_x = np.load("/Users/osama/GP-2025-Strain/Data/ACDC/Simulated Data 04-03-2025/2-steps validations/Displacements_loc/patient031_frame10_slice_8_ACDC_#1_x.npy")
    # displacement_y = np.load("/Users/osama/GP-2025-Strain/Data/ACDC/Simulated Data 04-03-2025/2-steps validations/Displacements_loc/patient031_frame10_slice_8_ACDC_#1_y.npy")
    displacement_x = np.load("Data/ACDC/Simulated Data 04-03-2025/Post-simulation validation/Displacements_loc/patient029_frame12_slice_8_ACDC_#1_x.npy")
    displacement_y = np.load("Data/ACDC/Simulated Data 04-03-2025/Post-simulation validation/Displacements_loc/patient029_frame12_slice_8_ACDC_#1_y.npy")

    # Apply strain limiting
    strain_lower_bound = 0.1    # Minimum desired strain
    strain_upper_bound = 0.25  # Maximum allowable strain
    
    # Apply strain limiting/amplification
    result = limit_strain_range(
        displacement_x, displacement_y, 
        stretch=False, strain_upper_bound= strain_upper_bound
    )

    dx_adj, dy_adj, initial_strain, final_strain, max_initial, max_final, min_initial, min_final = result
    
    print(f"Original max displacement: x={np.max(np.abs(displacement_x)):.6f}, y={np.max(np.abs(displacement_y)):.6f}")
    print(f"Adjusted max displacement: x={np.max(np.abs(dx_adj)):.6f}, y={np.max(np.abs(dy_adj)):.6f}")
    print(f"Initial max strain: {max_initial:.6f}")
    print(f"Final max strain: {max_final:.6f} (target: {strain_peak})")
    
    # Check if reduction was needed
    if max_initial <= strain_peak:
        print("No reduction needed, initial strain already below peak")
    else:
        reduction_percentage = (1 - np.max(np.abs(dx_adj)) / np.max(np.abs(displacement_x))) * 100
        print(f"Displacement reduced by approximately {reduction_percentage:.2f}%")


    plot_strain_results(
        initial_strain, final_strain, 
        min_initial, max_initial, 
        min_final, max_final,
        strain_lower_bound, strain_upper_bound
        )
