import numpy as np

def calculate_strain_and_displacement(frame_displ_x, frame_displ_y, delta_x, delta_y, strain_ep_peak, cine_no_frames):
    """
    Calculate strain tensors and adjust displacements to maintain strain thresholds.
    
    Parameters:
    frame_displ_x (np.ndarray): X-displacement field (2D array)
    frame_displ_y (np.ndarray): Y-displacement field (2D array)
    delta_x (float): X-direction spatial step
    delta_y (float): Y-direction spatial step
    strain_ep_peak (float): Maximum allowable strain threshold
    cine_no_frames (int): Number of frames (not used in current implementation)
    
    Returns:
    tuple: Updated frame_displ_x, frame_displ_y, max_displ, min_displ
    """
    delta_t = 1
    over_strain = True
    
    while over_strain:
        # UX, UY are the x and y displacements at specific spatial points
        ux = frame_displ_x
        uy = frame_displ_y
        
        # Calculate spatial gradients
        # For 2D arrays, np.gradient returns (y_grad, x_grad)
        ux_y, ux_x = np.gradient(ux, delta_y, delta_x)
        uy_y, uy_x = np.gradient(uy, delta_y, delta_x)
        
        # Calculate Eulerian Strain components
        # Based on left Cauchy-Green strain tensor E=I-FinvT*Finv
        # where Finv=I-dU
        exx_all = (2 * ux_x - (ux_x**2 + uy_x**2)) / 2
        exy_all = (ux_y + uy_x - (ux_x * ux_y + uy_x * uy_y)) / 2
        eyx_all = exy_all  # Symmetric tensor
        eyy_all = (2 * uy_y - (ux_y**2 + uy_y**2)) / 2
        
        # Calculate principal strains
        # Principal orientation angle
        theta_ep = 0.5 * np.arctan2(2 * (exy_all + eyx_all) / 2, (exx_all - eyy_all))
        
        # Principal strains calculation
        mean_strain = (exx_all + eyy_all) / 2
        strain_diff = ((exx_all - eyy_all) / 2)**2 + ((exy_all + eyx_all) / 2)**2
        
        ep1_all = mean_strain + np.sqrt(strain_diff)
        ep2_all = mean_strain - np.sqrt(strain_diff)
        
        # Through-plane principal strain using incompressibility
        ep3_all = ep1_all.copy()
        ep1_all = np.maximum(ep1_all, ep2_all)
        ep2_all = np.minimum(ep2_all, ep3_all)
        ep3_all = 1 / ((1 + ep1_all) * (1 + ep2_all)) - 1
        
        # Check if strains exceed threshold
        max_ep = max(
            np.max(np.abs(ep1_all)),
            np.max(np.abs(ep2_all)),
            np.max(np.abs(ep3_all))
        )
        
        if max_ep > strain_ep_peak:
            scale_factor = max(0.95, strain_ep_peak / max_ep)
            frame_displ_x *= scale_factor
            frame_displ_y *= scale_factor
            over_strain = True
        else:
            over_strain = False
    
    # Calculate maximum and minimum displacements
    max_displ = max(np.max(ux), np.max(uy))
    min_displ = min(np.min(ux), np.min(uy))
    
    return frame_displ_x, frame_displ_y, max_displ, min_displ,max_ep