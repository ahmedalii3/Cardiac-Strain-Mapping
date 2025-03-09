import numpy as np
def strain_conversion(frame_displ_x, frame_displ_y):
    """
    Calculate strain tensors
    
    Parameters:
    frame_displ_x (np.ndarray): X-displacement field (2D array)
    frame_displ_y (np.ndarray): Y-displacement field (2D array)
    delta_x (float): X-direction spatial step
    delta_y (float): Y-direction spatial step
    strain_ep_peak (float): Maximum allowable strain threshold
    
    Returns:
    tuple: frame_displ_x, frame_displ_y, max_displ, min_displ, max_ep, strain_map
    """

    delta_x = 1.0
    delta_y = 1.0

    ux = frame_displ_x
    uy = frame_displ_y

    # Compute spatial gradients
    ux_y, ux_x = np.gradient(ux, delta_y, delta_x)
    uy_y, uy_x = np.gradient(uy, delta_y, delta_x)

    # Compute strain components
    exx_all = (2 * ux_x - (ux_x**2 + uy_x**2)) / 2
    exy_all = (ux_y + uy_x - (ux_x * ux_y + uy_x * uy_y)) / 2
    eyx_all = exy_all  # Symmetric tensor
    eyy_all = (2 * uy_y - (ux_y**2 + uy_y**2)) / 2

    # Compute principal strains
    mean_strain = (exx_all + eyy_all) / 2
    strain_diff = ((exx_all - eyy_all) / 2)**2 + ((exy_all + eyx_all) / 2)**2

    ep1_all = mean_strain + np.sqrt(strain_diff)
    ep2_all = mean_strain - np.sqrt(strain_diff)

    # Ensure ep1 is the largest and ep2 is the smallest
    ep1_all, ep2_all = np.maximum(ep1_all, ep2_all), np.minimum(ep1_all, ep2_all)

    # Incompressibility condition for ep3
    ep3_all = 1 / ((1 + ep1_all) * (1 + ep2_all)) - 1

    # Ensure ep1_all is non-negative
    # ep1_all = np.maximum(ep1_all, 0)
    # ep2_all = np.minimum(ep2_all, 0)


    # Check if strains exceed threshold
    max_ep = max(
        np.max(np.abs(ep1_all)),
        np.max(np.abs(ep2_all)),
        np.max(np.abs(ep3_all))
    )


    strain_map = ep1_all

    # Compute displacement bounds
    max_displ = max(np.max(ux), np.max(uy))
    min_displ = min(np.min(ux), np.min(uy))

    return frame_displ_x, frame_displ_y, max_displ, min_displ, max_ep, strain_map
    
