import numpy as np

def polar_cartesian_conversion(frame_displ_x, frame_displ_y, heart_center):
    """
    Converts frame displacements from polar to Cartesian coordinates in-place.
    
    Parameters:
    -----------
    frame_displ_x : numpy.ndarray
        First wave generator output (radial direction) - modified in-place
    frame_displ_y : numpy.ndarray
        Second wave generator output (theta direction) - modified in-place
    heart_center : list or tuple or numpy.ndarray
        Center of the heart [y, x]
    """
    # Fixed image size and delta values
    size_image = 512
    delta_x = 1
    delta_y = 1
    
    # Generate coordinate arrays
    if size_image % 2 != 0:  # odd
        output_x_coord = np.arange(-(size_image-1)/2, (size_image-1)/2 + 1) * delta_x
        output_y_coord = np.arange(-(size_image-1)/2, (size_image-1)/2 + 1) * delta_y
    else:  # even
        output_x_coord = np.arange(-(size_image)/2, (size_image)/2) * delta_x
        output_y_coord = np.arange(-(size_image)/2, (size_image)/2) * delta_y
    
    # Consider the first output of the wave generator the wave in the radial direction
    frame_displ_rads = frame_displ_x
    
    # Consider the second output of the wave generator the wave in the theta direction
    frame_displ_thta = frame_displ_y
    
    nx = output_x_coord.shape[0]  # Num of Cols in the image
    ny = output_y_coord.shape[0]  # Num of Rows in the image
    
    # Form the coordinate matrices
    x_mat = np.tile(output_x_coord, (ny, 1))  # X grid map in mm
    y_mat = np.tile(output_y_coord.reshape(-1, 1), (1, nx))  # Y grid map in mm
    
    # The center of the heart or the origin of the polar coordinates
    x0 = heart_center[1] * delta_x  # The x-center of the heart mask, location in mm
    y0 = heart_center[0] * delta_y  # The y-center of the heart mask, location in mm
    
    # Creating a new grid with the center at the origin
    y_mat_shft = y_mat - x0
    x_mat_shft = x_mat - y0
    
    # Compute the polar coordinate grids R and Theta, centered around the heart
    r = np.sqrt((y_mat_shft)**2 + (x_mat_shft)**2)  # Radial distance
    theta = np.arctan2(y_mat_shft, x_mat_shft)  # Angle in radians
    
    # theta_scale = ((r + 0.001) / np.max(r + 0.001)) * 3
    # theta_scale[theta_scale > 2] = 2
    
    # radia_scale = -(r + np.finfo(float).eps) / np.max(r) * 1
    # radia_scale[np.abs(radia_scale) < 0.95] = 0.95 * np.sign(radia_scale[np.abs(radia_scale) < 0.95])
    radia_scale = 1
    theta_scale = 1
    
    # Using the old frame_displ_x as the radial direction displacement
    u_r = frame_displ_rads / radia_scale
    
    # Using the old frame_displ_y as the theta direction displacement
    u_theta = frame_displ_thta * theta_scale
    
    # Convert displacements from polar to Cartesian
    u_x = u_r * np.cos(theta) - u_theta * np.sin(theta)
    u_y = u_r * np.sin(theta) + u_theta * np.cos(theta)

    return u_x, u_y