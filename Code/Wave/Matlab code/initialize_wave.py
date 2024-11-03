import numpy as np

def initialize_wave(param):
    """
    Initializes wave height coefficients H0 and W for given parameters.
    
    Parameters:
        param (dict): Dictionary containing wave simulation parameters like
                      meshsize, patchsize, windSpeed, winddir, A, g, etc.
                      
    Returns:
        H0 (ndarray): Wave height coefficients at time t = 0.
        W  (ndarray): Deep water frequencies.
        Grid_Sign (ndarray, optional): Grid sign for wave simulation.
    """
    grid_size = (param['meshsize'], param['meshsize'])
    mesh_lim = np.pi * param['meshsize'] / param['patchsize']
    
    N = np.linspace(-mesh_lim, mesh_lim, param['meshsize'])
    M = np.linspace(-mesh_lim, mesh_lim, param['meshsize'])
    Kx, Ky = np.meshgrid(N, M)

    K = np.sqrt(Kx**2 + Ky**2)  # ||K||
    W = np.sqrt(K * param['g'])  # Deep water frequencies (empirical parameter)

    windx, windy = np.cos(np.deg2rad(param['winddir'])), np.sin(np.deg2rad(param['winddir']))

    P = phillips_spectrum(Kx, Ky, np.array([windx, windy]), param['windSpeed'], param['A'], param['g'])

    H0 = (1/np.sqrt(2)) * (np.random.randn(*grid_size) + 1j * np.random.randn(*grid_size)) * np.sqrt(P)
    
    # Optionally return Grid_Sign if required
    if 'return_grid_sign' in param and param['return_grid_sign']:
        Grid_Sign = sign_grid(param['meshsize'])
        return H0, W, Grid_Sign
    else:
        return H0, W, None


def phillips_spectrum(Kx, Ky, wind_dir, wind_speed, A, g):
    """
    Computes the Phillips spectrum for the given wave parameters.
    
    Parameters:
        Kx, Ky (ndarray): Wave number grids.
        wind_dir (ndarray): Wind direction in Cartesian coordinates (x, y).
        wind_speed (float): Wind speed.
        A (float): Scaling factor for the Phillips spectrum.
        g (float): Gravitational constant.
        
    Returns:
        P (ndarray): Phillips spectrum.
    """
    K = np.sqrt(Kx**2 + Ky**2)
    L = (wind_speed**2) / g  # Wave factor related to wind speed and gravity
    
    K_dot_wind = Kx * wind_dir[0] + Ky * wind_dir[1]
    P = A * np.exp(-1 / (K**2 * L**2)) * (K_dot_wind**2) / (K**4)
    
    # Set spectrum to 0 where K is 0 to avoid division by zero errors
    P[K == 0] = 0
    return P


def sign_grid(meshsize):
    """
    Generates a sign grid (+1/-1) for wave simulations.
    
    Parameters:
        meshsize (int): Size of the mesh grid.
        
    Returns:
        grid_sign (ndarray): Sign grid of size (meshsize, meshsize).
    """
    grid_sign = np.zeros((meshsize, meshsize))
    grid_sign[::2, ::2] = 1
    grid_sign[1::2, 1::2] = 1
    grid_sign[::2, 1::2] = -1
    grid_sign[1::2, ::2] = -1
    return grid_sign
