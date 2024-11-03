import numpy as np
# from calc_wave import sign_grid
def calc_wave(H0, W, time=0, Grid_Sign=None):
    """
    Calculates the wave height Z based on the wave coefficients H0 and W at a given time.
    
    Parameters:
        H0 (ndarray): Wave height coefficients initialized by initialize_wave.
        W  (ndarray): Deep water frequencies initialized by initialize_wave.
        time (float, optional): Time at which the wave height is calculated (default is 0).
        Grid_Sign (ndarray, optional): Optional grid sign for the wave simulation.
        
    Returns:
        Z (ndarray): Real part of the inverse FFT of the wave heights.
    """
    if Grid_Sign is None:
        Grid_Sign = sign_grid(H0.shape[0])  # Default grid sign if not provided

    wt = np.exp(1j * W * time)
    Ht = H0 * wt + np.conj(np.rot90(H0, 2)) * np.conj(wt)

    Z = np.real(np.fft.ifft2(Ht) * Grid_Sign)
    return Z
