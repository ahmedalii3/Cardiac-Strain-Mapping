import numpy as np
import os
from dataclasses import dataclass, field
import random
from scipy.ndimage import gaussian_filter
from helper import dilate_mask, dilate_mask_fade, dilate_mask_fade_cosine, dilate_mask_fade_smooth, save_if_not_exists, save_json_array
import shutil
import matplotlib as mpl


ffmpeg_path = shutil.which('ffmpeg')
if ffmpeg_path:
    mpl.rcParams['animation.ffmpeg_path'] = ffmpeg_path
else:
    raise EnvironmentError("ffmpeg not found in PATH. Please install ffmpeg.")

import matplotlib.animation as animation
animation.writers['ffmpeg'] 

@dataclass
class StrainWaveParams:
    """
    A class to store and manage strain wave simulation parameters.

    Attributes:
        meshsize (int): Size of the main grid (NxN).
        patchsize (int): Patch size for wave computations.
        wind_speed (float): Wind speed influencing the wave generation.
        wind_dir_vector (tuple): Wind direction as a unit vector (wind_x, wind_y).
        rng_seed (int): Random seed for reproducibility.
        A (float): Scaling factor for the Phillips spectrum.
        g (float): Gravitational constant (m/s²).
    """
    meshsize: int = 512  # Grid size (NxN)
    patchsize: int = 50  # Patch size
    wind_speed: float = 50  # Wind speed affecting waves
    wind_dir_vector: tuple = (np.cos(np.radians(20)), np.sin(np.radians(20)))  # Wind direction as unit vector
    rng_seed: int = field(default_factory=lambda: random.randint(0, 100000))  # Random seed for reproducibility
    A: float = 1e-7  # Scaling factor for Phillips spectrum
    g: float = 9.81  # Gravitational constant (m/s²)

    def __post_init__(self):
        """Sets the random seed after initialization to ensure repeatable simulations."""
        np.random.seed(self.rng_seed)


###################################################################################################
###################################################################################################
###################################################################################################
def initialize_wave(param):
    """
    Initialize wave height coefficients H0, wave frequency W, and Grid_Sign.

    Args:
        param (StrainWaveParams): The wave parameters.

    Returns:
        H0 (np.ndarray): Complex-valued wave height field at t=0.
        W (np.ndarray): Deep water frequency grid.
        Grid_Sign (np.ndarray): Grid sign array.
    """
    # Set up the grid size
    grid_size = (param.meshsize, param.meshsize)

    # Define the mesh limits
    mesh_lim = np.pi * param.meshsize / param.patchsize
    N = np.linspace(-mesh_lim, mesh_lim, param.meshsize)
    M = np.linspace(-mesh_lim, mesh_lim, param.meshsize)
    Kx, Ky = np.meshgrid(N, M)

    # Compute the magnitude K = ||K||
    K = np.sqrt(Kx**2 + Ky**2)

    # Compute the deep water frequency W
    W = np.sqrt(K * param.g)

    # Compute wind direction components
    windx, windy = param.wind_dir_vector

    # Compute the Phillips spectrum P
    P = phillips_spectrum(Kx, Ky, (windx, windy), param.wind_speed, param.A, param.g)

    # Generate the initial height field H0
    H0 = (1 / np.sqrt(2)) * (np.random.randn(*grid_size) + 1j * np.random.randn(*grid_size)) * np.sqrt(P)

    # Generate Grid_Sign using the checkerboard pattern function
    Grid_Sign = sign_grid(param.meshsize)

    return H0, W, Grid_Sign

###################################################################################################
###################################################################################################
###################################################################################################
def sign_grid(n):
    """
    Generates a checkerboard matrix of size (n, n) with alternating signs.

    Args:
        n (int): Grid size.

    Returns:
        np.ndarray: (n, n) array with alternating +1 and -1 values.
    """
    x, y = np.meshgrid(np.arange(n), np.arange(n))
    sgn = np.ones((n, n))
    sgn[(x + y) % 2 == 0] = -1  # Apply checkerboard pattern
    return sgn

###################################################################################################
###################################################################################################
###################################################################################################
def phillips_spectrum(Kx, Ky, wind_dir_vector, wind_speed, A, g):
    """
    Computes the Phillips spectrum for ocean surface waves given wind and wave parameters.

    The Phillips spectrum models how energy is distributed across different wave numbers.

    Args:
        Kx (np.ndarray): 2D array of wave number x-components.
        Ky (np.ndarray): 2D array of wave number y-components.
        wind_dir_vector (tuple): Wind direction as a unit vector (wind_x, wind_y).
        wind_speed (float): Wind speed, influencing wave heights.
        A (float): Scaling factor for the Phillips spectrum.
        g (float): Gravitational constant (m/s²).

    Returns:
        np.ndarray: A 2D array representing the Phillips spectrum for wave height variance.
    """
    # Compute squared wave number magnitude
    K_sq = Kx**2 + Ky**2
    K = np.sqrt(K_sq)  # Magnitude of wave vector
    K4 = K_sq**2  # Avoid division by zero

    # Extract wind direction vector components
    wind_x, wind_y = wind_dir_vector

    # Compute the dot product between the wind direction and wave vector (normalized)
    WK = (Kx / (K + 1e-8)) * wind_x + (Ky / (K + 1e-8)) * wind_y  # Avoid division by zero

    # Compute characteristic length scale L = (wind_speed²) / g
    L = (wind_speed**2) / g

    # Compute the Phillips spectrum using its standard formulation
    P = A * np.exp(-1.0 / (K_sq * L**2 + 1e-8)) * (WK**2) / (K4 + 1e-8)

    # Set spectrum values to zero where K=0 or where the wind is in the opposite direction (WK < 0)
    P[(K_sq == 0) | (WK < 0)] = 0

    return P

###################################################################################################
###################################################################################################
###################################################################################################
def calc_wave(H0, W, time, Grid_Sign, damping_factor=0.95, random_phase=True):
    """
    Computes the displacement wave surface at a given time step with damping and random phase shifts.

    Args:
        H0 (np.ndarray): Initial complex wave height coefficients.
        W (np.ndarray): Wave frequency grid.
        time (float): Time step value.
        Grid_Sign (np.ndarray): Grid sign array.
        damping_factor (float): Factor to reduce wave amplitude over time.
        random_phase (bool): If True, adds random phase shifts for realism.

    Returns:
        np.ndarray: Real-valued displacement wave at the given time.
    """
    # Compute time evolution factor
    wt = np.exp(1j * W * time)

    if random_phase:
        phase_shift = np.exp(1j * np.random.uniform(0, 0.1*np.pi, H0.shape))
        wt *= phase_shift  # Apply a random phase shift

    # Compute conjugate symmetric wave field
    Ht = H0 * wt + np.conj(np.rot90(H0, 2)) * np.conj(wt)

    # Apply damping to simulate energy loss
    Ht *= damping_factor ** time  

    # Apply inverse FFT and Grid_Sign correction
    Z = np.real(np.fft.ifft2(Ht) * Grid_Sign)

    return Z


###################################################################################################
###################################################################################################
###################################################################################################
def compute_displacement_frames(param_x, param_y, num_frames, max_displ_x, max_displ_y, apply_smoothing=True, interaction_factor=0.25):
    """
    Computes displacement wave evolution for both X and Y components.

    Steps:
        1. Compute displacement for X and Y separately.
        2. Apply interaction after both are computed.
        3. Scale displacement frames within limits.

    Args:
        param_x (StrainWaveParams): Wave parameters for X direction.
        param_y (StrainWaveParams): Wave parameters for Y direction.
        num_frames (int): Number of frames in the simulation.
        max_displ_x (float): Maximum displacement for X.
        max_displ_y (float): Maximum displacement for Y.
        interaction_factor (float): Strength of X-Y interaction.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Displacement frames for X and Y.
    """
    # Create 3D arrays to store displacement over time
    displacement_x = np.zeros((param_x.meshsize, param_x.meshsize, num_frames))
    displacement_y = np.zeros((param_y.meshsize, param_y.meshsize, num_frames))

    # Initialize wave parameters for X and Y
    H0_x, W_x, Grid_Sign_x = initialize_wave(param_x)
    # First, compute independent displacement for both X and Y
    for i, time in enumerate(np.linspace(0, num_frames / 3, num_frames)):
        displacement_x[:, :, i] = calc_wave(H0_x, W_x, time, Grid_Sign_x)

    H0_y, W_y, Grid_Sign_y = initialize_wave(param_y)
    for i, time in enumerate(np.linspace(0, num_frames / 3, num_frames)):
        displacement_y[:, :, i] = calc_wave(H0_y, W_y, time, Grid_Sign_y)

    if apply_smoothing:
        displacement_x = gaussian_filter(displacement_x, sigma=(15, 15, 0))
        displacement_y = gaussian_filter(displacement_y, sigma=(15, 15, 0))

    # Second, apply interaction after computing both independently
    displacement_x += interaction_factor * displacement_y
    displacement_y += interaction_factor * displacement_x


    # Scale displacement frames to fit within max_displ ranges
    displacement_x = np.clip(displacement_x, -max_displ_x, max_displ_x)
    displacement_y = np.clip(displacement_y, -max_displ_y, max_displ_y)

    return displacement_x, displacement_y


###################################################################################################
###################################################################################################
###################################################################################################
def run_wave_simulation(num_frames=30):
    """
    Runs the full wave simulation, including displacement computation,
    optional Gaussian shape testing, and inversion of displacement direction.

    Args:
        num_frames (int): Number of frames to compute.

    Returns:
        dict: Dictionary containing displacement frames for X and Y.
    """
    # Initialize simulation settings
    wind_dir_x = np.random.uniform(0, 360)  # Random wind direction for X
    wind_dir_y = np.random.uniform(0, 360)  # Random wind direction for Y
    
    wind_speed_x = np.random.uniform(20, 100)  # Random wind speed for X
    wind_speed_y = np.random.uniform(20, 100)  # Random wind speed for Y

    # Convert wind directions to unit vectors
    # Convert wind direction to unit vector (cosine and sine of the angle)
    wind_vec_x = (np.cos(np.radians(wind_dir_x)), np.sin(np.radians(wind_dir_x)))
    wind_vec_y = (np.cos(np.radians(wind_dir_y)), np.sin(np.radians(wind_dir_y)))

    # Initialize simulation settings with different random wind directions
    params_x = StrainWaveParams(wind_dir_vector=wind_vec_x, wind_speed=wind_speed_x)
    params_y = StrainWaveParams(wind_dir_vector=wind_vec_y, wind_speed=wind_speed_y)
    # Define max displacement values
    max_displ_x = 10
    max_displ_y = 10

      # Generate a random interaction factor between 0.1 and 0.5
    interaction_factor = np.random.uniform(0.05, 0.2)
    # Compute wave displacement frames for X and Y directions
    FrameDisplX, FrameDisplY = compute_displacement_frames(params_x, params_y, num_frames, max_displ_x, max_displ_y,interaction_factor)

    # Invert displacement for Y-direction (as per MATLAB code)
    FrameDisplY = -FrameDisplY

    return {
        "FrameDisplX": FrameDisplX,
        "FrameDisplY": FrameDisplY,
        "wind_dir_x": wind_dir_x,
        "wind_dir_y": wind_dir_y,
        "wind_speed_x": wind_speed_x,
        "wind_speed_y": wind_speed_y
    }


###################################################################################################
###################################################################################################
###################################################################################################

def adjust_displacement_for_strain(FrameDisplX, FrameDisplY, deltaX, deltaY, StrainEpPeak, strain_tolerance=0.01, max_iterations=20):
    """
    Adjusts displacement fields to ensure that the computed strain values are as close as possible to StrainEpPeak.
    Uses vectorized operations for efficiency and allows early stopping with strain_tolerance.

    Args:
        FrameDisplX (np.ndarray): X-displacement field (shape: [H, W, Frames])
        FrameDisplY (np.ndarray): Y-displacement field (shape: [H, W, Frames])
        deltaX (float): Spatial resolution in the X direction.
        deltaY (float): Spatial resolution in the Y direction.
        StrainEpPeak (float): Target strain level for normalization.
        strain_tolerance (float): Allowable tolerance for strain values.
        max_iterations (int): Maximum number of iterations to adjust displacement.

    Returns:
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list: 
        Adjusted FrameDisplX, FrameDisplY, Ep1All, Ep2All, Ep3All, and strain history.
    """

    CineNoFrames = FrameDisplX.shape[2]  # Number of frames
    strain_history = []  # To store the maximum strain per frame over iterations

    for iteration in range(max_iterations):
        # Compute gradients over the spatial dimensions (axis 0 and 1)
        UXx, UXy = np.gradient(FrameDisplX, deltaX, deltaY, axis=(0, 1))
        UYx, UYy = np.gradient(FrameDisplY, deltaX, deltaY, axis=(0, 1))

        # Eulerian strain tensor components
        ExxAll = (2 * UXx - (UXx**2 + UYx**2)) / 2
        ExyAll = (UXy + UYx - (UXx * UXy + UYx * UYy)) / 2
        EyyAll = (2 * UYy - (UXy**2 + UYy**2)) / 2

        # Principal strains (in-plane)
        ThetaEp = 0.5 * np.arctan2(2 * ExyAll, ExxAll - EyyAll)
        Ep1All = (ExxAll + EyyAll) / 2 + np.sqrt(((ExxAll - EyyAll) / 2) ** 2 + ExyAll ** 2)
        Ep2All = (ExxAll + EyyAll) / 2 - np.sqrt(((ExxAll - EyyAll) / 2) ** 2 + ExyAll ** 2)

        # Compute the third strain component using incompressibility assumption
        Ep3All = 1 / ((1 + np.maximum(Ep1All, Ep2All)) * (1 + np.minimum(Ep1All, Ep2All))) - 1

        # Compute the maximum absolute strain per frame
        stacked_strains = np.stack([Ep1All, Ep2All, Ep3All])  # Shape: (3, H, W, CineNoFrames)
        max_per_component = np.max(np.abs(stacked_strains), axis=(1, 2))  # Shape: (3, CineNoFrames)
        MaxEp_per_frame = np.max(max_per_component, axis=0)  # Shape: (CineNoFrames,)
        strain_history.append(MaxEp_per_frame.copy())

        # Compute scaling factors for each frame
        scale_factors = StrainEpPeak / (MaxEp_per_frame + np.finfo(float).eps)  # Avoid division by zero

        # Clip scaling factors to avoid excessive changes in one iteration
        scale_factors = np.clip(scale_factors, 0.95, 1.05)  # Now allows scaling UP (1.05 max)
        scale_factors = scale_factors.reshape(1, 1, CineNoFrames)  # Reshape for broadcasting
        
        # print(scale_factors)

        # Apply scaling to adjust the displacement fields
        FrameDisplX *= scale_factors
        FrameDisplY *= scale_factors

        UXx, UXy = np.gradient(FrameDisplX, deltaX, deltaY, axis=(0, 1))
        UYx, UYy = np.gradient(FrameDisplY, deltaX, deltaY, axis=(0, 1))

        # Eulerian strain tensor components
        ExxAll = (2 * UXx - (UXx**2 + UYx**2)) / 2
        ExyAll = (UXy + UYx - (UXx * UXy + UYx * UYy)) / 2
        EyyAll = (2 * UYy - (UXy**2 + UYy**2)) / 2

        # Principal strains (in-plane)
        ThetaEp = 0.5 * np.arctan2(2 * ExyAll, ExxAll - EyyAll)
        Ep1All = (ExxAll + EyyAll) / 2 + np.sqrt(((ExxAll - EyyAll) / 2) ** 2 + ExyAll ** 2)
        Ep2All = (ExxAll + EyyAll) / 2 - np.sqrt(((ExxAll - EyyAll) / 2) ** 2 + ExyAll ** 2)

        # Compute the third strain component using incompressibility assumption
        Ep3All = 1 / ((1 + np.maximum(Ep1All, Ep2All)) * (1 + np.minimum(Ep1All, Ep2All))) - 1

        # Check if the strain in all frames is within the tolerance
        if np.all(np.abs(MaxEp_per_frame - StrainEpPeak) < strain_tolerance):
            # print(f"Converged in {iteration + 1} iterations")
            break

    return FrameDisplX, FrameDisplY, Ep1All, Ep2All, Ep3All, strain_history

###################################################################################################
###################################################################################################
###################################################################################################

def enforce_full_principal_strain_order(Ep1All, Ep2All, Ep3All=None):
    """
    Ensure Ep1All >= Ep2All >= Ep3All at every voxel (pixel) location.
    Sorts the three principal strains per point.

    Args:
        Ep1All (np.ndarray): First principal strain field.
        Ep2All (np.ndarray): Second principal strain field.
        Ep3All (np.ndarray): Third principal strain field (incompressibility strain).

    Returns:
        Ep1_sorted (np.ndarray): Largest principal strain.
        Ep2_sorted (np.ndarray): Middle principal strain.
        Ep3_sorted (np.ndarray): Smallest principal strain.
    """

    if Ep3All is not None:
        # Stack all principal strains along a new axis
        strain_stack = np.stack([Ep1All, Ep2All, Ep3All], axis=0)  # Shape (3, H, W, T)
    else:
        # Stack only the first two principal strains
        strain_stack = np.stack([Ep1All, Ep2All, Ep2All], axis=0) # Shape (2, H, W, T)
    # Sort along the new axis (axis=0) descending
    strain_sorted = np.sort(strain_stack, axis=0)[::-1, ...]  # Reverse to get descending

    Ep1_sorted = strain_sorted[0]
    Ep2_sorted = strain_sorted[1]
    Ep3_sorted = strain_sorted[2]

    return Ep1_sorted, Ep2_sorted, Ep3_sorted


def compute_strains(FrameDisplX, FrameDisplY, deltaX, deltaY):
    """
    Compute principal strains (Ep1, Ep2) and incompressibility strain (Ep3) 
    from displacement fields.

    Args:
        FrameDisplX (np.ndarray): X displacement field (shape: H, W, T).
        FrameDisplY (np.ndarray): Y displacement field (shape: H, W, T).
        deltaX (float): Pixel spacing in the X direction (mm).
        deltaY (float): Pixel spacing in the Y direction (mm).

    Returns:
        Ep1All (np.ndarray): Principal strain 1 (shape: H, W, T).
        Ep2All (np.ndarray): Principal strain 2 (shape: H, W, T).
        Ep3All (np.ndarray): Incompressibility strain (shape: H, W, T).
    """
    # Compute spatial gradients
    UXx, UXy = np.gradient(FrameDisplX, deltaX, deltaY, axis=(0, 1))
    UYx, UYy = np.gradient(FrameDisplY, deltaX, deltaY, axis=(0, 1))

    # Compute Eulerian strain tensor components
    ExxAll = (2 * UXx - (UXx**2 + UYx**2)) / 2
    ExyAll = (UXy + UYx - (UXx * UXy + UYx * UYy)) / 2
    EyyAll = (2 * UYy - (UXy**2 + UYy**2)) / 2

    # Compute principal strains
    Ep1All = (ExxAll + EyyAll) / 2 + np.sqrt(((ExxAll - EyyAll) / 2) ** 2 + ExyAll ** 2)
    Ep2All = (ExxAll + EyyAll) / 2 - np.sqrt(((ExxAll - EyyAll) / 2) ** 2 + ExyAll ** 2)

    Ep1All, Ep2All, _ = enforce_full_principal_strain_order(Ep1All, Ep2All)

    # Compute incompressibility strain using the determinant rule
    Ep3All = 1 / ((1 + np.maximum(Ep1All, Ep2All)) * (1 + np.minimum(Ep1All, Ep2All))) - 1



    return Ep1All, Ep2All, Ep3All


def adjust_displacement_with_ring(
    FrameDisplX, FrameDisplY, deltaX, deltaY, StrainEpPeak,
    xMat_shft, yMat_shft,
    inner_radius=0, outer_radius=np.inf,
    strain_tolerance=0.01, max_iterations=20
):
    """
    Adjusts displacement fields to match a target peak strain, using a ring mask to exclude
    both the center and edge regions from peak strain detection.

    Parameters:
        FrameDisplX (np.ndarray): X displacement field of shape (H, W, T).
        FrameDisplY (np.ndarray): Y displacement field of shape (H, W, T).
        deltaX (float): Spatial resolution in X direction (mm).
        deltaY (float): Spatial resolution in Y direction (mm).
        StrainEpPeak (float): Target maximum strain to normalize to.
        xMat_shft (np.ndarray): Shifted X coordinate grid (H, W), centered at heart.
        yMat_shft (np.ndarray): Shifted Y coordinate grid (H, W), centered at heart.
        inner_radius (float): Inner exclusion radius (in mm) from heart center.
        outer_radius (float): Outer exclusion radius (in mm) from heart center.
        strain_tolerance (float): Convergence threshold for strain matching.
        max_iterations (int): Maximum number of scaling iterations.

    Returns:
        Tuple of:
            FrameDisplX (np.ndarray): Adjusted X displacement.
            FrameDisplY (np.ndarray): Adjusted Y displacement.
            Ep1All (np.ndarray): Principal strain 1 over time.
            Ep2All (np.ndarray): Principal strain 2 over time.
            Ep3All (np.ndarray): Incompressibility strain over time.
            strain_history (List[np.ndarray]): Max strain per frame per iteration.
    """
    CineNoFrames = FrameDisplX.shape[2]
    strain_history = []

    # Create the ring mask based on radial distance
    R = np.sqrt(xMat_shft**2 + yMat_shft**2)
    ring_mask = ((R >= inner_radius) & (R <= outer_radius)).astype(float)
    ring_mask_3d = np.repeat(ring_mask[:, :, np.newaxis], CineNoFrames, axis=2)

    for _ in range(max_iterations):
        # Compute gradients of displacement fields
        #UXx, UXy = np.gradient(FrameDisplX, deltaX, deltaY, axis=(0, 1))
        #UYx, UYy = np.gradient(FrameDisplY, deltaX, deltaY, axis=(0, 1))

        # Compute Eulerian strain tensor components
        #ExxAll = (2 * UXx - (UXx**2 + UYx**2)) / 2
        #ExyAll = (UXy + UYx - (UXx * UXy + UYx * UYy)) / 2
        #EyyAll = (2 * UYy - (UXy**2 + UYy**2)) / 2

        # Principal strains
        #Ep1All = (ExxAll + EyyAll) / 2 + np.sqrt(((ExxAll - EyyAll) / 2)**2 + ExyAll**2)
        #Ep2All = (ExxAll + EyyAll) / 2 - np.sqrt(((ExxAll - EyyAll) / 2)**2 + ExyAll**2)
        #Ep3All = 1 / ((1 + np.maximum(Ep1All, Ep2All)) * (1 + np.minimum(Ep1All, Ep2All))) - 1
        Ep1All, Ep2All, Ep3All = compute_strains(FrameDisplX, FrameDisplY, deltaX, deltaY)

        # Apply ring mask and calculate peak strain in the region of interest
        Ep_all_masked = np.stack([Ep1All, Ep2All, Ep3All]) * ring_mask_3d
        max_per_component = np.max(np.abs(Ep_all_masked), axis=(1, 2))
        MaxEp_per_frame = np.max(max_per_component, axis=0)
        strain_history.append(MaxEp_per_frame.copy())

        # Compute and apply scaling factors to match target strain
        scale_factors = StrainEpPeak / (MaxEp_per_frame + np.finfo(float).eps)
        scale_factors = np.clip(scale_factors, 0.95, 1.05).reshape(1, 1, CineNoFrames)

        FrameDisplX *= scale_factors
        FrameDisplY *= scale_factors

        # Compute gradients of displacement fields
        UXx, UXy = np.gradient(FrameDisplX, deltaX, deltaY, axis=(0, 1))
        UYx, UYy = np.gradient(FrameDisplY, deltaX, deltaY, axis=(0, 1))

        # Compute Eulerian strain tensor components
        ExxAll = (2 * UXx - (UXx**2 + UYx**2)) / 2
        ExyAll = (UXy + UYx - (UXx * UXy + UYx * UYy)) / 2
        EyyAll = (2 * UYy - (UXy**2 + UYy**2)) / 2

        # Principal strains
        Ep1All = (ExxAll + EyyAll) / 2 + np.sqrt(((ExxAll - EyyAll) / 2)**2 + ExyAll**2)
        Ep2All = (ExxAll + EyyAll) / 2 - np.sqrt(((ExxAll - EyyAll) / 2)**2 + ExyAll**2)
        Ep3All = 1 / ((1 + np.maximum(Ep1All, Ep2All)) * (1 + np.minimum(Ep1All, Ep2All))) - 1

        if np.all(np.abs(MaxEp_per_frame - StrainEpPeak) < strain_tolerance):
            break

    return FrameDisplX, FrameDisplY, Ep1All, Ep2All, Ep3All, strain_history




def adjust_displacement_ignore_center(FrameDisplX, FrameDisplY, deltaX, deltaY, StrainEpPeak,
                                      xMat_shft, yMat_shft, exclusion_radius=20,
                                      strain_tolerance=0.01, max_iterations=20):
    """
    Adjusts displacement fields to match a target peak strain, excluding a circular region
    around the center from peak strain computation.

    Args:
        FrameDisplX, FrameDisplY: 3D displacement fields (H, W, T).
        deltaX, deltaY: pixel spacing in mm.
        StrainEpPeak: target peak strain to scale to.
        xMat_shft, yMat_shft: shifted coordinate grids (centered at heart).
        exclusion_radius: radius (in mm) of disc to exclude from strain peak search.
        strain_tolerance: convergence threshold.
        max_iterations: max adjustment iterations.

    Returns:
        Adjusted FrameDisplX, FrameDisplY, Ep1All, Ep2All, Ep3All, strain_history
    """
    CineNoFrames = FrameDisplX.shape[2]
    strain_history = []

    # Create exclusion mask based on radial distance
    R = np.sqrt(xMat_shft**2 + yMat_shft**2)
    exclusion_mask = (R >= exclusion_radius).astype(float)  # 1 outside exclusion, 0 inside
    exclusion_mask_3d = np.repeat(exclusion_mask[:, :, np.newaxis], CineNoFrames, axis=2)

    for iteration in range(max_iterations):
        UXx, UXy = np.gradient(FrameDisplX, deltaX, deltaY, axis=(0, 1))
        UYx, UYy = np.gradient(FrameDisplY, deltaX, deltaY, axis=(0, 1))

        ExxAll = (2 * UXx - (UXx**2 + UYx**2)) / 2
        ExyAll = (UXy + UYx - (UXx * UXy + UYx * UYy)) / 2
        EyyAll = (2 * UYy - (UXy**2 + UYy**2)) / 2

        Ep1All = (ExxAll + EyyAll) / 2 + np.sqrt(((ExxAll - EyyAll) / 2)**2 + ExyAll**2)
        Ep2All = (ExxAll + EyyAll) / 2 - np.sqrt(((ExxAll - EyyAll) / 2)**2 + ExyAll**2)
        Ep3All = 1 / ((1 + np.maximum(Ep1All, Ep2All)) * (1 + np.minimum(Ep1All, Ep2All))) - 1

        # Masked max strain outside exclusion zone
        Ep_all_masked = np.stack([Ep1All, Ep2All, Ep3All]) * exclusion_mask_3d
        max_per_component = np.max(np.abs(Ep_all_masked), axis=(1, 2))
        MaxEp_per_frame = np.max(max_per_component, axis=0)
        strain_history.append(MaxEp_per_frame.copy())

        scale_factors = StrainEpPeak / (MaxEp_per_frame + np.finfo(float).eps)
        scale_factors = np.clip(scale_factors, 0.95, 1.05).reshape(1, 1, CineNoFrames)

        FrameDisplX *= scale_factors
        FrameDisplY *= scale_factors

        UXx, UXy = np.gradient(FrameDisplX, deltaX, deltaY, axis=(0, 1))
        UYx, UYy = np.gradient(FrameDisplY, deltaX, deltaY, axis=(0, 1))

        ExxAll = (2 * UXx - (UXx**2 + UYx**2)) / 2
        ExyAll = (UXy + UYx - (UXx * UXy + UYx * UYy)) / 2
        EyyAll = (2 * UYy - (UXy**2 + UYy**2)) / 2

        Ep1All = (ExxAll + EyyAll) / 2 + np.sqrt(((ExxAll - EyyAll) / 2)**2 + ExyAll**2)
        Ep2All = (ExxAll + EyyAll) / 2 - np.sqrt(((ExxAll - EyyAll) / 2)**2 + ExyAll**2)
        Ep3All = 1 / ((1 + np.maximum(Ep1All, Ep2All)) * (1 + np.minimum(Ep1All, Ep2All))) - 1
        
        if np.all(np.abs(MaxEp_per_frame - StrainEpPeak) < strain_tolerance):
            # print(f"Converged in {iteration + 1} iterations")
            break

    return FrameDisplX, FrameDisplY, Ep1All, Ep2All, Ep3All, strain_history

###################################################################################################
###################################################################################################
###################################################################################################

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

def animate_strain_histograms(Ep1All, Ep2All, Ep3All, output_filename="strain_histogram_animation.mp4", num_bins=50, bin_range=0.3, save_file=False):
    """
    Creates an animated histogram of strain distributions over time and saves it as an MP4.

    Args:
        Ep1All (np.ndarray): Principal Strain 1 (shape: [H, W, Frames])
        Ep2All (np.ndarray): Principal Strain 2 (shape: [H, W, Frames])
        Ep3All (np.ndarray): Principal Strain 3 (shape: [H, W, Frames])
        output_filename (str): Name of the saved MP4 file.
        num_bins (int): Number of bins for the histogram.

    Returns:
        ani (FuncAnimation): The animation object for display in Jupyter Notebook.
    """
    
    # Compute histogram bins
    bins = np.linspace(-bin_range, bin_range, num_bins)
    
    # Compute histograms for each frame
    hist_data = {
        "Ep1": [np.histogram(Ep1All[:, :, frame], bins=bins)[0] for frame in range(Ep1All.shape[2])],
        "Ep2": [np.histogram(Ep2All[:, :, frame], bins=bins)[0] for frame in range(Ep2All.shape[2])],
        "Ep3": [np.histogram(Ep3All[:, :, frame], bins=bins)[0] for frame in range(Ep3All.shape[2])],
        "bins": bins
    }

    # Set up the figure for animation
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Initialize bars for histograms
    bars = []
    titles = [r"Histogram of $\varepsilon_1$ (Principal Strain 1)", 
              r"Histogram of $\varepsilon_2$ (Principal Strain 2)", 
              r"Histogram of $\varepsilon_3$ (Incompressibility Strain)"]

    # Initialize empty bar plots
    for i, ax in enumerate(axes):
        bar = ax.bar(hist_data["bins"][:-1], np.zeros_like(hist_data["bins"][:-1]), width=0.005, color='blue', alpha=0.6)
        bars.append(bar)
        ax.set_title(titles[i])
        ax.set_xlim(-bin_range, bin_range)
        ax.set_ylim(0, np.max(hist_data["Ep1"]))  # Use max bin count for scaling
        ax.set_xlabel("Strain Value")
        ax.set_ylabel("Frequency")

    plt.tight_layout()

    # Update function for animation
    def update(frame):
        for i, key in enumerate(["Ep1", "Ep2", "Ep3"]):
            for rect, h in zip(bars[i], hist_data[key][frame]):
                rect.set_height(h)  # Update bar height
        return bars

    # Create the animation
    num_frames = Ep1All.shape[2]
    ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=50, blit=False)
    # plt.show()
    if save_file:
        # Save the animation as an MP4 file
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=20, metadata=dict(artist='Strain Histogram Analysis'), bitrate=1800)
        ani.save(output_filename, writer=writer)

        print(f"Animation saved as {output_filename}")

    # # Return the animation object for inline display in Jupyter Notebook
    return ani

###################################################################################################
###################################################################################################
###################################################################################################



def animate_wave_displacement(FrameDisplX, FrameDisplY, wind_dir_x, wind_dir_y, wind_speed_x, wind_speed_y, output_filename="wave_simulation.mp4", save_file=False):
    """
    Creates an animated visualization of X and Y wave displacement over time and saves it as an MP4.

    Args:
        FrameDisplX (np.ndarray): X-displacement field (shape: [H, W, Frames])
        FrameDisplY (np.ndarray): Y-displacement field (shape: [H, W, Frames])
        wind_dir_x (float): Wind direction for X displacement (in degrees).
        wind_dir_y (float): Wind direction for Y displacement (in degrees).
        wind_speed_x (float): Wind speed for X displacement.
        wind_speed_y (float): Wind speed for Y displacement.
        output_filename (str): Name of the saved MP4 file.

    Returns:
        ani (FuncAnimation): The animation object for display in Jupyter Notebook.
    """

    num_frames = FrameDisplX.shape[2]  # Number of frames in the animation

    # Set up the figure for animation
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    # Initialize images
    im_x = ax[0].imshow(FrameDisplX[:, :, 0], cmap='coolwarm', origin='lower', animated=True)
    ax[0].set_title(f"Wave Displacement X\n(Wind Dir: {wind_dir_x:.2f}°, Wind Speed: {wind_speed_x:.2f})")
    ax[0].set_xlabel("X")
    ax[0].set_ylabel("Y")
    fig.colorbar(im_x, ax=ax[0])

    im_y = ax[1].imshow(FrameDisplY[:, :, 0], cmap='coolwarm', origin='lower', animated=True)
    ax[1].set_title(f"Wave Displacement Y\n(Wind Dir: {wind_dir_y:.2f}°, Wind Speed: {wind_speed_y:.2f})")
    ax[1].set_xlabel("X")
    ax[1].set_ylabel("Y")
    fig.colorbar(im_y, ax=ax[1])

    # Update function for animation
    def update(frame):
        im_x.set_array(FrameDisplX[:, :, frame])
        im_y.set_array(FrameDisplY[:, :, frame])
        return im_x, im_y

    # Create the animation
    ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=50, blit=False)
    # plt.show()
    if save_file:
        # Save the animation as an MP4 file
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=20, metadata=dict(artist='Wave Simulation'), bitrate=1800)
        ani.save(output_filename, writer=writer)

        print(f"Animation saved as {output_filename}")

    # Return the animation object for inline display in Jupyter Notebook
    return ani

###################################################################################################
###################################################################################################
###################################################################################################

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

def animate_strain_maps(Ep1All, Ep2All, Ep3All,
                        FrameDisplX=None, FrameDisplY=None,
                        vmin=-0.15, vmax=0.15,
                        output_filename="strain_animation.mp4",
                        save_file=False):
    """
    Animate principal strains and optionally displacements over time.

    Args:
        Ep1All, Ep2All, Ep3All: 3D arrays (H, W, T) of strains.
        FrameDisplX, FrameDisplY (optional): 3D arrays (H, W, T) of displacements.
        vmin, vmax: Colorbar limits for strain display.
        output_filename: MP4 filename to save the animation.
        save_file: Whether to save the animation to disk.

    Returns:
        ani: Animation object for inline Jupyter display.
    """
    include_displacement = (FrameDisplX is not None) and (FrameDisplY is not None)

    # Set up the figure
    num_plots = 5 if include_displacement else 3
    fig, axes = plt.subplots(1, num_plots, figsize=(5*num_plots, 5))

    if num_plots == 1:
        axes = [axes]  # Ensure it's always iterable

    ims = []
    colorbars = []

    # Titles for each plot
    titles = [
        r"$\varepsilon_1$ (Principal Strain 1)",
        r"$\varepsilon_2$ (Principal Strain 2)",
        r"$\varepsilon_3$ (Incompressibility Strain)"
    ]
    if include_displacement:
        titles += ["X Displacement", "Y Displacement"]

    # Initial plots
    datasets = [Ep1All, Ep2All, Ep3All]
    if include_displacement:
        datasets += [FrameDisplX, FrameDisplY]

    vmin_list = [vmin, vmin, vmin]
    vmax_list = [vmax, vmax, vmax]
    if include_displacement:
        # Dynamically set displacement vmin/vmax based on max of displacements
        disp_vmax = max(np.abs(FrameDisplX).max(), np.abs(FrameDisplY).max())
        vmin_list += [-disp_vmax, -disp_vmax]
        vmax_list += [disp_vmax, disp_vmax]

    for i, ax in enumerate(axes):
        im = ax.imshow(np.zeros_like(Ep1All[:, :, 0]), cmap='coolwarm', origin='lower',
                       vmin=vmin_list[i], vmax=vmax_list[i], animated=True)
        ax.set_title(titles[i])
        ax.axis("off")
        ims.append(im)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        colorbars.append(cbar)

    # Update function for animation
    def update(frame):
        for i, im in enumerate(ims):
            im.set_array(datasets[i][:, :, frame])
        return ims

    # Create animation
    num_frames = Ep1All.shape[2]
    ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=50, blit=False)

    if save_file:
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=20, metadata=dict(artist='Strain Animation'), bitrate=1800)
        ani.save(output_filename, writer=writer)
        print(f"Animation saved as {output_filename}")

    plt.close(fig)  # Prevent double rendering in Jupyter
    return ani



###################################################################################################
###################################################################################################
###################################################################################################

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cv2
from IPython.display import HTML

def animate_deformed_mri(MaskHT0, FrameDisplX, FrameDisplY, output_filename="deformed_mri.mp4", Masks = None,save_file=False):
    """
    Creates an animated visualization of the deformed MRI image over time and saves it as an MP4.

    Args:
        MaskHT0 (np.ndarray): Original MRI image (grayscale, shape: [H, W])
        FrameDisplX (np.ndarray): X-displacement field (shape: [H, W, Frames])
        FrameDisplY (np.ndarray): Y-displacement field (shape: [H, W, Frames])
        output_filename (str): Name of the saved MP4 file.

    Returns:
        ani (FuncAnimation): The animation object for display in Jupyter Notebook.
    """
    num_frames = FrameDisplX.shape[2]  # Number of frames in the animation
    height, width, _ = MaskHT0.shape  # Image dimensions

    # Create a meshgrid for remapping
    x, y = np.meshgrid(np.arange(width), np.arange(height))

    # Set up the figure for animation
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(MaskHT0, cmap="gray", animated=True)
    ax.set_title("Deformed MRI Image Over Time")
    ax.axis("off")

    # Update function for animation
    def update(frame):
        nonlocal FrameDisplX , FrameDisplY
        # Compute displacement (negate to match MATLAB)
        T3DDispX = -FrameDisplX[:, :, frame].astype(np.float64)
        T3DDispY = -FrameDisplY[:, :, frame].astype(np.float64)

        if Masks is not None:
            T3DDispX =  T3DDispX * Masks[frame]
            T3DDispY =  T3DDispY * Masks[frame]
        
        # Compute new coordinates
        x_new = (x + T3DDispX).astype(np.float32)
        y_new = (y + T3DDispY).astype(np.float32)

        # Apply remap to warp the image
        MaskHT_deformed = cv2.remap(MaskHT0, x_new, y_new, interpolation=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT)        
        # Update the animation frame
        im.set_array(MaskHT_deformed)
        return im

    # Create the animation
    
    ani = animation.FuncAnimation(fig, update, frames=range(num_frames), interval=50, blit=False)
    # plt.show()
    if save_file:
        # Save the animation as an MP4 file
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=20, metadata=dict(artist='MRI Deformation'), bitrate=1800)
        ani.save(output_filename, writer=writer)

        print(f"Animation saved as {output_filename}")
        # Save the animation as an MP4 file
    

    # Return the animation object for inline display in Jupyter Notebook
    return ani


def animate_deformed_masked_mri(Image, MaskHT0, FrameDisplX, FrameDisplY, 
                                output_filename="deformed_mri.mp4", save_file=False
                                , save_mode= False, json_mode= False, patinet_file_name = ""):
    """
    Creates an animated visualization of the deformed MRI image over time and 
    returns both the animation and the deformation data arrays.

    Args:
        Image (np.ndarray): Original MRI image (grayscale, shape: [H, W, 1] or [H, W]).
        MaskHT0 (np.ndarray): Mask image (grayscale, shape: [H, W, 1] or [H, W]).
        FrameDisplX (np.ndarray): X-displacement field (H, W, T).
        FrameDisplY (np.ndarray): Y-displacement field (H, W, T).
        output_filename (str): Name of the saved MP4 file.
        save_file (bool): If True, saves the animation to disk.

    Returns:
        ani (FuncAnimation): The animation object for display.
        Image_deformed_all (np.ndarray): Deformed images over frames (H, W, T).
        MaskHT0_deformed_all (np.ndarray): Deformed masks over frames (H, W, T).
        T3DDispX_masked_all (np.ndarray): Masked displacement X field (H, W, T).
        T3DDispY_masked_all (np.ndarray): Masked displacement Y field (H, W, T).
    """


    num_frames = FrameDisplX.shape[2]  # Number of frames in the animation
    height, width = MaskHT0.shape[:2]  # Image dimensions

    # Create a meshgrid for remapping
    x, y = np.meshgrid(np.arange(width), np.arange(height))

    # Prepare to store all frames
    Image_deformed_all = np.zeros((height, width, num_frames), dtype=np.float32)
    MaskHT0_deformed_all = np.zeros((height, width, num_frames), dtype=np.float32)
    T3DDispX_masked_all = np.zeros((height, width, num_frames), dtype=np.float32)
    T3DDispY_masked_all = np.zeros((height, width, num_frames), dtype=np.float32)
    MaskFadedDefrmd_all = np.zeros((height, width, num_frames), dtype=np.float32)

    # Set up the figure for animation
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(Image, cmap="gray", animated=True)
    ax.set_title("Deformed MRI Image Over Time")
    ax.axis("off")

    #MaskHT0_deformed = MaskHT0
    # Update function for animation
    def update(frame):
        nonlocal patinet_file_name
        # Compute displacement (negate to match MATLAB)
        T3DDispX = -FrameDisplX[:, :, frame].astype(np.float64)
        T3DDispY = -FrameDisplY[:, :, frame].astype(np.float64)

        # Normalize the mask
        Mask_norm = MaskHT0.astype(np.float64)
        if Mask_norm.ndim == 3:
            Mask_norm = Mask_norm[..., 0]
        Mask_norm /= 255.0

        #MaskHT0_deformed = MaskHT0_deformed[...,0].astype(np.float64)/255
        #dilated_MaskHT0 = dilate_mask(Mask_norm)
        # Compute new coordinates
        x_new = (x + T3DDispX).astype(np.float32)
        y_new = (y + T3DDispY).astype(np.float32)
        MaskHT0_deformed = cv2.remap(Mask_norm, x_new, y_new, interpolation=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT)        
        dilated_MaskHT0 = dilate_mask_fade(MaskHT0_deformed, decay_distance=20, binary_closing_size=15, grey_closing_size=5)
        #dilated_MaskHT0 = dilate_mask(MaskHT0_deformed)
        #dilated_MaskHT0=dilate_mask_fade_cosine(MaskHT0_deformed)
        #dilated_MaskHT0=dilate_mask_fade_smooth(MaskHT0_deformed)
        

        T3DDispX_masked = T3DDispX * dilated_MaskHT0
        T3DDispY_masked = T3DDispY * dilated_MaskHT0
        
        # Compute new coordinates
        x_new_masked = (x + T3DDispX_masked).astype(np.float32)
        y_new_masked = (y + T3DDispY_masked).astype(np.float32)

        # Apply remap to warp the image
        Image_deformed = cv2.remap(Image, x_new_masked, y_new_masked, interpolation=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT)
        
        # Store the frames
        Image_deformed_all[:, :, int(frame)] = Image_deformed
        MaskHT0_deformed_all[:, :, int(frame)] = MaskHT0_deformed
        T3DDispX_masked_all[:, :, int(frame)] = T3DDispX_masked
        T3DDispY_masked_all[:, :, int(frame)] = T3DDispY_masked
        MaskFadedDefrmd_all[:, :, int(frame)] = dilated_MaskHT0

        if save_mode:
            frame1 = Image
            frame2 = Image_deformed
            displacementX_save = T3DDispX_masked
            displacementY_save = T3DDispY_masked
            if np.random.rand() > 0.5:
                base_name = os.path.basename(patinet_file_name)
                patinet_file_name = os.path.splitext(base_name)[0]
                suffix = f"{patinet_file_name}_#{frame}"

                # Construct paths
                frame1_path = f"generatedData/Frames/{suffix}_1"
                frame2_path = f"generatedData/Frames/{suffix}_2"
                disp_x_path = f"generatedData/Displacements/{suffix}_x"
                disp_y_path = f"generatedData/Displacements/{suffix}_y"

                ext = ".json" if json_mode else ".npy"
                frame1_file = frame1_path + ext
                frame2_file = frame2_path + ext
                disp_x_file = disp_x_path + ext
                disp_y_file = disp_y_path + ext

                paths = [frame1_file, frame2_file, disp_x_file, disp_y_file]

                if save_if_not_exists(paths):
                    os.makedirs("generatedData/Frames", exist_ok=True)
                    os.makedirs("generatedData/Displacements", exist_ok=True)

                    if json_mode:
                        save_json_array(frame1, frame1_file)
                        save_json_array(frame2, frame2_file)
                        save_json_array(displacementX_save, disp_x_file)
                        save_json_array(displacementY_save, disp_y_file)
                    else:
                        np.save(frame1_path, frame1)
                        np.save(frame2_path, frame2)
                        np.save(disp_x_path, displacementX_save)
                        np.save(disp_y_path, displacementY_save)
                else:
                    print(f"Skipped saving: One or more files already exist for {suffix}")


        


        # Update the animation frame
        im.set_array(Image_deformed)
        return im

    # Create the animation
    
    ani = animation.FuncAnimation(fig, update, frames=range(num_frames), interval=50, blit=False)
    # plt.show()
    # Save the animation as an MP4 file
    if save_file:
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=20, metadata=dict(artist='MRI Deformation'), bitrate=1800)
        ani.save(output_filename, writer=writer)

        print(f"Animation saved as {output_filename}")

    # Return the animation object for inline display in Jupyter Notebook
    plt.close(fig)  # prevent double Jupyter display

    return ani, Image_deformed_all, MaskHT0_deformed_all, T3DDispX_masked_all, T3DDispY_masked_all, MaskFadedDefrmd_all
    #return ani



def generate_radial_logistic_label_from_shifted_grids(xMat_shft, yMat_shft, steepness=10, cutoff=0.3):
    """
    Generate a 2D high-pass logistic label image using pre-shifted coordinate grids.

    Args:
        xMat_shft (np.ndarray): Shifted X grid (i.e., x - x0), shape (H, W)
        yMat_shft (np.ndarray): Shifted Y grid (i.e., y - y0), shape (H, W)
        steepness (float): Controls transition sharpness in the logistic function.
        cutoff (float): Normalized radius where value transitions (value ≈ 0.5).

    Returns:
        np.ndarray: Radial high-pass logistic label map with values in [0, 1].
    """
    R = np.sqrt(xMat_shft**2 + yMat_shft**2)
    R_norm = R / np.max(R)  # Normalize to [0, 1]

    label = 1 / (1 + np.exp(-steepness * (R_norm - cutoff)))
    return label

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

def animate_deformation_cines(
    Image_deformed_all, Mask_deformed_all, 
    T3DDispX_masked_all, T3DDispY_masked_all,
    Ep1All_dilated_mask, Ep2All_dilated_mask, Ep3All_dilated_mask, MaskFadedDefrmd_all,
    strain_peak_neg=-0.4, strain_peak_pos=0.4,
    output_filename="deformation_cines.mp4",
    save_file=True
):
    """
    Animate deformation and strain maps over time.

    Args:
        Image_deformed_all, Mask_deformed_all: Deformed MRI and mask cine (H, W, T).
        T3DDispX_masked_all, T3DDispY_masked_all: Masked displacement fields (H, W, T).
        Ep1All_dilated_mask, Ep2All_dilated_mask, Ep3All_dilated_mask: Strain fields (H, W, T).
        strain_peak_neg, strain_peak_pos: Color limits for strain maps.
        output_filename: Filename for MP4 output.
        save_file: Whether to save the animation to disk.

    Returns:
        ani (FuncAnimation): The animation object for display.
    """

    num_frames = Image_deformed_all.shape[2]

    # Set up the figure: 2 rows, 4 columns
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()  # Make it 1D for easy indexing

    # Titles
    titles = [
        "Deformed MRI Image",
        "Deformed Mask",
        "Masked X Displacement",
        "Masked Y Displacement",
        r"Strain $\varepsilon_1$",
        r"Strain $\varepsilon_2$",
        r"Strain $\varepsilon_3$",
        "Fading Binary Mask"  # Last subplot empty
    ]

    # Cines to animate
    cines = [
        Image_deformed_all,
        Mask_deformed_all,
        T3DDispX_masked_all,
        T3DDispY_masked_all,
        Ep1All_dilated_mask,
        Ep2All_dilated_mask,
        Ep3All_dilated_mask,
        MaskFadedDefrmd_all  # Placeholder for empty plot
    ]

    # Calculate vmin and vmax separately
    vmin_list = [
        np.min(Image_deformed_all),
        np.min(Mask_deformed_all),
        np.min(T3DDispX_masked_all),
        np.min(T3DDispY_masked_all),
        strain_peak_neg,
        strain_peak_neg,
        strain_peak_neg,
        np.min(MaskFadedDefrmd_all)  # Fading mask
    ]
    vmax_list = [
        np.max(Image_deformed_all),
        np.max(Mask_deformed_all),
        np.max(T3DDispX_masked_all),
        np.max(T3DDispY_masked_all),
        strain_peak_pos,
        strain_peak_pos,
        strain_peak_pos,
        np.max(MaskFadedDefrmd_all)  # Fading mask
    ]

    # Initialize images
    ims = []
    colorbars = []

    for i, ax in enumerate(axes):
        if cines[i] is not None:
            cmap = 'gray' if i in [0, 1] else 'coolwarm'
            im = ax.imshow(cines[i][:, :, 0], cmap=cmap, origin='lower',
                           vmin=vmin_list[i], vmax=vmax_list[i], animated=True)
            ax.set_title(titles[i])
            ax.axis("off")
            ims.append(im)
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            colorbars.append(cbar)
        else:
            ax.axis("off")  # Hide the last empty subplot

    # Update function for animation
    def update(frame):
        idx = 0
        for i in range(len(cines)):
            if cines[i] is not None:
                ims[idx].set_array(cines[i][:, :, frame])
                idx += 1
        return ims

    ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=50, blit=False)

    if save_file:
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=20, metadata=dict(artist='MRI Deformation'), bitrate=1800)
        ani.save(output_filename, writer=writer)
        print(f"Animation saved as {output_filename}")

    plt.close(fig)

    return ani

