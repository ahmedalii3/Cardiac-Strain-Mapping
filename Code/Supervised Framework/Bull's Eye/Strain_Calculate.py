import numpy as np
import tensorflow as tf
import os




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


def limit_strain_range(FrameDisplX, FrameDisplY, deltaX=1, deltaY=1):
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
    final_tensor = {}
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

    final_tensor['E1'] = Ep1All
    final_tensor['E2'] = Ep2All
    final_tensor['E3'] = Ep3All
    

    return None, None, final_tensor, final_tensor, np.max(Ep1All), np.max(Ep2All), np.min(Ep1All), np.min(Ep2All)
