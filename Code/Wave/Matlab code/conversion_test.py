import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
def calculate_displacement(FrameDisplX, FrameDisplY, deltaX, deltaY, StrainEpPeak, CineNoFrames):
    OverStrain = True

    # Calculate initial strain maps
    UX_initial = FrameDisplX.copy()
    UY_initial = FrameDisplY.copy()
    UX_initial = gaussian_filter(UX_initial, sigma=2)
    UY_initial = gaussian_filter(UY_initial, sigma=2)
    UXx_initial, UXy_initial = np.gradient(UX_initial, deltaX, deltaY)
    UYx_initial, UYy_initial = np.gradient(UY_initial, deltaX, deltaY)

    ExxAll_initial = (2 * UXx_initial - (UXx_initial**2 + UYx_initial**2)) / 2
    ExyAll_initial = (UXy_initial + UYx_initial - (UXx_initial * UXy_initial + UYx_initial * UYy_initial)) / 2
    EyxAll_initial = ExyAll_initial
    EyyAll_initial = (2 * UYy_initial - (UXy_initial**2 + UYy_initial**2)) / 2

    Ep1All_initial = (ExxAll_initial + EyyAll_initial) / 2 + np.sqrt(((ExxAll_initial - EyyAll_initial) / 2)**2 + ((ExyAll_initial + EyxAll_initial) / 2)**2)
    Ep2All_initial = (ExxAll_initial + EyyAll_initial) / 2 - np.sqrt(((ExxAll_initial - EyyAll_initial) / 2)**2 + ((ExyAll_initial + EyxAll_initial) / 2)**2)
    Ep3All_initial = 1 / ((1 + Ep1All_initial) * (1 + Ep2All_initial)) - 1

    while OverStrain:
        UX = FrameDisplX.copy()
        UY = FrameDisplY.copy()
        UX = gaussian_filter(UX, sigma=2)
        UY = gaussian_filter(UY, sigma=2)
        # Calculate gradients
        UXx, UXy = np.gradient(UX, deltaX, deltaY)
        UYx, UYy = np.gradient(UY, deltaX, deltaY)

        # Eulerian Strain Calculation
        ExxAll = (2 * UXx - (UXx**2 + UYx**2)) / 2
        ExyAll = (UXy + UYx - (UXx * UXy + UYx * UYy)) / 2
        EyxAll = ExyAll
        EyyAll = (2 * UYy - (UXy**2 + UYy**2)) / 2

        # Inplane principal strains
        ThetaEp = 0.5 * np.arctan2(2 * ExyAll, ExxAll - EyyAll)
        Ep1All = (ExxAll + EyyAll) / 2 + np.sqrt(((ExxAll - EyyAll) / 2)**2 + ((ExyAll + EyxAll) / 2)**2)
        Ep2All = (ExxAll + EyyAll) / 2 - np.sqrt(((ExxAll - EyyAll) / 2)**2 + ((ExyAll + EyxAll) / 2)**2)
        Ep3All = Ep1All.copy()
        Ep1All = np.maximum(Ep1All, Ep2All)
        Ep2All = np.minimum(Ep2All, Ep3All)
        Ep3All = 1 / ((1 + Ep1All) * (1 + Ep2All)) - 1

        # Adjust UX and UY to ensure principal strains do not exceed the threshold
        OverStrain = False
        if UX.ndim == 3:  # If the input is 3D
            for iframe in range(CineNoFrames):
                MaxEp = max(
                    np.max(np.abs(Ep1All[:, :, iframe])),
                    np.max(np.abs(Ep2All[:, :, iframe])),
                    np.max(np.abs(Ep3All[:, :, iframe]))
                )
                if MaxEp > StrainEpPeak:
                    FrameDisplX[:, :, iframe] *= max(0.95, StrainEpPeak / MaxEp)
                    FrameDisplY[:, :, iframe] *= max(0.95, StrainEpPeak / MaxEp)
                    OverStrain = True
        else:  # If the input is 2D
            MaxEp = max(
                np.max(np.abs(Ep1All)),
                np.max(np.abs(Ep2All)),
                np.max(np.abs(Ep3All))
            )
            if MaxEp > StrainEpPeak:
                FrameDisplX *= max(0.95, StrainEpPeak / MaxEp)
                FrameDisplY *= max(0.95, StrainEpPeak / MaxEp)
                OverStrain = True

    MaxDispl = max(np.max(UX), np.max(UY))
    MinDispl = min(np.min(UX), np.min(UY))

    # Return the initial and final strain maps along with displacements
    return MaxDispl, MinDispl, FrameDisplX, FrameDisplY, Ep1All_initial,Ep1All


frame_displ_x = np.load("/Users/osama/GP-2025-Strain/Data/ACDC/Simulated_data_localized/Displacements_Loc/patient002_frame12_slice_1_ACDC_#1_x.npy")
frame_displ_y = np.load("/Users/osama/GP-2025-Strain/Data/ACDC/Simulated_data_localized/Displacements_Loc/patient002_frame12_slice_1_ACDC_#1_y.npy")  # Example dimensions

delta_x = 1.0
delta_y = 1.0
strain_ep_peak = 0.1
cine_no_frames = 1

max_displ, min_displ, new_displ_x, new_displ_y , initial, last= calculate_displacement(
    frame_displ_x, frame_displ_y, delta_x, delta_y, strain_ep_peak, cine_no_frames
)
print(max_displ, min_displ)


# # Create a figure with 2x2 subplots with increased figure size
fig, axes = plt.subplots(2, 2, figsize=(16, 14))
# Increase spacing between subplots
plt.subplots_adjust(hspace=0.5, wspace=0.4)

# Plot initial strain map
im1 = axes[0,0].imshow(initial)
axes[0,0].set_title('Initial Strain', pad=20, fontsize=12)
plt.colorbar(im1, ax=axes[0,0], pad=0.1)

# Plot last strain map
im2 = axes[0,1].imshow(last)
axes[0,1].set_title('Last Strain', pad=20, fontsize=12)
plt.colorbar(im2, ax=axes[0,1], pad=0.1)

# Plot initial histogram
axes[1,0].hist(initial.flatten(), bins=50, color='blue', alpha=0.7)
axes[1,0].set_title('Initial Strain Histogram', pad=20, fontsize=12)
axes[1,0].set_xlabel('Strain Value')
axes[1,0].set_ylabel('Frequency')

# Plot last histogram
axes[1,1].hist(last.flatten(), bins=50, color='red', alpha=0.7)
axes[1,1].set_title('Last Strain Histogram', pad=20, fontsize=12)
axes[1,1].set_xlabel('Strain Value')
axes[1,1].set_ylabel('Frequency')

# Adjust layout with padding
plt.tight_layout(pad=3.0)
plt.show()

