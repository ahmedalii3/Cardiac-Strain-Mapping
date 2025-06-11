import numpy as np

# Initialize variables (assuming deltaX, deltaY, CineNoFrames, StrainEpPeak are predefined)
deltaT = 1
OverStrain = 1

FrameDisplX = np.load("/Users/osama/GP-2025-Strain/Data/ACDC/Simulated_data_localized/Displacements_Loc/patient002_frame12_slice_1_ACDC_#1_x.npy")
FrameDisplY = np.load("/Users/osama/GP-2025-Strain/Data/ACDC/Simulated_data_localized/Displacements_Loc/patient002_frame12_slice_1_ACDC_#1_y.npy")  # Example dimensions

deltaX = 1.0
deltaY = 1.0
strain_ep_peak = 0.25
CineNoFrames = 1

while OverStrain == 1:
    # Copy current displacements to avoid modifying during iteration
    UX = FrameDisplX.copy()
    UY = FrameDisplY.copy()
    
    # Compute gradients for UX and UY
    UX_grad = np.gradient(UX, deltaX, deltaY, deltaT)
    UXx, UXy, ExtAll = UX_grad[0], UX_grad[1], UX_grad[2]
    
    UY_grad = np.gradient(UY, deltaX, deltaY, deltaT)
    UYx, UYy, EytAll = UY_grad[0], UY_grad[1], UY_grad[2]
    
    # Calculate Eulerian strain components
    ExxAll = (2 * UXx - (UXx**2 + UYx**2)) / 2
    ExyAll = (UXy + UYx - (UXx * UXy + UYx * UYy)) / 2
    EyxAll = ExyAll
    EyyAll = (2 * UYy - (UXy**2 + UYy**2)) / 2
    
    # Compute principal strains
    denominator = ExxAll - EyyAll
    denominator[denominator == 0] = 1e-9  # Prevent division by zero
    ThetaEp = 0.5 * np.arctan(2 * ExyAll / denominator)
    
    sqrt_term = np.sqrt(((ExxAll - EyyAll)/2)**2 + ExyAll**2)
    Ep1All = (ExxAll + EyyAll)/2 + sqrt_term
    Ep2All = (ExxAll + EyyAll)/2 - sqrt_term
    
    # Adjust principal strains and compute Ep3All
    Ep3_temp = Ep1All.copy()
    Ep1All = np.maximum(Ep1All, Ep2All)
    Ep2All = np.minimum(Ep2All, Ep3_temp)
    Ep3All = 1.0 / ((1 + Ep1All) * (1 + Ep2All)) - 1
    
    # Check each frame for exceeding strain threshold
    OverStrain = 0
    for iframe in range(CineNoFrames):
        max_ep1 = np.abs(Ep1All[:, :, iframe]).max()
        max_ep2 = np.abs(Ep2All[:, :, iframe]).max()
        max_ep3 = np.abs(Ep3All[:, :, iframe]).max()
        MaxEp = max(max_ep1, max_ep2, max_ep3)
        
        if MaxEp > StrainEpPeak:
            scale_factor = max(0.95, StrainEpPeak / MaxEp)
            FrameDisplX[:, :, iframe] *= scale_factor
            FrameDisplY[:, :, iframe] *= scale_factor
            OverStrain = 1

# Calculate max and min displacements
UX = FrameDisplX
UY = FrameDisplY
MaxDispl = max(UX.max(), UY.max())
MinDispl = min(UX.min(), UY.min())