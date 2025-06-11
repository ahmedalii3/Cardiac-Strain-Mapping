import numpy as np
import random

from calc_wave import calc_wave
from initialize_wave import initialize_wave

# Initialize the simulation parameters
CNRList = np.arange(10, 19, 1)            # CNR List
SpeckleVarList = np.arange(1, 3.1, 0.1)   # Speckle variance list
SliceThickList = np.arange(6, 9.2, 0.2)   # Slice thickness list
FadingList = np.arange(25, 36) / 1000     # Fading list
CineLength = 40                           # Cine length

StrainEp1Peak = StrainEp2Peak = StrainEp3Peak = 50 / 100

fovX = 280  # Field of view in mm
fovY = 280  # Field of view in mm
TagSpacing = 7  # Tagging spacing in mm
sizeImage = 256  # Size of the image in pixels

# Form the tagging frequency in rad/mm
wx0 = 2 * np.pi / TagSpacing
wy0 = 2 * np.pi / TagSpacing

deltaX = fovX / sizeImage  # mm/pixel
deltaY = fovY / sizeImage  # mm/pixel

# Frame displacement parameters
MaxFrameFracDiplX = 10
MaxFrameDisplX = MaxFrameFracDiplX * deltaX  # in mm
MinFrameDisplX = -MaxFrameFracDiplX * deltaX  # in mm
MaxFrameFracDiplY = 10
MaxFrameDisplY = MaxFrameFracDiplY * deltaY  # in mm
MinFrameDisplY = -MaxFrameFracDiplY * deltaY  # in mm

# Select random parameters from lists
SlcThick = random.choice(SliceThickList)
SpeckleVariance = random.choice(SpeckleVarList)
CNR = random.choice(CNRList)
CineNoFrames = CineLength  # Since it's a fixed value
TissueFading = random.choice(FadingList)

# Strain X-Displacement wave parameters
strainwaveparamX = {
    'meshsize': 256,
    'patchsize': 50,
    'windSpeed': 50,
    'winddir': 20,
    'A': 1e-7,
    'g': 9.81,
    'rng': np.random.RandomState(seed=None)
}

# Here you would call the initialize_wave and calc_wave functions from the earlier script
H0, W, Grid_Sign = initialize_wave(strainwaveparamX)
Z = calc_wave(H0, W, time=0, Grid_Sign=Grid_Sign)
