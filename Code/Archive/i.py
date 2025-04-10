import numpy as np
import matplotlib.pyplot as plt
image = np.load("/Users/ahmed_ali/Documents/GitHub/GP-2025-Strain/Data/ACDC/database/train_numpy/patient005/patient005_frame01_slice_0_ACDC.npy")
plt.imshow(image[0], cmap='gray')
# plt.axis('off')
plt.show()
