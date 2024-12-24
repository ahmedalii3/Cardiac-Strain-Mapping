import numpy as np
import matplotlib.pyplot as plt
import time
import os
from pathlib import Path
np.random.seed(int(time.time()))

from create_displaced_masks import Create_Displacement_Masks
from mask_dilation import Mask_Dilation
from wave_displacement import Apply_Displacement

size = 0
processed_combinations = set()  # To track already processed (patient, slice, frame)

# Create the displacement masks
while size < 600:
    # Generate random patient number from 1 to 100
    patient_number = np.random.randint(1, 101)
    patient_number = str(patient_number).zfill(3)

    # Generate random slice number from 0 to 10
    slice_number = np.random.randint(0, 11)

    # Loop over frame numbers 01 to 30
    for frame_number in range(1, 31):
        combination = (patient_number, slice_number, frame_number)

        # Skip if this combination is already processed
        if combination in processed_combinations:
            continue

        # Check if the file exists
        path = f'/Users/osama/GP-2025-Strain/Data/ACDC/train_numpy/patient{patient_number}/patient{patient_number}_frame{frame_number}_slice_{slice_number}_ACDC.npy'
        if os.path.exists(path):
            print(f"Patient: {patient_number}, Frame: {frame_number}, Slice: {slice_number}")

            # Add the combination to the processed set
            processed_combinations.add(combination)

            # Process the file
            mask_creator = Create_Displacement_Masks(path=path, save_mode=True)
            mask_dilation = Mask_Dilation()
            wave_displacer = Apply_Displacement(path=path, save_mode=True)

            # Create the displacement masks
            mask_creator.plot()
            while not mask_creator.check_status():
                # Wait until the masks are created
                pass

            mask_dilation.import_masks('displaced_images/displaced_images.npz')
            mask_dilation.create_dilated_masks()
            mask_dilation.save_dilated_masks()
            while not mask_dilation.check_status():
                # Wait until the masks are created
                pass

            wave_displacer.plot()
            while not wave_displacer.check_status():
                # Wait until the masks are created
                pass

            # Count files in the Saved/Displacements directory
            folder_path = Path("/Users/osama/GP-2025-Strain/Code/Wave/Saved/Displacements")
            file_count = len([f for f in folder_path.glob('*') if f.is_file() and f.name != ".DS_Store"])
            size = file_count
            print(f"Size: {size}")
            
            break  # Move to the next patient/slice combination

print("Done :D")