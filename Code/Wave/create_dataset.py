import numpy as np
import matplotlib.pyplot as plt
import time
import os
from pathlib import Path
np.random.seed(int(time.time()))

from create_displaced_masks import Create_Displacement_Masks
from mask_dilation import Mask_Dilation
from wave_displacement import Wave_Displacer
# ================== Configuration ==================
NEW_DATASET = False
DATASET_SIZE = 700

# Get path to current script (create_dataset.py)
current_script = Path(__file__)

# Calculate project root (GP-2025-Strain directory)
project_root = current_script.parent.parent.parent  # Goes up from Code/Wave to main project
data_dir = project_root / "Data" / "ACDC" / "train_numpy"

# Define important directories
saved_displacements = current_script.parent / "Saved" / "Displacements"
saved_frames = current_script.parent / "Saved" / "Frames"

# Create directories if they don't exist
for directory in [saved_displacements, saved_frames]:
    directory.mkdir(parents=True, exist_ok=True)
    
    if(NEW_DATASET):
    # Delete all existing files in directory
        for file in directory.glob('*'):
            if file.is_file() and file.name != ".DS_Store":  # Preserve .DS_Store on macOS
                file.unlink()


# ================== Main Code ==================
size = 0
processed_combinations = set()

while size < DATASET_SIZE:
    patient_number = str(np.random.randint(1, 101)).zfill(3)
    slice_number = np.random.randint(0, 11)

    for frame_number in range(1, 31):
        combination = (patient_number, slice_number, frame_number)
        if combination in processed_combinations:
            continue

        # Construct dynamic path
        patient_folder = data_dir / f"patient{patient_number}"
        npy_file = patient_folder / f"patient{patient_number}_frame{frame_number}_slice_{slice_number}_ACDC.npy"
        
        if npy_file.exists():
            print(f"Patient: {patient_number}, Frame: {frame_number}, Slice: {slice_number}")
            processed_combinations.add(combination)

            # Process the file
            mask_creator = Create_Displacement_Masks(path=str(npy_file), save_mode=True)
            mask_dilation = Mask_Dilation()
            wave_displacer = Wave_Displacer(path=str(npy_file), save_mode=True)

            # Create displacement masks
            mask_creator.plot()
            while not mask_creator.check_status():
                pass

            mask_dilation.import_masks(str(current_script.parent / 'displaced_images' / 'displaced_images.npz'))
            mask_dilation.create_dilated_masks()
            mask_dilation.save_dilated_masks()
            while not mask_dilation.check_status():
                pass

            wave_displacer.plot()
            while not wave_displacer.check_status():
                pass

            # Count files using dynamic path
            file_count = len([f for f in saved_displacements.glob('*') 
                            if f.is_file() and f.name != ".DS_Store"])
            size = file_count
            print(f"Size: {size}")
            
            break

print("Done :D")