import nibabel as nib
import numpy as np
import os

# Base directory of your dataset
base_dir = r'D:\study\graduation project\GP-2025-Strain\Data\ACDC\database\training'

print("Starting processing of ground truth files...\n")

# Iterate through all directories in the base directory and subdirectories
for root, dirs, files in os.walk(base_dir):
    for dir_name in dirs:
        gt_folder_path = os.path.join(root, dir_name)

        # Find the .nii file inside this folder
        found_file = False
        for file in os.listdir(gt_folder_path):
            if (file.endswith('.nii') or file.endswith('.nii.gz')) and '_gt' in file:
                found_file = True
                file_path = os.path.join(gt_folder_path, file)
                print(f"  Found ground truth file: {file_path}")
                
                # Load the NIfTI file
                mask_img = nib.load(file_path)
                mask_data = mask_img.get_fdata()
                
                # Replace intensity 1 (right ventricle) with 0
                mask_data[mask_data == 1] = 0
                print("    Replaced intensity 1 with 0.")

                # Replace intensity 3 (left atrium) with 1
                mask_data[mask_data == 3] = 1
                print("    Replaced intensity 3 with 1.")
                
                # Create a new NIfTI image with the modified data
                new_mask_img = nib.Nifti1Image(mask_data, mask_img.affine, mask_img.header)
                
                # Save the modified mask to a new file
                new_file_path = os.path.join(gt_folder_path, file.replace('.nii.gz', '_modified.nii.gz'))
                nib.save(new_mask_img, new_file_path)
                
                print(f"    Saved modified file: {new_file_path}\n")

        if not found_file:
            print(f"  No .nii file found in {gt_folder_path}\n")

print("Processing complete for all ground truth files.")
