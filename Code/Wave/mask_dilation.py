import os
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
os.chdir(os.path.dirname(__file__)) #change working directory to current directory

class Mask_Dilation:
    def __init__(self):
        self.mask = None
        self.all_masks = None
        self.dilated_masks = None
        self.finished = False


    def import_masks(self, path):
        # Load the .npz file
        self.all_masks = np.load(path)
        # Convert to a regular dictionary for mutability
        self.all_masks = {key: self.all_masks[key].astype(np.float64) for key in self.all_masks.keys()}
    def dilate_mask(self, mask):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        value = 0.9
        dilated_mask = mask
        process_mask = mask
        for i in range(10):            
            old_process_mask = process_mask
            process_mask = cv2.dilate(process_mask, kernel)

            # Identify the newly added pixels
            added_region = (process_mask - old_process_mask).astype(np.float64)

            # Update the dilated image
            dilated_mask = dilated_mask + added_region * value
            value -= 0.1
        return dilated_mask
    
    def create_dilated_masks(self):
        self.dilated_masks = {}
        for key in self.all_masks.keys():
            self.dilated_masks[key] = self.dilate_mask(self.all_masks[key])
            self.dilated_masks[key] = gaussian_filter(self.dilated_masks[key], sigma=2)
            print(f"dilating mask number : {key}")
        return self.dilated_masks
    
    def save_dilated_masks(self):
        output_dir = "dilated_masks"
        os.makedirs(output_dir, exist_ok=True)

        np.savez_compressed("dilated_masks/dilated_masks.npz", **self.dilated_masks)
        print(f"Dilated masks saved to {output_dir}")
        self.finished = True

    def check_status(self):
        return self.finished
    
# mask_dilation = Mask_Dilation()
# mask_dilation.import_masks('displaced_images/displaced_images.npz')
# mask_dilation.create_dilated_masks()
# mask_dilation.save_dilated_masks()
        