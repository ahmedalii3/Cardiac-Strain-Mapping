import os
import numpy as np
import nibabel as nib
import logging

from utils import get_frame_gt_dict

logging.getLogger().setLevel(logging.INFO) # Allow printing info level logs
os.chdir(os.path.dirname(__file__)) #change working directory to current directory

class Prepocessing:
    def __init__(self):
        self.pixel_size = 512

    def check_size(self, img: nib.Nifti1Image) -> bool:
        logging.info("Checking Image size")
        img = img.get_fdata()
        if img.shape[0] != self.pixel_size or img.shape[1] != self.pixel_size:
            logging.error(f"Image size is not {self.pixel_size}x{self.pixel_size}")
            return False
        logging.info("Image size is correct")
        return True
    

    def convert(self, img: nib.Nifti1Image) -> np.ndarray:
        logging.info("Start converting image")
        img = img.get_fdata()
        img = np.moveaxis(img, -1, 2)
        slices = np.split(img, img.shape[2], axis=2)
        logging.info("Image converted")
        return slices
    
    def stack_mask(self,img,mask):
        logging.info("Start stacking mask")
        image_slices = self.convert(img)
        mask_slices = self.convert(mask)
        stacked_images = []
        for i in range(len(image_slices)):
            stacked_images.append(np.dstack((image_slices[i-1],mask_slices[i-1])))
        logging.info("Mask stacked")
        return stacked_images
    
    def save(self,img,path):
        logging.info("Start saving image")
        path_name = f"{path['dataset']}_{path['patient']}_{path['series']}.npy"
        np.save(os.path.join(path['root'], path_name),img)
        logging.info("Image saved")

    def preprocess(self,img: nib.Nifti1Image, mask: nib.Nifti1Image, path) -> None:
        # if self.check_size(img):
        #     stacked = self.stack_mask(img,mask)
        #     self.save(stacked,path)
        # else:
        #     logging.error("Image size is not correct")  
        stacked = self.stack_mask(img,mask)
        for i in range(len(stacked)):
            path['series'] = f"frame{path['frame']}#{i+1}"
            self.save(stacked,path)



# Test the class
preprocessor = Prepocessing()
folder_path = "../../Data/ACDC/training/patient001/"
patient = os.path.basename(folder_path[:-1])
frame_gt_list = get_frame_gt_dict(folder_path)

image = nib.load(str(frame_gt_list[0]['frame']['frame']))
mask = nib.load(str(frame_gt_list[0]['gt']))


frame_info = frame_gt_list[0]['gt'].split('_frame')[1]  # Split and get the part after 'frame'
frame_number = frame_info.split('_')[0]  # Extract the frame number before '_gt' or file extension

stacked = preprocessor.stack_mask(image,mask)
print((stacked[0].shape))
print((stacked[1].shape))
path_info = {
    "dataset": "ACDC",
    "patient": patient,
    "series": None,
    "root" : folder_path,
    "frame": frame_number
}
preprocessor.preprocess(image,mask,path_info)
