import matplotlib.pyplot as plt
import monai
import torch
import numpy as np
import SimpleITK as sitk
import scipy.interpolate as spi
from scipy.ndimage import zoom
import sys
import os
from scipy.ndimage import center_of_mass

class Localize:
    def __init__(self, time_series):
        self.image = None
        self.seg = None
        self.center_mass = None
        self.median = None  
        self.array_of_cropped_images = []
        self.array_of_cropped_masks = []
        self.array_of_segmenations = []
        self.array_of_masks = []
        self.array_of_images_in_time_series = {}
        self.time_series = time_series
        self.center_mass_avg = None
        self.images_with_center_mass = []
        self.center_of_masses_dictionary = {}
        self.calculate_median(self.time_series)
        self.localize_image()

    def get_cropped_image(self, image_number):
        return self.array_of_cropped_images[image_number]
    
    def get_cropped_mask(self, image_number):
        return self.array_of_cropped_masks[image_number]
    
    def get_center_mass_avg(self):
        return self.center_mass_avg
    
    def view_center_of_mass(self, image_number):
        i = 0
        for key, value in self.center_of_masses_dictionary.items():
            if i == image_number:
                plt.imshow(self.array_of_images_in_time_series[key], cmap='gray')
                plt.scatter(value[0], value[1], c='red')
                plt.show()
                break
            i += 1

    def localize_image(self):
        self.array_of_cropped_images = []
        for root, dirs, files in os.walk(self.time_series):
            for file in files:
                if file.endswith(".npy"):
                    im = np.load(os.path.join(root, file))
                    im = im[0,:,:]
                    self.array_of_images_in_time_series[file] = im
                    center = self.center_of_masses_dictionary[file]
                    if np.linalg.norm(center - self.median) > 0:
                        center = self.median
                        self.center_of_masses_dictionary[file] = center
                    cropped_image = self.crop_image(im, center)
                    self.array_of_cropped_images.append(cropped_image)

    def compare_masks_with_cropped_masks(self):
        IoU_list = []
        for i in range(len(self.array_of_masks)):
            original_mask = self.array_of_masks[i]   
            cropped_mask = self.array_of_cropped_masks[i]
            number_of_pixels_equal_to_two_in_original_mask = np.sum(original_mask == 2)
            number_of_pixels_equal_to_two_in_cropped_mask = np.sum(cropped_mask == 2)
            if number_of_pixels_equal_to_two_in_original_mask != number_of_pixels_equal_to_two_in_cropped_mask:
                print("Error in cropping the mask")
                print("the difference between them is ", number_of_pixels_equal_to_two_in_original_mask - number_of_pixels_equal_to_two_in_cropped_mask)
                self.plot_image(cropped_mask)
            else:
                print("The cropping is correct")

    def calculate_median(self, time_series):
        center_mass = []
    
        # Step 1: Collect all .npy files and sort them by filename
        npy_files = []
        for root, dirs, files in os.walk(time_series):
            for file in files:
                if file.endswith(".npy"):
                    npy_files.append(file)
        npy_files.sort()  # Sort files to ensure consistent order
        
        if len(npy_files) == 0:
            raise ValueError("No .npy files found in the time series directory")
        
        # Step 2: Process all files for center of mass
        for file in npy_files:
            file_path = os.path.join(time_series, file)
            im = np.load(file_path)
            im = im[0,:,:]
            self.image = im
            seg = self.predict_seg()
            self.array_of_segmenations.append(seg)
            center = self.calculate_center_of_mass(seg)
            center_mass.append(center)
            print(f"The center of mass of {file} is: {center}")
            self.center_of_masses_dictionary[file] = center
            self.array_of_images_in_time_series[file] = im
    
        # Step 3: Compute the median of all centers of mass
        self.median = np.median(center_mass, axis=0)
        self.center_mass_avg = np.mean(center_mass, axis=0)

        # Step 4: Update the center of mass for all other files

    def remove_outliers_zscore(self, mask, target_value, threshold=0.5):
        x_distribution = np.sum(mask == target_value, axis=0)
        x_mean = np.mean(x_distribution)
        x_std = np.std(x_distribution)
        x_zscores = (x_distribution - x_mean) / x_std
        x_outliers = (x_zscores < threshold)
        mask[:, x_outliers] = np.where(mask[:, x_outliers] == target_value, 0, mask[:, x_outliers])

        y_distribution = np.sum(mask == target_value, axis=1)
        y_mean = np.mean(y_distribution)
        y_std = np.std(y_distribution)
        y_zscores = (y_distribution - y_mean) / y_std
        y_outliers = (y_zscores < threshold)
        mask[y_outliers, :] = np.where(mask[y_outliers, :] == target_value, 0, mask[y_outliers, :])

        return mask
            
    def predict_seg(self):
        bundle_path = os.path.join(os.path.dirname(__file__), "..")
        model_path = os.path.join(bundle_path, "models", "model.pt")

        parser = monai.bundle.load_bundle_config(bundle_path, "train.json")
        net = parser.get_parsed_content("network_def")
        net.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=True))
        inputd = torch.from_numpy(self.image / self.image.max()).float()
        pred = net(inputd[None, None, :, :])
        pred = torch.softmax(pred[0], dim=0)
        self.seg = torch.argmax(pred, dim=0).data.numpy()
        self.process_seg()
        return self.seg

    def process_seg(self):
        self.seg[self.seg == 3] = 0
        self.seg = self.remove_outliers_zscore(self.seg, 1)
        self.seg = self.remove_outliers_zscore(self.seg, 2)

    def calculate_center_of_mass(self, seg):
        self.center_mass = center_of_mass(seg)
        return self.center_mass
    
    def crop_image(self, image, center_mass, size=128):
        cropx, cropy = size, size
        y, x = center_mass
        startx = int(x - cropx // 2)
        starty = int(y - cropy // 2)
        return image[starty:starty + cropy, startx:startx + cropx]

    def crop_mask(self, mask, center_mass, size=128):
        cropx, cropy = size, size
        y, x = center_mass
        startx = int(x - cropx // 2)
        starty = int(y - cropy // 2)
        return mask[starty:starty + cropy, startx:startx + cropx]

    def plot_image_seg(self, image, seg=None):
        plt.imshow(image, cmap='gray')
        plt.imshow(seg, cmap='jet', alpha=0.5)
        plt.show()

    def plot_image(self, image):
        plt.imshow(image, cmap='gray')
        plt.show()