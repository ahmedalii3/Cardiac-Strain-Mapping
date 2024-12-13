import matplotlib.pyplot as plt
import monai
import torch
import numpy as np
import SimpleITK as sitk
import scipy.interpolate as spi
from scipy.ndimage import zoom
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
        self.images_with_center_mass = []
        self.center_of_masses_dictionary = {}
        self.calculate_median(self.time_series)
        self.localize_image()
        

    def get_cropped_image(self, image_number):
        return self.array_of_cropped_images[image_number]
    
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
        self.array_of_cropped_masks = []
        self.array_of_masks = []
        for root, dirs, files in os.walk(self.time_series):
            for file in files:
                if file.endswith(".npy"):
                    im = np.load(os.path.join(root, file))
                    mask = im[1,:,:]
                    im = im[0,:,:]
                    self.array_of_images_in_time_series[file] = im
                    center = self.center_of_masses_dictionary[file]
                    if np.linalg.norm(center - self.median) > 0:
                        center = self.median
                        self.center_of_masses_dictionary[file] = center
                    cropped_image = self.crop_image(im, center)
                    cropped_mask = self.crop_mask(mask, center)
                    self.array_of_cropped_images.append(cropped_image)
                    self.array_of_cropped_masks.append(cropped_mask)
                    self.array_of_masks.append(mask)

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
        
        for root, dirs, files in os.walk(time_series):
            for file in files:
                if file.endswith(".npy"):
                    im = np.load(os.path.join(root, file))
                    # print(im.shape)
                    mask = im[1,:,:]
                    im = im[0,:,:]
                    self.image = im
                    seg = self.predict_seg()
                    
                    # self.plot_image_seg(im, seg)
                    self.array_of_segmenations.append(seg)  
                    center = self.calculate_center_of_mass(seg)
                    center_mass.append(center)
                    self.center_of_masses_dictionary[file] = center
                    self.array_of_images_in_time_series[file] = im

                    # plt.imshow(self.array_of_images_in_time_series[file], cmap='gray')
                    # plt.scatter(center[1], center[0], c='red')
            # plt.scatter(self.median[0], self.median[1], c='blue')
                    # plt.show()
        
        
        self.median = np.median(center_mass, axis=0)
      
        
                     
    def remove_outliers_zscore(self, mask, target_value, threshold = 0.5):
        x_distribution = np.sum(mask == target_value, axis=0)  # Sum target values along rows
        x_mean = np.mean(x_distribution)
        x_std = np.std(x_distribution)
        x_zscores = (x_distribution - x_mean) / x_std
        x_outliers = (x_zscores < threshold)  # Identify outlier columns
        mask[:, x_outliers] = np.where(mask[:, x_outliers] == target_value, 0, mask[:, x_outliers])

        # Y distribution (rows)
        y_distribution = np.sum(mask == target_value, axis=1)  # Sum target values along columns
        y_mean = np.mean(y_distribution)
        y_std = np.std(y_distribution)
        y_zscores = (y_distribution - y_mean) / x_std
        y_outliers = (y_zscores < threshold)  # Identify outlier rows
        mask[y_outliers, :] = np.where(mask[y_outliers, :] == target_value, 0, mask[y_outliers, :])

        return mask
            
    def predict_seg(self):
        # bundle_path = "/Users/ahmed_ali/Downloads/ventricular_short_axis_3label"
        bundle_path = os.path.join(os.path.dirname(__file__), "..")
        model_path = os.path.join(bundle_path, "models", "model.pt")

        parser = monai.bundle.load_bundle_config(bundle_path, "train.json")
        # parser = monai.bundle.load_bundle_config("..", "train.json")
        net = parser.get_parsed_content("network_def")  # "network" loads the network into GPU which we'll avoid for simplicity here
        # net.load_state_dict(torch.load("../models/model.pt"))
        # net.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        net.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=True))
        inputd = torch.from_numpy(self.image / self.image.max()).float()
        # inputd = torch.from_numpy(im).float()
        pred = net(inputd[None, None, :, :])  # adding batch and channel dimensions to inputd
        pred = torch.softmax(pred[0], dim=0)  # activation and remove batch dimension
        self.seg = torch.argmax(pred, dim=0).data.numpy()  # convert to segmentation
        self.process_seg()
        return self.seg

    def process_seg(self):
        self.seg[self.seg == 3] = 0   # remove the mask of value 3
        self.seg = self.remove_outliers_zscore(self.seg, 1)
        self.seg = self.remove_outliers_zscore(self.seg, 2)

    def calculate_center_of_mass(self, seg):
        self.center_mass = center_of_mass(seg)
        # # calculate center of mass
        # y_indices, x_indices = np.where((seg == 2) | (seg == 1))
        
        # x_center = np.mean(x_indices)
        # y_center = np.mean(y_indices)

        # self.center_mass = [y_center, x_center]
        # plt.imshow(seg, cmap='gray')
        # plt.scatter(self.center_mass[0], self.center_mass[1], c='red')
        # plt.show()
        return self.center_mass
    
    
    
    def crop_image(self, image, center_mass, size = 128):
        cropx, cropy = size, size   # size of the cropped image
        y, x = center_mass
        startx = int(x - cropx // 2)
        starty = int(y - cropy // 2)
        return image[starty:starty + cropy, startx:startx + cropx]

    def crop_mask(self, mask, center_mass, size = 128):
        cropx, cropy = size, size
        y, x = center_mass
        startx = int(x - cropx // 2)
        starty = int(y - cropy // 2)
        return mask[starty:starty + cropy, startx:startx + cropx]

    
    def plot_image_seg(self,image, seg = None):
        plt.imshow(image, cmap='gray')
        plt.imshow(seg, cmap='jet',alpha=0.5)
        plt.show()

    def plot_image(self , image):
        plt.imshow(image, cmap='gray')
        plt.show()  



    

# sample usage

# im = np.load("/Users/ahmed_ali/Documents/GitHub/GP-2025-Strain/Data/ACDC/database/train_numpy/patient085/patient085_frame01_slice_10_ACDC.npy")
# mask = im[1,:,:]
# im = im[0,:,:]
# loop over all patients
directory = "/Users/ahmed_ali/Documents/GitHub/GP-2025-Strain/Data/ACDC/database/train_numpy"
for root, dirs, files in os.walk(directory):
    list_of_directories = dirs
    break
for patient in list_of_directories:
    print(patient)
    Localizer = Localize(os.path.join(directory, patient))
    Localizer.compare_masks_with_cropped_masks()

# Localizer = Localize("/Users/ahmed_ali/Documents/GitHub/GP-2025-Strain/Data/ACDC/database/train_numpy/patient089")  # path to the time series folder
# image = Localizer.get_cropped_image(4)
# # Localizer.compare_masks_with_cropped_masks()
# Localizer.plot_image(image)
# Localizer.view_center_of_mass(2)


        
    

    
            