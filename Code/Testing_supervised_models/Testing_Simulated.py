import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
import matplotlib.pyplot as plt
from keras.layers import Input, concatenate, add, Multiply, Lambda
from keras.models import Model
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.models import Model
from scipy.ndimage import gaussian_filter
from pathlib import Path
import os
import cv2
from mpl_toolkits.axes_grid1 import make_axes_locatable
import sys
from mpl_toolkits.axes_grid1 import make_axes_locatable
sys.path.append(os.path.abspath("/Users/ahmed_ali/Documents/GitHub/GP-2025-Strain/Code/FrameWork/Models_Arch"))
sys.path.append(os.path.abspath("/Users/ahmed_ali/Documents/GitHub/GP-2025-Strain/Code/FrameWork"))
from mask_loss import MaskLoss
from mask_loss import MAELoss
# print(str(Path(__file__).parent))
from ResidualUnet import Residual_Unet
from Unet import Unet
from Unet_7Kernel import Unet_7Kernel
from Unet_5Kernel import Unet_5Kernel
from Unet_3Dense import Unet_3Dense
from Unet_1Dense import Unet_1Dense
from Unet_2Dense import Unet_2Dense
from Unet_1Dense_7Kernel import Unet_1Dense_7Kernel
from Unet_1Dense_5Kernel import Unet_1Dense_5Kernel
from Unet_2Dense_7Kernel import Unet_2Dense_7Kernel
from Unet_2Dense_5Kernel import Unet_2Dense_5Kernel
from Unet_3Dense_7Kernel import Unet_3Dense_7Kernel
from Unet_3Dense_5Kernel import Unet_3Dense_5Kernel
from ResidualUnet_1Dense import Residual_Unet_1D
from ResidualUnet_2Dense import Residual_Unet_2D
from ResidualUnet_3Dense import Residual_Unet_3D
from ResidualUnet_1Dense_7Kernels import Residual_Unet_1D_7K
from ResidualUnet_1Dense_5Kernels import Residual_Unet_1D_5K
from ResidualUnet_2Dense_7Kernels import Residual_Unet_2D_7K
from ResidualUner_2Dense_5Kernels import Residual_Unet_2D_5K
from ResidualUnet_3Dense_7Kernels import Residual_Unet_3D_7K
from ResidualUnet_3Dense_5Kernels import Residual_Unet_3D_5K


class test_model :
    def __init__(self, model, test_data_path, test_labels_path, mask_path, model_name):
        self.model = model
        self.test_data_path = test_data_path
        self.test_labels_path = test_labels_path
        self.mask_path = mask_path
        self.model_name = model_name
        self.x_displacement_label = {}
        self.y_displacement_label = {}
        self.frame1 = {}
        self.frame2 = {}
        self.masks = {}
        self.x_displacement_predicted = {}
        self.y_displacement_predicted = {}
        self.E1_predicted = {}
        self.E2_predicted = {}
        self.E1_label = {}
        self.E2_label = {}
        self.E1_loss = {}
        self.E2_loss = {}
        self.E1_std = {}
        self.E2_std = {}
        self.E1_first_quartile = {}
        self.E2_first_quartile = {}
        self.E1_second_quartile = {}
        self.E2_second_quartile = {}
        
        strain_loss_ranges_E1 = [round(x, 2) for x in np.arange(-0.3, 0.35, 0.05).tolist()]
        # print(strain_loss_ranges_E1)
        self.strain_loss_ranges_E1 = strain_loss_ranges_E1

        strain_loss_ranges_E2 = [round(x, 2) for x in np.arange(-0.3, 0.35, 0.05).tolist()]
        self.strain_loss_ranges_E2 = strain_loss_ranges_E2
        # print(self.strain_loss_ranges_E1)
        self.load_test_labels()
        self.load_test_data()
        self.predict_x_and_y()
        self.calculate_strain()
        self.calculate_MSE_E1()
        self.calculate_MSE_E2()
        self.plot_results()
    
    def load_test_labels(self):
        # Load the test data from the specified path
        files = os.listdir(self.test_labels_path)
        for file in files:
            if file.endswith('.npy'):
                file_code = file.split('_')[:-1]
                file_code = "_".join(file_code)
                file_path = os.path.join(self.test_labels_path, file)
                if 'x' in file:
                    self.x_displacement_label[file_code] = np.load(file_path)
                    print(file_path)
                elif 'y' in file:
                    self.y_displacement_label[file_code] = np.load(file_path)

    def load_test_data(self):
        # Load the test data from the specified path
        files = os.listdir(self.test_data_path)
        for file in files:
            if file.endswith('.npy'):
                file_code = file.split('_')[:-1]
                file_code = "_".join(file_code)
                file_path = os.path.join(self.test_data_path, file)
                if '1.npy' in file:
                    print(file_path)
                    self.frame1[file_code] = np.load(file_path)
                 
                  
                elif '2.npy' in file:
                    self.frame2[file_code] = np.load(file_path)
                    mask = np.load(os.path.join(self.mask_path, file))
                    mask[mask == 2] =0
                    self.masks[file_code] = mask


    def predict_x_and_y(self):
        for file_code in self.frame1.keys():
            if file_code in self.frame2.keys():
                frame1 = self.frame1[file_code]
                frame2 = self.frame2[file_code]
                # Call the model to predict the displacement
                frame1 = np.expand_dims(frame1, axis=0)
                frame2 = np.expand_dims(frame2, axis=0)
                disp = self.model.predict([frame1, frame2])
                x_displacement_predicted = disp[..., 0]
                y_displacement_predicted = disp[..., 1]
                x_displacement_predicted = x_displacement_predicted[0]
                y_displacement_predicted = y_displacement_predicted[0]
                self.x_displacement_predicted[file_code] = x_displacement_predicted
                self.y_displacement_predicted[file_code] = y_displacement_predicted
    
    def enforce_full_principal_strain_order(self,Ep1All, Ep2All, Ep3All=None):
        """
        Ensure Ep1All >= Ep2All >= Ep3All at every voxel (pixel) location.
        Sorts the three principal strains per point.

        Args:
            Ep1All (np.ndarray): First principal strain field.
            Ep2All (np.ndarray): Second principal strain field.
            Ep3All (np.ndarray): Third principal strain field (incompressibility strain).

        Returns:
            Ep1_sorted (np.ndarray): Largest principal strain.
            Ep2_sorted (np.ndarray): Middle principal strain.
            Ep3_sorted (np.ndarray): Smallest principal strain.
        """

        if Ep3All is not None:
            # Stack all principal strains along a new axis
            strain_stack = np.stack([Ep1All, Ep2All, Ep3All], axis=0)  # Shape (3, H, W, T)
        else:
            # Stack only the first two principal strains
            strain_stack = np.stack([Ep1All, Ep2All, Ep2All], axis=0) # Shape (2, H, W, T)
        # Sort along the new axis (axis=0) descending
        strain_sorted = np.sort(strain_stack, axis=0)[::-1, ...]  # Reverse to get descending

        Ep1_sorted = strain_sorted[0]
        Ep2_sorted = strain_sorted[1]
        Ep3_sorted = strain_sorted[2]

        return Ep1_sorted, Ep2_sorted, Ep3_sorted


    def limit_strain_range(self,FrameDisplX, FrameDisplY, deltaX=1, deltaY=1):
        """
        Compute principal strains (Ep1, Ep2) and incompressibility strain (Ep3) 
        from displacement fields.

        Args:
            FrameDisplX (np.ndarray): X displacement field (shape: H, W, T).
            FrameDisplY (np.ndarray): Y displacement field (shape: H, W, T).
            deltaX (float): Pixel spacing in the X direction (mm).
            deltaY (float): Pixel spacing in the Y direction (mm).

        Returns:
            Ep1All (np.ndarray): Principal strain 1 (shape: H, W, T).
            Ep2All (np.ndarray): Principal strain 2 (shape: H, W, T).
            Ep3All (np.ndarray): Incompressibility strain (shape: H, W, T).
        """
        final_tensor = {}
        # Compute spatial gradients
        UXx, UXy = np.gradient(FrameDisplX, deltaX, deltaY, axis=(0, 1))
        UYx, UYy = np.gradient(FrameDisplY, deltaX, deltaY, axis=(0, 1))

        # Compute Eulerian strain tensor components
        ExxAll = (2 * UXx - (UXx**2 + UYx**2)) / 2
        ExyAll = (UXy + UYx - (UXx * UXy + UYx * UYy)) / 2
        EyyAll = (2 * UYy - (UXy**2 + UYy**2)) / 2

        # Compute principal strains
        Ep1All = (ExxAll + EyyAll) / 2 + np.sqrt(((ExxAll - EyyAll) / 2) ** 2 + ExyAll ** 2)
        Ep2All = (ExxAll + EyyAll) / 2 - np.sqrt(((ExxAll - EyyAll) / 2) ** 2 + ExyAll ** 2)

        Ep1All, Ep2All, _ = self.enforce_full_principal_strain_order(Ep1All, Ep2All)

        # Compute incompressibility strain using the determinant rule
        Ep3All = 1 / ((1 + np.maximum(Ep1All, Ep2All)) * (1 + np.minimum(Ep1All, Ep2All))) - 1

        final_tensor['E1'] = Ep1All
        final_tensor['E2'] = Ep2All
        final_tensor['E3'] = Ep3All
        

        return None, None, final_tensor, final_tensor, np.max(Ep1All), np.max(Ep2All), np.min(Ep1All), np.min(Ep2All)


    def calculate_strain(self):
        strain_upper_bound = 1
        for file_code in self.frame1.keys():
            if file_code in self.frame2.keys():
                x_displacement = self.x_displacement_predicted[file_code]
                y_displacement = self.y_displacement_predicted[file_code]
                x_displacement_label = self.x_displacement_label[file_code]
                y_displacement_label = self.y_displacement_label[file_code]
             
                # Call the function to limit strain range
                dx, dy, initial_strain_tensor, final_strain_tensor, max_initial_strain, max_strain, min_initial_strain, min_strain = self.limit_strain_range(
                    x_displacement, y_displacement, strain_upper_bound)
                E1_predicted = final_strain_tensor['E1'] * self.masks[file_code]
                E2_predicted = final_strain_tensor['E2'] * self.masks[file_code]
                
                dx_label, dy_label, initial_strain_tensor_label, final_strain_tensor_label, max_initial_strain_label, max_strain_label, min_initial_strain_label, min_strain_label = self.limit_strain_range(
                    x_displacement_label, y_displacement_label, strain_upper_bound)
                E1_label = final_strain_tensor_label['E1'] * self.masks[file_code]
                E2_label = final_strain_tensor_label['E2'] * self.masks[file_code]

           
                # make it a list of lists


                self.E1_predicted[file_code] = E1_predicted
                self.E2_predicted[file_code] = E2_predicted
                self.E1_label[file_code] = E1_label.tolist()
                self.E2_label[file_code] = E2_label.tolist()

    def calculate_MSE_E1(self):
        all_E1_labelled = []
        all_E1_predicted = []

        # Gather all matching file codes from both frames
        for file_code in self.frame1.keys():
            if file_code in self.frame2:
                all_E1_labelled.append(self.E1_label[file_code])
                all_E1_predicted.append(self.E1_predicted[file_code])

        all_E1_labelled = np.array(all_E1_labelled)
        all_E1_predicted = np.array(all_E1_predicted)
        length = all_E1_labelled.shape[0]

        for rang in self.strain_loss_ranges_E1:
            total_error = 0.0
            total_pixels = 0
            all_errors = []

            for i in range(length):
                label = all_E1_labelled[i]
                pred = all_E1_predicted[i]

                # Apply individual masks to label and prediction
                label_mask = (label > rang - 0.025 ) & (label < rang+0.025)
                # pred_mask = (pred > rang ) & (pred < rang+0.05)

                label_masked = np.where(label_mask, label, 0.0)
                pred_masked = np.where(label_mask, pred, 0.0)

                # Only calculate over non-zero (active) pixels
                active_mask = (label_masked != 0) 
                active_pixel_count = np.sum(active_mask)

                if active_pixel_count > 0:
                    # error = (label_masked - pred_masked) ** 2
                    error = np.abs(label_masked - pred_masked)
                    all_errors.extend(error[active_mask].flatten().tolist())
                    # total_error += np.sum(error[active_mask])
                    total_pixels += active_pixel_count
                    # # all_errors.append(error[active_mask].flatten())
                    # error_values = error[active_mask].ravel()  # Flatten the array
                    # all_errors.extend(error_values.tolist())  # Convert to list and extend
                   
               
            # Save normalized MSE for this strain range
            if total_pixels > 0:
                # self.E1_loss[rang] = total_error / total_pixels
                # self.E1_loss[rang] = np.median(error[active_mask])
                self.E1_loss[rang] = np.median(all_errors)

                print(self.E1_loss[rang])
                self.E1_first_quartile[rang] = np.percentile(all_errors, 25)
                self.E1_second_quartile[rang] = np.percentile(all_errors, 75)


            else:
                self.E1_loss[rang] = 0.0  # or np.nan if you want to skip empty ranges
                self.E1_std[rang] = 0.0
                self.E1_first_quartile[rang] = 0.0
                self.E1_second_quartile[rang] = 0.0
            print(f"Range {rang:.2f}: MSE = {self.E1_loss[rang]:.6f}, Active Pixels = {total_pixels}")
            
    def calculate_MSE_E2(self):
        all_E2_labelled = []
        all_E2_predicted = []

        # Collect E2 values for common file codes
        for file_code in self.frame1.keys():
            if file_code in self.frame2:
                E2_label = self.E2_label[file_code]
                E2_predicted = self.E2_predicted[file_code]
                all_E2_labelled.append(E2_label)
                all_E2_predicted.append(E2_predicted)

        all_E2_labelled = np.array(all_E2_labelled)
        all_E2_predicted = np.array(all_E2_predicted)
        length = all_E2_labelled.shape[0]

        for rang in self.strain_loss_ranges_E2:
            total_error = 0.0
            total_pixels = 0
            all_errors = []
            for i in range(length):
                label = all_E2_labelled[i]
                pred = all_E2_predicted[i]

                # Create individual masks
                label_mask = (label > rang - 0.025) & (label < rang+0.025)
                # pred_mask = (pred > rang - 0.05) & (pred < rang)

                # Apply masks
                label_masked = np.where(label_mask, label, 0.0)
                pred_masked = np.where(label_mask, pred, 0.0)

                # Calculate MSE only on non-zero pixels
                active_mask = (label_masked != 0) 
                active_pixel_count = np.sum(active_mask)

                if active_pixel_count > 0:
                    # error = (label_masked - pred_masked) ** 2
                    error = np.abs(label_masked - pred_masked)
                    all_errors.extend(error[active_mask].flatten().tolist())
                    # total_error += np.sum(error[active_mask])
                    total_pixels += active_pixel_count
                    # all_errors.append(error[active_mask].flatten())
                    # error_values = error[active_mask].ravel()  # Flatten the array
                    # all_errors.extend(error_values.tolist())  # Convert to list and extend
                    # print(all_errors)
                    # all_errors = error
                   

            # Normalize by number of active pixels
            if total_pixels > 0:
                # self.E2_loss[rang] = total_error / total_pixels
                self.E2_loss[rang] = np.median(all_errors)
                print(error[active_mask])
                self.E2_first_quartile[rang] = np.percentile(all_errors, 25)
                self.E2_second_quartile[rang] = np.percentile(all_errors, 75)
                
            else:
                self.E2_loss[rang] = 0.0  # Or np.nan if you prefer
                self.E2_std[rang] = 0.0
                self.E2_first_quartile[rang] = 0.0
                self.E2_second_quartile[rang] = 0.0
            print(f"Range {rang:.2f}: E2 MSE = {self.E2_loss[rang]:.6f}, Active Pixels = {total_pixels}")

    def plot_results(self):
        # Save the computed data (E1_loss, E2_loss, etc.)
        np.save(f"E1_loss_{self.model_name}.npy", self.E1_loss)
        np.save(f"E2_loss_{self.model_name}.npy", self.E2_loss)
        np.save(f"E1_first_quartile_{self.model_name}.npy", self.E1_first_quartile)
        np.save(f"E2_first_quartile_{self.model_name}.npy", self.E2_first_quartile)
        np.save(f"E1_third_quartile_{self.model_name}.npy", self.E1_second_quartile)
        np.save(f"E2_third_quartile_{self.model_name}.npy", self.E2_second_quartile)


        E1_loss = np.load(f"E1_loss_{self.model_name}.npy", allow_pickle=True).item()
        E2_loss = np.load(f"E2_loss_{self.model_name}.npy", allow_pickle=True).item()
        E1_first_quartile = np.load(f"E1_first_quartile_{self.model_name}.npy", allow_pickle=True).item()
        E2_first_quartile = np.load(f"E2_first_quartile_{self.model_name}.npy", allow_pickle=True).item()
        E1_second_quartile = np.load(f"E1_third_quartile_{self.model_name}.npy", allow_pickle=True).item()
        E2_second_quartile = np.load(f"E2_third_quartile_{self.model_name}.npy", allow_pickle=True).item()
        # All unique sorted label
        all_labels = sorted([float(label) for label in E1_loss.keys()])

        # Separate E1 and E2 labels
        E1_labels = [label for label in all_labels if label >= 0]
        E2_labels = [label for label in all_labels if label <= 0]

        # E1 data
        E1_loss_list = [E1_loss[label] for label in E1_labels]
        E1_q1_list = [E1_first_quartile[label] for label in E1_labels]
        E1_median_list = [E1_second_quartile[label] for label in E1_labels]

        # E2 data
        E2_loss_list = [E2_loss[label] for label in E2_labels]
        E2_q1_list = [E2_first_quartile[label] for label in E2_labels]
        E2_median_list = [E2_second_quartile[label] for label in E2_labels]

        # Plot setup
        plt.figure(figsize=(12, 6))
        bar_width = 0.05

        # Plot E2 bars
        bars_E2 = plt.bar(E2_labels, E2_loss_list, width=bar_width, color='lightcoral', edgecolor='black', label='E2')
        for i, bar in enumerate(bars_E2):
            center = bar.get_x() + bar.get_width() / 2
            plt.plot([center, center], [E2_q1_list[i], E2_median_list[i]], color='darkred', lw=2)
            plt.scatter(center, E2_q1_list[i], color='red', zorder=3)
            plt.scatter(center, E2_median_list[i], color='green', zorder=3)

        # Plot E1 bars
        bars_E1 = plt.bar(E1_labels, E1_loss_list, width=bar_width, color='skyblue', edgecolor='black', label='E1')
        for i, bar in enumerate(bars_E1):
            center = bar.get_x() + bar.get_width() / 2
            plt.plot([center, center], [E1_q1_list[i], E1_median_list[i]], color='blue', lw=2)
            plt.scatter(center, E1_q1_list[i], color='red', zorder=3)
            plt.scatter(center, E1_median_list[i], color='green', zorder=3)

        # X-axis
        plt.xticks(all_labels, [str(label) for label in all_labels], rotation=45)
        plt.ylabel("Loss")
        plt.title("Loss with Median and Q1 (E1 vs E2) Residual Unet 2D 7K")
        plt.ylim(0, 0.15)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.show()
                            

                    


# model = tf.keras.models.load_model("/Users/ahmed_ali/Library/CloudStorage/GoogleDrive-ahmed.rajab502@eng-st.cu.edu.eg/My Drive/Models/Unet.keras", custom_objects={'MaskLoss': MaskLoss, 'MAELoss': MAELoss, 'Unet': Unet})
model = tf.keras.models.load_model("/Users/ahmed_ali/Documents/GitHub/GP-2025-Strain/Best_models/Unet_3Dense_5Kernel_with_mask (1).keras", custom_objects={'MaskLoss': MaskLoss, 'MAELoss': MAELoss, 'Unet_3Dense_5Kernel': Unet_3Dense_5Kernel})
# model = tf.keras.models.load_model("/Users/ahmed_ali/Downloads/Unet_3Dense_5Kernel_with_mask.keras", custom_objects={ 'Unet_3Dense_5Kernel': Unet_3Dense_5Kernel, 'MaskLoss': MaskLoss, 'MAELoss': MAELoss})
test_model = test_model(model = model, test_data_path = "/Users/ahmed_ali/Documents/GitHub/GP-2025-Strain/Code/Wave_SimulatorV2/generatedData/Frames", test_labels_path = "/Users/ahmed_ali/Documents/GitHub/GP-2025-Strain/Code/Wave_SimulatorV2/generatedData/Displacements",mask_path="/Users/ahmed_ali/Documents/GitHub/GP-2025-Strain/Code/Wave_SimulatorV2/generatedData/npy_masks", model_name = "Unet_3D_5K_w_mask")


                    

    