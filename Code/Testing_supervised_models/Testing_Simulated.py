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
    def __init__(self, model, test_data_path, test_labels_path, model_name):
        self.model = model
        self.test_data_path = test_data_path
        self.test_labels_path = test_labels_path
        self.model_name = model_name
        self.x_displacement_label = {}
        self.y_displacement_label = {}
        self.frame1 = {}
        self.frame2 = {}
        self.x_displacement_predicted = {}
        self.y_displacement_predicted = {}
        self.E1_predicted = {}
        self.E2_predicted = {}
        self.E1_label = {}
        self.E2_label = {}
        self.E1_loss = {}
        self.E2_loss = {}
        
        strain_loss_ranges_E1 = np.arange(-0.3, 0.3, 0.05).tolist()
        self.strain_loss_ranges_E1 = strain_loss_ranges_E1

        strain_loss_ranges_E2 = np.arange(-0.3, 0.3, 0.05).tolist()
        self.strain_loss_ranges_E2 = strain_loss_ranges_E2
        print(self.strain_loss_ranges_E1)
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
    
    def limit_strain_range(self,displacement_x, displacement_y, strain_upper_bound, stretch = False,
                     reduction_factor=0.99, amplification_factor=1.01, max_iterations=1000, tolerance=1e-6):
        """
        Convert displacement maps to strain tensors using Eulerian strain formulation.
        Iteratively adjust displacements until all strain values are within the specified bounds:
        - Reduce displacements if strain exceeds upper bound
        - Amplify displacements if strain is below lower bound

        Parameters:
        -----------
        displacement_x : numpy.ndarray
            Displacement field in x-direction
        displacement_y : numpy.ndarray
            Displacement field in y-direction
        strain_lower_bound : float
            Minimum desired strain value
        strain_upper_bound : float
            Maximum allowable strain value
        reduction_factor : float, optional
            Factor by which to reduce displacements each iteration (default: 0.99)
        amplification_factor : float, optional
            Factor by which to amplify displacements each iteration (default: 1.01)
        max_iterations : int, optional
            Maximum number of iterations to perform (default: 1000)
        tolerance : float, optional
            Convergence tolerance (default: 1e-6)

        Returns:
        --------
        tuple
            (adjusted_displacement_x, adjusted_displacement_y,
            initial_strain_tensor, final_strain_tensor, max_initial_strain, max_final_strain)
        """
        # Ensure input arrays have the same shape
        if displacement_x.shape != displacement_y.shape:
            raise ValueError("Displacement maps must have the same shape")
        if stretch:
            strain_lower_bound = 0.01
        else:
            strain_lower_bound = 0

        # Make copies of the input arrays to avoid modifying the originals
        dx = displacement_x.copy()
        dy = displacement_y.copy()

        # Create gradient operators for calculating spatial derivatives
        y_size, x_size = dx.shape

        # Calculate initial strain tensor
        # Calculate displacement gradients using central differences
        dudx_initial = np.zeros_like(dx)
        dudy_initial = np.zeros_like(dx)
        dvdx_initial = np.zeros_like(dx)
        dvdy_initial = np.zeros_like(dx)

        # Interior points (central differences)
        dudx_initial[1:-1, 1:-1] = (dx[1:-1, 2:] - dx[1:-1, :-2]) / 2
        dudy_initial[1:-1, 1:-1] = (dx[2:, 1:-1] - dx[:-2, 1:-1]) / 2
        dvdx_initial[1:-1, 1:-1] = (dy[1:-1, 2:] - dy[1:-1, :-2]) / 2
        dvdy_initial[1:-1, 1:-1] = (dy[2:, 1:-1] - dy[:-2, 1:-1]) / 2

        # Edges (forward/backward differences)
        # Left edge
        dudx_initial[:, 0] = dx[:, 1] - dx[:, 0]
        dvdx_initial[:, 0] = dy[:, 1] - dy[:, 0]
        # Right edge
        dudx_initial[:, -1] = dx[:, -1] - dx[:, -2]
        dvdx_initial[:, -1] = dy[:, -1] - dy[:, -2]
        # Top edge
        dudy_initial[0, :] = dx[1, :] - dx[0, :]
        dvdy_initial[0, :] = dy[1, :] - dy[0, :]
        # Bottom edge
        dudy_initial[-1, :] = dx[-1, :] - dx[-2, :]
        dvdy_initial[-1, :] = dy[-1, :] - dy[-2, :]

        # Calculate Eulerian strain tensor components
        # E = 1/2 * (∇u + ∇u^T + ∇u^T∇u)
        E_xx_initial = 0.5 * (2*dudx_initial + dudx_initial**2 + dvdx_initial**2)
        E_yy_initial = 0.5 * (2*dvdy_initial + dudy_initial**2 + dvdy_initial**2)
        E_xy_initial = 0.5 * (dudy_initial + dvdx_initial + dudx_initial*dudy_initial + dvdx_initial*dvdy_initial)
        E_yx_initial = E_xy_initial

        # Calculate principal strains
        avg_normal_strain_initial = (E_xx_initial + E_yy_initial) / 2
        diff_normal_strain_initial = (E_xx_initial - E_yy_initial) / 2
        radius_initial = np.sqrt(diff_normal_strain_initial**2 + E_xy_initial**2)


        E1_initial = avg_normal_strain_initial + radius_initial  # Maximum principal strain
        E2_initial = avg_normal_strain_initial - radius_initial  # Minimum principal strain

        # KHZ 250318: Corrected the calculation of principal strains
        E_xx_initial = 0.5 * (2*dudx_initial - dudx_initial**2 - dvdx_initial**2)
        E_yy_initial = 0.5 * (2*dvdy_initial - dudy_initial**2 - dvdy_initial**2)
        E_xy_initial = 0.5 * (dudy_initial + dvdx_initial - dudx_initial*dudy_initial - dvdx_initial*dvdy_initial)

        E1_initial = (E_xx_initial + E_yy_initial) / 2 + np.sqrt(((E_xx_initial - E_yy_initial) / 2) ** 2 + ((E_xy_initial + E_yx_initial) / 2) ** 2)
        E2_initial = (E_xx_initial + E_yy_initial) / 2 - np.sqrt(((E_xx_initial - E_yy_initial) / 2) ** 2 + ((E_xy_initial + E_yx_initial) / 2) ** 2)
        # KHZ 250318: Corrected the calculation of principal strains


        # Find maximum and minimum absolute strain values
        max_initial_strain = max(np.max(np.abs(E1_initial)), np.max(np.abs(E2_initial)))
        min_initial_strain = min(np.min(np.abs(E1_initial)), np.min(np.abs(E2_initial)))

        # Store initial strain tensor
        initial_strain_tensor = {
            'E_xx': E_xx_initial,
            'E_yy': E_yy_initial,
            'E_xy': E_xy_initial,
            'E1': E1_initial,
            'E2': E2_initial,
            'min_abs_strain': min_initial_strain,
            'max_abs_strain': max_initial_strain
        }

        # If initial strain is already within bounds, no need to iterate
        if (max_initial_strain <= strain_upper_bound) and (min_initial_strain >= strain_lower_bound):
            return dx, dy, initial_strain_tensor, initial_strain_tensor, max_initial_strain, max_initial_strain, min_initial_strain, min_initial_strain

        # Otherwise, proceed with iterative adjustment
        iterations = 0
        max_strain = max_initial_strain
        min_strain = min_initial_strain
        prev_max_strain = float('inf')
        prev_min_strain = 0

        # Initialize strain tensor components for the loop
        E_xx = E_xx_initial.copy()
        E_yy = E_yy_initial.copy()
        E_xy = E_xy_initial.copy()
        E1 = E1_initial.copy()
        E2 = E2_initial.copy()

        while ((max_strain > strain_upper_bound) or (min_strain < strain_lower_bound)) and (iterations < max_iterations):
            # Determine whether to reduce or amplify displacements
            if max_strain > strain_upper_bound:
                # Reduce displacements if above upper bound
                adjustment_factor = reduction_factor
            elif min_strain < strain_lower_bound:
                # Amplify displacements if below lower bound
                adjustment_factor = amplification_factor
            else:
                # This shouldn't happen due to the while condition, but just in case
                break

            # Apply adjustment
            dx *= adjustment_factor
            dy *= adjustment_factor

            # Recalculate displacement gradients
            dudx = np.zeros_like(dx)
            dudy = np.zeros_like(dx)
            dvdx = np.zeros_like(dx)
            dvdy = np.zeros_like(dx)

            # Interior points (central differences)
            dudx[1:-1, 1:-1] = (dx[1:-1, 2:] - dx[1:-1, :-2]) / 2
            dudy[1:-1, 1:-1] = (dx[2:, 1:-1] - dx[:-2, 1:-1]) / 2
            dvdx[1:-1, 1:-1] = (dy[1:-1, 2:] - dy[1:-1, :-2]) / 2
            dvdy[1:-1, 1:-1] = (dy[2:, 1:-1] - dy[:-2, 1:-1]) / 2

            # Edges (forward/backward differences)
            # Left edge
            dudx[:, 0] = dx[:, 1] - dx[:, 0]
            dvdx[:, 0] = dy[:, 1] - dy[:, 0]
            # Right edge
            dudx[:, -1] = dx[:, -1] - dx[:, -2]
            dvdx[:, -1] = dy[:, -1] - dy[:, -2]
            # Top edge
            dudy[0, :] = dx[1, :] - dx[0, :]
            dvdy[0, :] = dy[1, :] - dy[0, :]
            # Bottom edge
            dudy[-1, :] = dx[-1, :] - dx[-2, :]
            dvdy[-1, :] = dy[-1, :] - dy[-2, :]

            # Calculate Eulerian strain tensor components
            # E = 1/2 * (∇u + ∇u^T + ∇u^T∇u)
            E_xx = 0.5 * (2*dudx + dudx**2 + dvdx**2)
            E_yy = 0.5 * (2*dvdy + dudy**2 + dvdy**2)
            E_xy = 0.5 * (dudy + dvdx + dudx*dudy + dvdx*dvdy)

            # Calculate principal strains
            avg_normal_strain = (E_xx + E_yy) / 2
            diff_normal_strain = (E_xx - E_yy) / 2
            radius = np.sqrt(diff_normal_strain**2 + E_xy**2)

            E1 = avg_normal_strain + radius  # Maximum principal strain
            E2 = avg_normal_strain - radius  # Minimum principal strain

            # Find maximum and minimum absolute strain values
            max_strain = max(np.max(np.abs(E1)), np.max(np.abs(E2)))
            min_strain = min(np.min(np.abs(E1)), np.min(np.abs(E2)))

            # Check convergence
            if (abs(max_strain - prev_max_strain) < tolerance and
                abs(min_strain - prev_min_strain) < tolerance):
                break

            prev_max_strain = max_strain
            prev_min_strain = min_strain
            iterations += 1

        # Prepare final strain tensor
        final_strain_tensor = {
            'E_xx': E_xx,
            'E_yy': E_yy,
            'E_xy': E_xy,
            'E1': E1,
            'E2': E2,
            'min_abs_strain': min_strain,
            'max_abs_strain': max_strain
        }

        return dx, dy, initial_strain_tensor, final_strain_tensor, max_initial_strain, max_strain, min_initial_strain, min_strain

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
                E1_predicted = final_strain_tensor['E1']
                E2_predicted = final_strain_tensor['E2']
                
                dx_label, dy_label, initial_strain_tensor_label, final_strain_tensor_label, max_initial_strain_label, max_strain_label, min_initial_strain_label, min_strain_label = self.limit_strain_range(
                    x_displacement_label, y_displacement_label, strain_upper_bound)
                E1_label = final_strain_tensor_label['E1']
                E2_label = final_strain_tensor_label['E2']
                # make it a list of lists


                self.E1_predicted[file_code] = E1_predicted
                self.E2_predicted[file_code] = E2_predicted
                self.E1_label[file_code] = E1_label.tolist()
                self.E2_label[file_code] = E2_label.tolist()

    # def calculate_MSE_E1(self):
    #     all_E1_labelled = []
    #     all_E1_predicted = []

    #     # Collect common samples
    #     for file_code in self.frame1.keys():
    #         if file_code in self.frame2:
    #             all_E1_labelled.append(self.E1_label[file_code])
    #             all_E1_predicted.append(self.E1_predicted[file_code])

    #     all_E1_labelled = np.array(all_E1_labelled)
    #     all_E1_predicted = np.array(all_E1_predicted)
    #     length = all_E1_labelled.shape[0]
        
    #     print(f"Shape of E1 arrays: {all_E1_labelled.shape}")

    #     # Compute MSE for each strain range
    #     for rang in self.strain_loss_ranges:
    #         total_error = 0.0
    #         total_pixels = 0
    #         for i in range(length):
    #             label = all_E1_labelled[i]
    #             pred = all_E1_predicted[i]

    #             # Apply individual masks to label and prediction
    #             label_mask = (label > rang - 0.05) & (label < rang)
    #             pred_mask = (pred > rang - 0.05) & (pred < rang)

    #             label_masked = np.where(label_mask, label, 0.0)
    #             pred_masked = np.where(pred_mask, pred, 0.0)

    #             # Only calculate over non-zero (active) pixels
    #             active_mask = (label_masked != 0) | (pred_masked != 0)
    #             active_pixel_count = np.sum(active_mask)

    #             if active_pixel_count > 0:
    #                 error = (label_masked - pred_masked) ** 2
    #                 total_error += np.sum(error[active_mask])
    #                 total_pixels += active_pixel_count

    #         # Save normalized MSE for this strain range
    #         if total_pixels > 0:
    #             self.E1_loss[rang] = total_error / total_pixels
    #         else:
    #             self.E1_loss[rang] = 0.0  # or np.nan if you want to skip empty ranges

    #     print(f"Range {rang:.2f}: MSE = {self.E1_loss[rang]:.6f}, Active Pixels = {total_pixels}")
    #         # for i in range(length):
    #         #     label = all_E1_labelled[i]
    #         #     pred = all_E1_predicted[i]

    #         #     # Create a mask where BOTH label and pred are in the range
    #         #     mask = (label > rang - 0.05) & (label < rang) & \
    #         #         (pred > rang - 0.05) & (pred < rang)

    #         #     pixel_count = np.sum(mask)
    #         #     if pixel_count > 0:
    #         #         squared_error = (label - pred) ** 2
    #         #         total_squared_error += np.sum(squared_error[mask])
    #         #         total_pixel_count += pixel_count

    #         # if total_pixel_count > 0:
    #         #     mse = total_squared_error / total_pixel_count
    #         # else:
    #         #     mse = 0.0  # or np.nan to indicate missing data

    #         # self.E1_loss[rang] = {
    #         #     'mse': mse,
    #         #     'pixel_count': total_pixel_count
    #         # }

    #         # print(f"Range {rang:.2f}: MSE = {mse:.6f}, Pixels = {total_pixel_count}")
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

            for i in range(length):
                label = all_E1_labelled[i]
                pred = all_E1_predicted[i]

                # Apply individual masks to label and prediction
                label_mask = (label > rang ) & (label < rang+0.05)
                pred_mask = (pred > rang ) & (pred < rang+0.05)

                label_masked = np.where(label_mask, label, 0.0)
                pred_masked = np.where(pred_mask, pred, 0.0)

                # Only calculate over non-zero (active) pixels
                active_mask = (label_masked != 0) | (pred_masked != 0)
                active_pixel_count = np.sum(active_mask)

                if active_pixel_count > 0:
                    error = (label_masked - pred_masked) ** 2
                    total_error += np.sum(error[active_mask])
                    total_pixels += active_pixel_count

            # Save normalized MSE for this strain range
            if total_pixels > 0:
                self.E1_loss[rang] = total_error / total_pixels
            else:
                self.E1_loss[rang] = 0.0  # or np.nan if you want to skip empty ranges

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

            for i in range(length):
                label = all_E2_labelled[i]
                pred = all_E2_predicted[i]

                # Create individual masks
                label_mask = (label > rang - 0.05) & (label < rang)
                pred_mask = (pred > rang - 0.05) & (pred < rang)

                # Apply masks
                label_masked = np.where(label_mask, label, 0.0)
                pred_masked = np.where(pred_mask, pred, 0.0)

                # Calculate MSE only on non-zero pixels
                active_mask = (label_masked != 0) | (pred_masked != 0)
                active_pixel_count = np.sum(active_mask)

                if active_pixel_count > 0:
                    error = (label_masked - pred_masked) ** 2
                    total_error += np.sum(error[active_mask])
                    total_pixels += active_pixel_count

            # Normalize by number of active pixels
            if total_pixels > 0:
                self.E2_loss[rang] = total_error / total_pixels
            else:
                self.E2_loss[rang] = 0.0  # Or np.nan if you prefer

            print(f"Range {rang:.2f}: E2 MSE = {self.E2_loss[rang]:.6f}, Active Pixels = {total_pixels}")

    def plot_results(self):
        # plot the MSE loss as bar plot for each range
        plt.figure(figsize=(10, 5))
        plt.bar(self.E1_loss.keys(), self.E1_loss.values(), width=0.05)
        plt.xlabel('Strain Range')
        plt.ylabel('MSE Loss')
        plt.title('MSE Loss for E1')
        plt.xticks(self.strain_loss_ranges_E1)
        plt.show()
        plt.figure(figsize=(10, 5))
        plt.bar(self.strain_loss_ranges_E2, self.E2_loss.values(), width=0.05)
        plt.xlabel('Strain Range')

        plt.ylabel('MSE Loss')
        plt.title('MSE Loss for E2')
        plt.xticks(self.strain_loss_ranges_E2)
        plt.show()
        

            

                


# model = tf.keras.models.load_model("/Users/ahmed_ali/Library/CloudStorage/GoogleDrive-ahmed.rajab502@eng-st.cu.edu.eg/My Drive/Models/Unet.keras", custom_objects={'MaskLoss': MaskLoss, 'MAELoss': MAELoss, 'Unet': Unet})
model = tf.keras.models.load_model("/Users/ahmed_ali/Downloads/Unet_5Kernel.keras", custom_objects={'Unet_5Kernel': Unet_5Kernel})
test_model = test_model(model = model, test_data_path = "/Users/ahmed_ali/Documents/GitHub/GP-2025-Strain/Code/Wave_SimulatorV2/generatedData/Frames", test_labels_path = "/Users/ahmed_ali/Documents/GitHub/GP-2025-Strain/Code/Wave_SimulatorV2/generatedData/Cropped_Displacements", model_name = "test")



                    

    