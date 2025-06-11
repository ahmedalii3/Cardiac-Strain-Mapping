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
sys.path.append(os.path.abspath("Models_Arch"))
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


class Automate_Training():
    def __init__(self, dataset_path, real_test_data_path, models_list, save_dir, saved_model_dir):
        self.dataset_path = dataset_path
        self.real_test_data_path = real_test_data_path
        self.models_list = models_list
        self.save_dir = save_dir
        self.saved_model_dir = saved_model_dir
        self.fixed_images_train = None
        self.fixed_images_test = None
        self.moving_images_train = None
        self.moving_images_test = None
        self.y_train = None
        self.y_test = None
        self.original_flag = True
        self.all_losses = []
        self.all_val_losses = []
        self.model_names = []
        self.load_data()
    
    def load_data(self):
        folders_in_train_simulator_directory = os.listdir(self.dataset_path)
        for i, directory in enumerate(folders_in_train_simulator_directory):
            if directory == "Displacement":
                Displacement_directory = os.path.join(self.dataset_path, directory)
            elif directory == "Frames":
                Frames_directory = os.path.join(self.dataset_path, directory)
     

       
        files_in_Displacement_directory = os.listdir(Displacement_directory)
        files_in_Frames_directory = os.listdir(Frames_directory)
        first_image_in_Frames_directory = os.path.join(Frames_directory, files_in_Frames_directory[0])
        image = np.load(first_image_in_Frames_directory)
        first_image_in_Displacement_directory = os.path.join(Displacement_directory, files_in_Displacement_directory[0])
        image = np.load(first_image_in_Displacement_directory)
        files_in_Displacement_directory.sort()
        files_in_Frames_directory.sort()
        first_frame_dict = {}
        second_frame_dict = {}
        for file in files_in_Frames_directory:
            file_path = os.path.join(Frames_directory, file)
            if file_path.endswith('.DS_Store'):
                continue
            image = np.load(file_path, allow_pickle=True)
            #convert image to float
        
            #normalize image
            # image = image / 255.0
            
            
            frame_id = file.split('_')[-1].split('.')[0]
            
            id = file.split('_')[:-1]
            id = '_'.join(id)
            
            if frame_id == "1" :
                first_frame_dict[id] = image
            else:
                second_frame_dict[id] = image
        x_displacement_dict = {}
        y_displacement_dict = {}
        for file in files_in_Displacement_directory:
            file_path = os.path.join(Displacement_directory , file)
            image = np.load(file_path, allow_pickle=True)

            
            frame_id = file.split('_')[-1].split('.')[0]
            
            id = file.split('_')[:-1]
            id = '_'.join(id)
            
            if frame_id == "x" :
                x_displacement_dict[id] = image
            else:
                y_displacement_dict[id] = image

        fixed_image = []
        moving_image = []
        x_image = []
        y_image = []
        for key in first_frame_dict.keys():
            if key not in second_frame_dict.keys() or key not in x_displacement_dict.keys() or key not in y_displacement_dict.keys():
                continue
            moving_image.append(first_frame_dict[key])
            fixed_image.append(second_frame_dict[key])
            x_image.append(x_displacement_dict[key])
            y_image.append(y_displacement_dict[key])

        displacement_array = np.array([x_image, y_image])
        displacement_array = np.transpose(displacement_array, (1, 2, 3,0))
        fixed_image_array = np.array(fixed_image)
        moving_image_array = np.array(moving_image)
        length_dataset = len(fixed_image_array) 
        train_index = int(length_dataset * 0.90)
        valid_index = int(length_dataset * 0.95)
        

        self.fixed_images_train = fixed_image_array[:train_index]
        self.fixed_images_valid = fixed_image_array[train_index:valid_index]
        self.fixed_images_test = fixed_image_array[valid_index:]
        self.moving_images_train = moving_image_array[:train_index]
        self.moving_images_valid = moving_image_array[train_index:valid_index]
        self.moving_images_test = moving_image_array[valid_index:]

        self.y_train = displacement_array[:train_index]
        self.y_valid = displacement_array[train_index:valid_index]
        self.y_test = displacement_array[valid_index:]


        # save these arrays
 
        

    def apply_displacement( self,image, x_displacement, y_displacement):
        # Prepare meshgrid for remap
        height, width, _ = image.shape
        x, y = np.meshgrid(np.arange(width), np.arange(height))

        # Apply displacement (scale the displacements for more visible effect)
        x_new = (x + x_displacement).astype(np.float32)
        y_new = (y + y_displacement).astype(np.float32)
        # convert image tensor to numpy
        image = image.numpy()
        

        # Warp the image using remap for both x and y displacements
        displaced_image = cv2.remap(image, x_new, y_new, interpolation=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT)
        return displaced_image
    def calculate_strain(self, x_displacement, y_displacement):
                # Example initialization (replace with real data)
        deltaX = 1  # Spatial resolution in X
        deltaY = 1  # Spatial resolution in Y
        
        StrainEpPeak = 0.2  # Strain limit
        CineNoFrames = 1  # Number of frames

        # Simulated displacement field (random example, replace with actual data)
        FrameDisplX = x_displacement.copy()
        FrameDisplY =  y_displacement.copy()
        OverStrain = True

        while OverStrain:
            UX = FrameDisplX.copy()
            UY = FrameDisplY.copy()

            # Compute displacement gradients
            UXx, UXy= np.gradient(UX, deltaX, deltaY, axis=(0, 1))
            UYx, UYy = np.gradient(UY, deltaX, deltaY, axis=(0, 1))

            # Compute Eulerian Strain Tensor
            ExxAll = (2 * UXx - (UXx**2 + UYx**2)) / 2
            ExyAll = (UXy + UYx - (UXx * UXy + UYx * UYy)) / 2
            EyxAll = ExyAll  # Symmetric tensor
            EyyAll = (2 * UYy - (UXy**2 + UYy**2)) / 2

            # Compute principal strains (eigenvalues of the strain tensor)
            ThetaEp = 0.5 * np.arctan2(2 * ExyAll, ExxAll - EyyAll)
            Ep1All = (ExxAll + EyyAll) / 2 + np.sqrt(((ExxAll - EyyAll) / 2) ** 2 + ExyAll**2)
            Ep2All = (ExxAll + EyyAll) / 2 - np.sqrt(((ExxAll - EyyAll) / 2) ** 2 + ExyAll**2)

            # Through-plane principal strain using incompressibility assumption
            Ep3All = 1.0 / ((1 + Ep1All) * (1 + Ep2All)) - 1
            
            # Adjust displacement if strain exceeds threshold
            OverStrain = False
            for iframe in range(CineNoFrames):
                max_ep = max(
                    np.max(np.abs(Ep1All[:, :])),
                    np.max(np.abs(Ep2All[:, :])),
                    np.max(np.abs(Ep3All[:, :]))
                )
                
                if max_ep > StrainEpPeak:
                    FrameDisplX[:, :] *= max(0.95, StrainEpPeak / max_ep)
                    FrameDisplY[:, :] *= max(0.95, StrainEpPeak / max_ep)
                    OverStrain = True

        return Ep1All, Ep2All, Ep3All
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



    def create_interactive_plots(self,data, sample_idx, MODEL_TESTING_PATH):
        """
        Create interactive plots with core images, strain analysis, and strain overlays.

        Parameters:
        -----------
        data : dict
            Dictionary containing:
            - 'moving': Moving images (numpy array).
            - 'fixed': Fixed images (numpy array).
            - 'warped': Warped images (numpy array).
            - 'displacements': Displacement fields (numpy array).
        sample_idx : int, optional
            Index of the sample to plot (default: 0).

        Returns:
        --------
        None
            Displays the plots.
        """
        # Extract data for the selected sample
        moving = data['moving']
        fixed = data['fixed']
        warped = data['warped']
        disp = data['displacements']

        # Calculate strain using the displacement fields
        result = self.limit_strain_range(disp[..., 0], disp[..., 1])
        dx, dy, initial_strain_tensor, final_strain_tensor, max_initial_strain, max_strain, min_initial_strain, min_strain = result

        # Create a figure with 3 rows and 3 columns
        fig, axes = plt.subplots(3, 5, figsize=(40, 21), constrained_layout=True)
        fig.suptitle(f"Sample {sample_idx} Analysis", fontsize=20, y=1.02)

        # --- First Row: Core Images ---
        images = [moving, fixed, warped]
        titles = ["Moving Image", "Fixed Image", "Warped Image"]

        Current_Row=0

        for i, (img, title) in enumerate(zip(images, titles)):
            axes[Current_Row, i].imshow(img, cmap='gray')
            axes[Current_Row, i].set_title(title, fontsize=16)
            axes[Current_Row, i].axis('off')

        # Create RGB image: R and G from warped, B from fixed
        warped_norm = (warped - warped.min()) / (np.ptp(warped))
        fixed_norm = (fixed - fixed.min()) / (np.ptp(fixed))
        moving_norm = (moving - moving.min()) / (np.ptp(moving))

        rgb_wrpd_fxd = np.stack([
            warped_norm,      # Red channel
            fixed_norm,      # Green channel
            fixed_norm        # Blue channel
        ], axis=-1)

        axes[Current_Row, 3].imshow(rgb_wrpd_fxd)
        axes[Current_Row, 3].set_title("Warped (Red) over Fixed (RGB)", fontsize=20)
        axes[Current_Row, 3].axis('off')

        rgb_mvg_fxd = np.stack([
            moving_norm,      # Red channel
            fixed_norm,      # Green channel
            fixed_norm        # Blue channel
        ], axis=-1)

        axes[Current_Row, 4].imshow(rgb_mvg_fxd)
        axes[Current_Row, 4].set_title("Moving (Red) over Fixed (RGB)", fontsize=20)
        axes[Current_Row, 4].axis('off')


        # --- Second Row: Strain Analysis (Heatmaps) ---
        Current_Row=2
        # Auto-adjust color limits for E1 and E2 strains
        strain_min = min(np.min(final_strain_tensor['E1']), np.min(final_strain_tensor['E2']))
        strain_max = max(np.max(final_strain_tensor['E1']), np.max(final_strain_tensor['E2']))
        abs_max = max(abs(strain_min), abs(strain_max))
        vmin, vmax = -abs_max, abs_max  # Symmetric colormap
        vmin, vmax = -0.5, 0.5  # Symmetric colormap

        strain_images = [final_strain_tensor['E1'], final_strain_tensor['E2']]
        strain_titles = ["Final E1 Strain", "Final E2 Strain"]


        for i, (strain_img, title) in enumerate(zip(strain_images, strain_titles)):
            im = axes[Current_Row, i].imshow(strain_img, cmap='jet', vmin=vmin, vmax=vmax)
            axes[Current_Row, i].set_title(title, fontsize=16)
            axes[Current_Row, i].axis('off')
            self.add_colorbar(fig, axes[Current_Row, i], im, label="Strain (unitless)")

        # Warped Difference Image (Use Signed Differences)
        diff = fixed - warped
        im6 = axes[Current_Row, 2].imshow(diff, cmap='bwr', vmin=-np.max(np.abs(diff)), vmax=np.max(np.abs(diff)))
        axes[Current_Row, 2].set_title("Warped Difference", fontsize=16)
        axes[Current_Row, 2].axis('off')
        self.add_colorbar(fig, axes[Current_Row, 2], im6, label="Intensity Difference")

        axes[Current_Row, 3].axis('off')
        axes[Current_Row, 4].axis('off')



        # --- Third Row: Strain Overlays on Fixed Image ---
        Current_Row=1
        overlay_titles = ["E1 Strain Overlay", "E2 Strain Overlay"]

        for i, (strain_img, title) in enumerate(zip(strain_images, overlay_titles)):
            # Display fixed image in grayscale
            axes[Current_Row, i].imshow(fixed, cmap='gray', alpha=0.95)
            # Overlay strain with semi-transparency
            im_overlay = axes[Current_Row, i].imshow(strain_img, cmap='jet', alpha=0.5, vmin=vmin, vmax=vmax)
            axes[Current_Row, i].set_title(title, fontsize=16)
            axes[Current_Row, i].axis('off')
            self.add_colorbar(fig, axes[Current_Row, i], im_overlay, label="Strain (unitless)")

        # Compute local absolute error
        error_map = np.abs(fixed_norm - warped_norm)

        im = axes[Current_Row, 3].imshow(error_map, cmap='hot')
        axes[Current_Row, 3].set_title("F-W Local Registration Error Heatmap", fontsize=16)
        axes[Current_Row, 3].axis('off')
        self.add_colorbar(fig, axes[Current_Row, 3], im, label="Absolute Intensity Difference")

        error_map = np.abs(fixed_norm - moving_norm)
        im = axes[Current_Row, 4].imshow(error_map, cmap='hot')
        axes[Current_Row, 4].set_title("F-M Local Registration Error Heatmap", fontsize=16)
        axes[Current_Row, 4].axis('off')
        self.add_colorbar(fig, axes[Current_Row, 4], im, label="Absolute Intensity Difference")



        axes[Current_Row, 2].axis('off')


        # Save figure
        save_path = os.path.join(MODEL_TESTING_PATH, f"sample_{sample_idx}_analysis.png")
        plt.savefig(save_path, bbox_inches='tight', dpi=300)  # Increased DPI for higher resolution
        plt.close()


    def add_colorbar(self,fig, ax, im, label=""):
        """
        Add a standardized colorbar to a plot axis.

        Parameters:
        -----------
        fig : matplotlib.figure.Figure
            The figure containing the plot.
        ax : matplotlib.axes.Axes
            The axis to which the colorbar will be added.
        im : matplotlib.image.AxesImage
            The image for which the colorbar is created.
        label : str, optional
            Label for the colorbar.

        Returns:
        --------
        None
            Adds a colorbar to the specified axis.
        """
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = fig.colorbar(im, cax=cax)
        cbar.ax.set_ylabel(label, fontsize=12)
        cbar.ax.tick_params(labelsize=10)

    def visualise_outputs(self, model_name, history, model, inc_history = True): 
        #Create folder for the model
        #mkdir
        data = {}
        model_folder = os.path.join(self.save_dir, model_name)
        os.makedirs(model_folder, exist_ok=True)
        if inc_history:
            self.all_losses = (history.history['loss'])
            self.all_val_losses = (history.history['val_loss'])
            np.save(os.path.join(self.save_dir, f"losses{model_name}"), self.all_losses, allow_pickle=True)  
            np.save(os.path.join(self.save_dir, f"val_losses{model_name}"), self.all_val_losses, allow_pickle=True)  
            
            # Plot the training and validation loss
            plt.plot(history.history['loss'], label='Training Loss')
            plt.plot(history.history['val_loss'], label='Validation Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.title('Loss During Training')
            plt.legend()
            loss_plot_path = os.path.join(model_folder, f"loss_plot{model_name}.png")
            plt.savefig(loss_plot_path)
            plt.close()


        test_loss, test_mae = model.evaluate(
            [self.moving_images_test, self.fixed_images_test],  # Test inputs
            self.y_test,                                   # Ground truth displacement fields
            batch_size=32,
            verbose=1
        )
        ############ save the model results ##########################################-------------------------------------------------
        # Save the evaluation results
        results_path = os.path.join(model_folder, f"evaluation_results{model_name}.txt")

        with open(results_path, "w") as f:
            f.write(f"Test Loss: {test_loss:.4f}\n")
            f.write(f"Test MAE: {test_mae:.4f}\n")
        
        #############plot direction real test ##########################################-------------------------------------------------
        files_real_test = os.listdir(self.real_test_data_path)
        for file in files_real_test:
            if file == "patient_4d_frame_1.npy":
                real_moving_image = np.load(os.path.join(self.real_test_data_path, file))
            elif file == "patient_4d_frame_13.npy":
                real_fixed_image = np.load(os.path.join(self.real_test_data_path, file))
            elif file == "patient_053_frame_4_slice3_mask.npy":
                mask_image = np.load(os.path.join(self.real_test_data_path, file))
        # real_moving_image = real_moving_image.astype(np.int16)
        # real_fixed_image = real_fixed_image.astype(np.int16)
        # real_moving_image = cv2.normalize(real_moving_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        # real_fixed_image = cv2.normalize(real_fixed_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
  
        # real_moving_image= np.rot90(real_moving_image, 3)
        # real_moving_image = np.fliplr(real_moving_image)
        # real_fixed_image = np.rot90(real_fixed_image,3)
        # real_fixed_image = np.fliplr(real_fixed_image)
        # mask_image = np.rot90(mask_image,3)
        # mask_image = np.fliplr(mask_image)

        data['moving'] = real_moving_image
        data['fixed'] = real_fixed_image

        real_moving_image_expanded = tf.expand_dims(real_moving_image, axis=0)
        real_fixed_image_expanded = tf.expand_dims(real_fixed_image, axis=0)
        
        predicted_deformation_field = model.predict([real_moving_image_expanded, real_fixed_image_expanded])
    
        data['displacements'] = predicted_deformation_field[0]

        x_displacement_predicted = predicted_deformation_field[0, :, :, 0]  
        y_displacement_predicted = predicted_deformation_field[0, :, :, 1]
        # fig, ax = plt.subplots(1, 3, figsize=(10, 5))
        # ax[0].imshow(real_moving_image, cmap='gray')
        # ax[0].set_title('Moving Image')

        # ax[1].imshow(real_fixed_image, cmap='gray')
        # ax[1].set_title('Fixed Image')

        #         # Prepare the meshgrid for arrows
        # grid_spacing = 5  # adjust for density of arrows
        # h, w = real_moving_image.shape
        # Y, X = np.mgrid[0:h:grid_spacing, 0:w:grid_spacing]

        # # Subsample displacements according to grid spacing
        # U = x_displacement_predicted[::grid_spacing, ::grid_spacing]
        # V = y_displacement_predicted[::grid_spacing, ::grid_spacing]

        # ax[2].imshow(real_moving_image, cmap='gray')

        # # Overlay the displacement vectors (arrows)
        # ax[2].quiver(X, Y, U, V, color='red', angles='xy', scale_units='xy', scale=0.8, width=0.004)

        # ax[2].set_title('Predicted Displacement Field over Moving Image')
        # ax[2].axis('off')

        # # Save the plot
        
        # direction_plot_path = os.path.join(model_folder, f"direction_plot_real_test{model_name}.png")
        # plt.savefig(direction_plot_path)
        # plt.close()

        #################################### plot strain over the real data ########################################################
        # Ep1All, Ep2All, Ep3All = self.calculate_strain(x_displacement_predicted, y_displacement_predicted)
        # vmin, vmax = np.min(Ep2All), np.max(Ep1All)
        # fig, ax = plt.subplots(1, 4, figsize=(15, 5))

        # im1 = ax[0].imshow(Ep1All, cmap='coolwarm', vmin=vmin, vmax=vmax)
        # divider1 = make_axes_locatable(ax[0])
        # cax1 = divider1.append_axes("right", size="5%", pad=0.1)
        # fig.colorbar(im1, cax=cax1, shrink=0.8)
        # ax[0].set_title('Ep1All')

        
        # im2 = ax[1].imshow(Ep2All, cmap='coolwarm', vmin=vmin, vmax=vmax)
        # divider2 = make_axes_locatable(ax[1])
        # cax2 = divider2.append_axes("right", size="5%", pad=0.1)
        # fig.colorbar(im2, cax=cax2, shrink=0.8)
        # ax[1].set_title('Ep2All')

        # im3 = ax[2].imshow(Ep3All, cmap='coolwarm', vmin=vmin, vmax=vmax)
        # divider3 = make_axes_locatable(ax[2])
        # cax3 = divider3.append_axes("right", size="5%", pad=0.1)
        # fig.colorbar(im3, cax=cax3, shrink=0.8)
        # ax[2].set_title('Ep3All')

        # # Prepare the meshgrid for arrows
        # grid_spacing = 5  # adjust for density of arrows
        # h, w = real_moving_image.shape
        # Y, X = np.mgrid[0:h:grid_spacing, 0:w:grid_spacing]

        # # Subsample displacements according to grid spacing
        # U = x_displacement_predicted[::grid_spacing, ::grid_spacing]
        # V = y_displacement_predicted[::grid_spacing, ::grid_spacing]

        # ax[3].imshow(real_moving_image, cmap='gray')

        # # Overlay the displacement vectors (arrows)
        # ax[3].quiver(X, Y, U, V, color='red', angles='xy', scale_units='xy', scale=0.8, width=0.004)

        # ax[3].set_title('Predicted Displacement Field over Moving Image')
        # ax[3].axis('off')


        # # Save the plot

        # for i in range (3):
        #     ax[i].axis('off')
        # strain_plot_path = os.path.join(model_folder, f"strain_plot_real_test{model_name}.png")
        # plt.savefig(strain_plot_path)
        # plt.close()

        ################################################ plot moving and fixed and warped images real test ############################################

        fixed_image_try = real_fixed_image
        moving_image_try = real_moving_image
        original_map = fixed_image_try - moving_image_try
        # absolute value of the original map
        original_map = np.abs(original_map)
        
        moving_image_try = tf.expand_dims(moving_image_try, axis=-1)
        
        warped_image = self.apply_displacement(moving_image_try, x_displacement_predicted, y_displacement_predicted)
        data['warped'] = warped_image
        difference_image = real_fixed_image - warped_image
        #absolute value of the difference image
        difference_image = np.abs(difference_image)

        vmin, vmax = np.min(original_map), np.max(original_map)/3
        
        fig, ax = plt.subplots(1, 5, figsize=(18, 5))  # Increase figure width

        ax[0].imshow(fixed_image_try)
        ax[0].set_title('Fixed Image')

        ax[1].imshow(real_moving_image)
        ax[1].set_title('Moving Image')

        ax[2].imshow(warped_image)
        ax[2].set_title('Warped Image')
        # Use the same color range for both difference images
        
        # Original Difference with colorbar
        im1 = ax[3].imshow(original_map, cmap='hot', vmin=vmin, vmax=vmax)
        
        divider1 = make_axes_locatable(ax[3])
        cax1 = divider1.append_axes("right", size="5%", pad=0.1)
        fig.colorbar(im1, cax=cax1, shrink=0.8)
        ax[3].set_title('Original Difference')

        # Difference Image with the same colorbar scale
        
        im2 =  ax[4].imshow(difference_image, cmap='hot', vmin=vmin, vmax=vmax)
        # ax[4].imshow(fixed_image_try, cmap='gray')
        masked_overlay = np.ma.masked_where(mask_image != 1, mask_image)
        ax[4].imshow(masked_overlay, cmap='gray', alpha=0.1)
        
        divider2 = make_axes_locatable(ax[4])
        cax2 = divider2.append_axes("right", size="5%", pad=0.1)
        fig.colorbar(im2, cax=cax2, shrink=0.8)
        ax[4].set_title('Final Difference')

        for i in range (5):
            ax[i].axis('off')
        # Adjust spacing between subplots
        fig.subplots_adjust(wspace=0.4)

        warped_image_path = os.path.join(model_folder, f"warped_image_real_test{model_name}.png")
        plt.savefig(warped_image_path)
        plt.close()


        self.create_interactive_plots(data, model_name, model_folder)

        ##################################### plot the direction train  ########################################################################
        train_sample = 992
        
        moving_image_try = self.moving_images_train[train_sample]
        fixed_image_try = self.fixed_images_train[train_sample]
        data['moving'] = moving_image_try
        data['fixed'] = fixed_image_try
        moving_image_try = tf.expand_dims(moving_image_try, axis=0)
        fixed_image_try = tf.expand_dims(fixed_image_try, axis=0)
        predicted_deformation_field = model.predict([moving_image_try, fixed_image_try])
        data['displacements'] = predicted_deformation_field[0]
        x_displacement_predicted = predicted_deformation_field[0, :, :, 0]
        y_displacement_predicted = predicted_deformation_field[0, :, :, 1]
        x_displacement = self.y_train[train_sample, :, :, 0]
        y_displacement = self.y_train[train_sample, :, :, 1]
        difference_x = x_displacement - x_displacement_predicted
        difference_y = y_displacement - y_displacement_predicted
        fig, ax = plt.subplots(1, 3, figsize=(10, 5))
        for i in range(0, 128, 10):  # Adjust the step for better visualization
            for j in range(0, 128, 10):
                ax[0].arrow(j, i, x_displacement[i, j], y_displacement[i, j], head_width=0.5, head_length=0.7, fc='b', ec='b')

        for i in range(0, 128, 10):  # Adjust the step for better visualization
            for j in range(0, 128, 10):
                ax[1].arrow(j, i, x_displacement_predicted[i, j], y_displacement_predicted[i, j], head_width=0.5, head_length=0.7, fc='b', ec='b')

        for i in range(0, 128, 10):  # Adjust the step for better visualization
            for j in range(0, 128, 10):
                ax[2].arrow(j, i, difference_x[i, j], difference_y[i, j], head_width=0.5, head_length=0.7, fc='b', ec='b')
        ax[0].set_title('Actual Displacement')
        ax[1].set_title('Predicted Displacement')
        ax[2].set_title('Difference')
        direction_plot_path = os.path.join(model_folder, f"direction_plot_train{model_name}.png")
        plt.savefig(direction_plot_path)
        plt.close()
        
        ######################################## plot magnitude of the displacement train ####################################################################
        magnitude_actual = np.sqrt(x_displacement ** 2 + y_displacement ** 2)
        magnitude_predicted = np.sqrt(x_displacement_predicted ** 2 + y_displacement_predicted ** 2)
        magnitude_difference = np.sqrt(difference_x ** 2 + difference_y ** 2)
        fig, ax = plt.subplots(1, 3, figsize=(10, 5))
        vmin, vmax = np.min(magnitude_actual), np.max(magnitude_actual)

        im1 = ax[0].imshow(magnitude_actual, cmap='hot', vmin=vmin, vmax=vmax)
        divider1 = make_axes_locatable(ax[0])
        cax1 = divider1.append_axes("right", size="5%", pad=0.1)
        fig.colorbar(im1, cax=cax1, shrink=0.8)
        ax[0].set_title('Actual Magnitude')

        im2 = ax[1].imshow(magnitude_predicted, cmap='hot', vmin=vmin, vmax=vmax)
        divider1 = make_axes_locatable(ax[1])
        cax1 = divider1.append_axes("right", size="5%", pad=0.1)
        fig.colorbar(im1, cax=cax1, shrink=0.8)
        ax[1].set_title('Predicted Magnitude')

        im3 = ax[2].imshow(magnitude_difference, cmap='hot', vmin=vmin, vmax=vmax)
        divider1 = make_axes_locatable(ax[2])
        cax1 = divider1.append_axes("right", size="5%", pad=0.1)
        fig.colorbar(im1, cax=cax1, shrink=0.8)
        ax[2].set_title('Difference')

        
        for i in range (3):
            ax[i].axis('off')
        
        magnitude_plot_path = os.path.join(model_folder, f"magnitude_plot_train{model_name}.png")
        plt.savefig(magnitude_plot_path)
        plt.close()


        ####################################### plot the moving and fixed and warped images train #####################################################
        fixed_image_try = self.fixed_images_train[train_sample]
        moving_image_try = self.moving_images_train[train_sample]
        
        moving_image_try = tf.expand_dims(moving_image_try, axis=-1)
        
        warped_image = self.apply_displacement(moving_image_try, x_displacement_predicted, y_displacement_predicted)
        data['warped'] = warped_image
        
        fig, ax = plt.subplots(1, 4, figsize=(18, 5))  # Increase figure width

        ax[0].imshow(fixed_image_try)
        ax[0].set_title('Fixed Image')

        ax[1].imshow(moving_image_try)
        ax[1].set_title('Moving Image')

        ax[2].imshow(warped_image)
        ax[2].set_title('Warped Image')
        # Use the same color range for both difference images
        
       
        grid_spacing = 5  # adjust for density of arrows
        h, w = fixed_image_try.shape
        Y, X = np.mgrid[0:h:grid_spacing, 0:w:grid_spacing]

        # Subsample displacements according to grid spacing
        U = x_displacement_predicted[::grid_spacing, ::grid_spacing]
        V = y_displacement_predicted[::grid_spacing, ::grid_spacing]

        ax[3].imshow(fixed_image_try, cmap='gray')

        # Overlay the displacement vectors (arrows)
        ax[3].quiver(X, Y, U, V, color='red', angles='xy', scale_units='xy', scale=0.8, width=0.004)

        ax[3].set_title('Predicted Displacement Field over Fixed Image')
        

        for i in range (4):
            ax[i].axis('off')
        # Adjust spacing between subplots
        fig.subplots_adjust(wspace=0.4)
        warped_image_path = os.path.join(model_folder, f"warped_image_train{model_name}.png")
        plt.savefig(warped_image_path)
        plt.close()

        model_name_train = model_name + "_train"
        self.create_interactive_plots(data, model_name_train, model_folder)

        ##################################### plot strain over the real data ############################################################
        # fixed_image = self.fixed_images_train[train_sample]
        # moving_image = self.moving_images_train[train_sample]
        # moving_image_try = self.moving_images_train[train_sample]
        # fixed_image_try = self.fixed_images_train[train_sample]
        # moving_image_try = tf.expand_dims(moving_image_try, axis=0)
        # fixed_image_try = tf.expand_dims(fixed_image_try, axis=0)
      
        # predicted_deformation_field = model.predict([moving_image_try, fixed_image_try])
        # x_displacement_predicted = predicted_deformation_field[0, :, :, 0]
        # y_displacement_predicted = predicted_deformation_field[0, :, :, 1]
        
        # Ep1All, Ep2All, Ep3All = self.calculate_strain(x_displacement_predicted, y_displacement_predicted)
        # vmin, vmax = np.min(Ep2All), np.max(Ep1All)
        # fig, ax = plt.subplots(1, 4, figsize=(15, 5))

        # im1 = ax[0].imshow(Ep1All, cmap='coolwarm', vmin=vmin, vmax=vmax)
        # divider1 = make_axes_locatable(ax[0])
        # cax1 = divider1.append_axes("right", size="5%", pad=0.1)
        # fig.colorbar(im1, cax=cax1, shrink=0.8)
        # ax[0].set_title('Ep1All')

        # im2 = ax[1].imshow(Ep2All, cmap='coolwarm', vmin=vmin, vmax=vmax)
        # divider2 = make_axes_locatable(ax[1])
        # cax2 = divider2.append_axes("right", size="5%", pad=0.1)
        # fig.colorbar(im2, cax=cax2, shrink=0.8)
        # ax[1].set_title('Ep2All')


        # im3 = ax[2].imshow(Ep3All, cmap='coolwarm', vmin=vmin, vmax=vmax)
        # divider3 = make_axes_locatable(ax[2])
        # cax3 = divider3.append_axes("right", size="5%", pad=0.1)
        # fig.colorbar(im3, cax=cax3, shrink=0.8)
        # ax[2].set_title('Ep3All')

        

        #          # Prepare the meshgrid for arrows
        # grid_spacing = 5  # adjust for density of arrows
        # h, w = moving_image.shape
        # Y, X = np.mgrid[0:h:grid_spacing, 0:w:grid_spacing]

        # # Subsample displacements according to grid spacing
        # U = x_displacement_predicted[::grid_spacing, ::grid_spacing]
        # V = y_displacement_predicted[::grid_spacing, ::grid_spacing]

        # ax[3].imshow(moving_image, cmap='gray')

        # # Overlay the displacement vectors (arrows)
        # ax[3].quiver(X, Y, U, V, color='red', angles='xy', scale_units='xy', scale=0.8, width=0.004)

        # ax[3].set_title('Predicted Displacement Field over MOving Image')
        # ax[3].axis('off')

        # # Save the plot
        # ax[3].set_title('Predicted Displacement')

        # for i in range (4):
        #     ax[i].axis('off')

        # strain_plot_path = os.path.join(model_folder, f"strain_plot_train_simulated{model_name}.png")
        # plt.savefig(strain_plot_path)
        # plt.close()


         ########################################### plot the direction test  ########################################################################
        test_sample = 67
        
        moving_image_try = self.moving_images_test[test_sample]
        fixed_image_try = self.fixed_images_test[test_sample]
        data['moving'] = moving_image_try
        data['fixed'] = fixed_image_try
        moving_image_try = tf.expand_dims(moving_image_try, axis=0)
        fixed_image_try = tf.expand_dims(fixed_image_try, axis=0)
        predicted_deformation_field = model.predict([moving_image_try, fixed_image_try])
        # data['displacements'] = predicted_deformation_field[0]
        data['displacements'] = self.y_test[test_sample]
        x_displacement_predicted = predicted_deformation_field[0, :, :, 0]
        y_displacement_predicted = predicted_deformation_field[0, :, :, 1]
        x_displacement = self.y_test[test_sample, :, :, 0]
        y_displacement = self.y_test[test_sample, :, :, 1]
        difference_x = x_displacement - x_displacement_predicted
        difference_y = y_displacement - y_displacement_predicted
        fig, ax = plt.subplots(1, 3, figsize=(10, 5))
        for i in range(0, 128, 10):  # Adjust the step for better visualization
            for j in range(0, 128, 10):
                ax[0].arrow(j, i, x_displacement[i, j], y_displacement[i, j], head_width=0.5, head_length=0.7, fc='b', ec='b')

        for i in range(0, 128, 10):  # Adjust the step for better visualization
            for j in range(0, 128, 10):
                ax[1].arrow(j, i, x_displacement_predicted[i, j], y_displacement_predicted[i, j], head_width=0.5, head_length=0.7, fc='b', ec='b')

        for i in range(0, 128, 10):  # Adjust the step for better visualization
            for j in range(0, 128, 10):
                ax[2].arrow(j, i, difference_x[i, j], difference_y[i, j], head_width=0.5, head_length=0.7, fc='b', ec='b')
        ax[0].set_title('Actual Displacement')
        ax[1].set_title('Predicted Displacement')
        ax[2].set_title('Difference')
        direction_plot_path = os.path.join(model_folder, f"direction_plot_test{model_name}.png")
        plt.savefig(direction_plot_path)
        plt.close()
        # self.create_interactive_plots(data, model_name, model_folder)
        
        ############################## plot magnitude of the displacement test ####################################################################
        magnitude_actual = np.sqrt(x_displacement ** 2 + y_displacement ** 2)
        magnitude_predicted = np.sqrt(x_displacement_predicted ** 2 + y_displacement_predicted ** 2)
        magnitude_difference = np.sqrt(difference_x ** 2 + difference_y ** 2)
        fig, ax = plt.subplots(1, 3, figsize=(10, 5))
        vmin, vmax = np.min(magnitude_actual), np.max(magnitude_actual)

        im1 = ax[0].imshow(magnitude_actual, cmap='hot', vmin=vmin, vmax=vmax)
        divider1 = make_axes_locatable(ax[0])
        cax1 = divider1.append_axes("right", size="5%", pad=0.1)
        fig.colorbar(im1, cax=cax1, shrink=0.8)
        ax[0].set_title('Actual Magnitude')

        im2 = ax[1].imshow(magnitude_predicted, cmap='hot', vmin=vmin, vmax=vmax)
        divider1 = make_axes_locatable(ax[1])
        cax1 = divider1.append_axes("right", size="5%", pad=0.1)
        fig.colorbar(im2, cax=cax1, shrink=0.8)
        ax[1].set_title('Predicted Magnitude')

        im3 = ax[2].imshow(magnitude_difference, cmap='hot', vmin=vmin, vmax=vmax)
        divider1 = make_axes_locatable(ax[2])
        cax1 = divider1.append_axes("right", size="5%", pad=0.1)
        fig.colorbar(im3, cax=cax1, shrink=0.8)
        ax[2].set_title('Difference')

        for i in range (3):
            ax[i].axis('off')
        magnitude_plot_path = os.path.join(model_folder, f"magnitude_plot_test{model_name}.png")
        plt.savefig(magnitude_plot_path)
        plt.close()


        #################################### plot the moving and fixed and warped images test ############################################
        fixed_image_try = self.fixed_images_test[test_sample]
        moving_image_try = self.moving_images_test[test_sample]
        
        moving_image_try = tf.expand_dims(moving_image_try, axis=-1)
      
        warped_image = self.apply_displacement(moving_image_try, x_displacement_predicted, y_displacement_predicted)
        data['warped'] = warped_image
        
        fig, ax = plt.subplots(1, 4, figsize=(18, 5))  # Increase figure width

        ax[0].imshow(fixed_image_try)
        ax[0].set_title('Fixed Image')

        ax[1].imshow(moving_image_try)
        ax[1].set_title('Moving Image')

        ax[2].imshow(warped_image)
        ax[2].set_title('Warped Image')
        # Use the same color range for both difference images
        
        grid_spacing = 5  # adjust for density of arrows
        h, w = fixed_image_try.shape
        Y, X = np.mgrid[0:h:grid_spacing, 0:w:grid_spacing]

        # Subsample displacements according to grid spacing
        U = x_displacement_predicted[::grid_spacing, ::grid_spacing]
        V = y_displacement_predicted[::grid_spacing, ::grid_spacing]

        ax[3].imshow(fixed_image_try, cmap='gray')

        # Overlay the displacement vectors (arrows)
        ax[3].quiver(X, Y, U, V, color='red', angles='xy', scale_units='xy', scale=0.8, width=0.004)

        ax[3].set_title('Predicted Displacement Field over fixed Image')
        

        for i in range (4):
            ax[i].axis('off')
        # Adjust spacing between subplots
        fig.subplots_adjust(wspace=0.4)
        warped_image_path = os.path.join(model_folder, f"warped_image_test{model_name}.png")
        plt.savefig(warped_image_path)
        plt.close()
        model_name_test = model_name + "_test"
        self.create_interactive_plots(data, model_name_test, model_folder)


        ############# plot strain over the test data ############
    
        # moving_image = self.moving_images_test[test_sample]
        # moving_image_try = self.moving_images_test[test_sample]
        # fixed_image_try = self.fixed_images_test[test_sample]
        # moving_image_try = tf.expand_dims(moving_image_try, axis=0)
        # fixed_image_try = tf.expand_dims(fixed_image_try, axis=0)

        # predicted_deformation_field = model.predict([moving_image_try, fixed_image_try])
        # x_displacement_predicted = predicted_deformation_field[0, :, :, 0]
        # y_displacement_predicted = predicted_deformation_field[0, :, :, 1]
    
        # Ep1All, Ep2All, Ep3All = self.calculate_strain(x_displacement_predicted, y_displacement_predicted)
        # vmin, vmax = np.min(Ep2All), np.max(Ep1All)
        # fig, ax = plt.subplots(1, 4, figsize=(15, 5))

        # im1 = ax[0].imshow(Ep1All, cmap='coolwarm', vmin=vmin, vmax=vmax)
        # divider1 = make_axes_locatable(ax[0])
        # cax1 = divider1.append_axes("right", size="5%", pad=0.1)
        # fig.colorbar(im1, cax=cax1, shrink=0.8)
        # ax[0].set_title('Ep1All')

        # im2 = ax[1].imshow(Ep2All, cmap='coolwarm', vmin=vmin, vmax=vmax)
        # divider2 = make_axes_locatable(ax[1])
        # cax2 = divider2.append_axes("right", size="5%", pad=0.1)
        # fig.colorbar(im2, cax=cax2, shrink=0.8)
        # ax[1].set_title('Ep2All')

        # im3 = ax[2].imshow(Ep3All, cmap='coolwarm', vmin=vmin, vmax=vmax)
        # divider3 = make_axes_locatable(ax[2])
        # cax3 = divider3.append_axes("right", size="5%", pad=0.1)
        # fig.colorbar(im3, cax=cax3, shrink=0.8)
        # ax[2].set_title('Ep3All')
    
        #          # Prepare the meshgrid for arrows
        # grid_spacing = 5  # adjust for density of arrows
        # h, w = moving_image.shape
        # Y, X = np.mgrid[0:h:grid_spacing, 0:w:grid_spacing]

        # # Subsample displacements according to grid spacing
        # U = x_displacement_predicted[::grid_spacing, ::grid_spacing]
        # V = y_displacement_predicted[::grid_spacing, ::grid_spacing]

        # ax[3].imshow(moving_image, cmap='gray')

        # # Overlay the displacement vectors (arrows)
        # ax[3].quiver(X, Y, U, V, color='red', angles='xy', scale_units='xy', scale=0.8, width=0.004)

        # ax[3].set_title('Predicted Displacement Field over MOving Image')
        # ax[3].axis('off')

        # # Save the plot
        # ax[3].set_title('Predicted Displacement')

        # for i in range (4):
        #     ax[i].axis('off')
        # strain_plot_path = os.path.join(model_folder, f"strain_plot_test_simulated{model_name}.png")
        # plt.savefig(strain_plot_path)
        # plt.close()


        ############# plot x_displacement over train #############
        moving_image   = self.moving_images_test[test_sample]
        x_displacement = self.y_test[test_sample,...,0]
        plt.imshow(moving_image, cmap= 'gray')
        plt.imshow(x_displacement, cmap= 'jet', alpha = 0.5)
        displacement_plot_path = os.path.join(model_folder, f"displacement_plot_test_simulated{model_name}.png")
        plt.savefig(displacement_plot_path)
        plt.close()

    def train_models(self, num_epochs = 10, batch_size = 32):
       
        for model in self.models_list:
            
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, clipvalue=1.0)
            fixed_input = Input(shape=(128, 128, 1), name="fixed_image")
            moving_input = Input(shape=(128, 128, 1), name="moving_image")
            modelnet = model
            model_name = model.__class__.__name__
            self.model_names.append(model_name)
            out_def = modelnet([moving_input, fixed_input])
            model = Model(inputs=[moving_input, fixed_input], outputs=out_def)

            model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

            file_name = f"{model_name}.keras"  
            check_point_path = os.path.join(self.saved_model_dir, file_name)
            # Create an empty .keras file
            open(check_point_path, "w").close()


            # Define callbacks
            checkpoint_callback = ModelCheckpoint(
                check_point_path,
                monitor='val_loss',
                save_best_only=True,
                mode='min',
                verbose=1

            )

            stopping_callback = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=20,
                mode='min',
                verbose=1
            )
            with tf.device('/GPU:0'):
                history = model.fit(
                    [self.moving_images_train, self.fixed_images_train],
                    self.y_train,
                    batch_size=batch_size,
                    epochs = num_epochs,
                    validation_data=([self.moving_images_valid, self.fixed_images_valid], self.y_valid),
                    callbacks=[checkpoint_callback, stopping_callback],
                )
            #Create folder for the model
            self.visualise_outputs(model_name, history, model)
            print(f"Model {model_name} has been trained successfully")


            

            
            
            




if __name__ == '__main__':
    # check for the gpu
    # physical_devices = tf.config.experimental.list_physical_devices('GPU')
    # print("Num GPUs Available: ", len(physical_devices))

    dataset_path = "Data"
    real_test_data_path = "real_test_data"
    current_script = Path(__file__)
    models_list = [ Residual_Unet(),Unet(), Unet_7Kernel(), Unet_5Kernel(), Unet_3Dense(), Unet_1Dense(), Unet_2Dense(), Unet_1Dense_7Kernel(), Unet_1Dense_5Kernel(), Unet_2Dense_7Kernel(), Unet_2Dense_5Kernel(), Unet_3Dense_7Kernel(), Unet_3Dense_5Kernel(), Residual_Unet_1D(), Residual_Unet_2D(), Residual_Unet_3D(), Residual_Unet_1D_7K(), Residual_Unet_1D_5K(), Residual_Unet_2D_7K(), Residual_Unet_2D_5K(), Residual_Unet_3D_7K(), Residual_Unet_3D_5K()]
    # models_list=[Unet()]
    save_dir = current_script.parent / "Saved"
    saved_model_dir = current_script.parent / "Models"
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(saved_model_dir, exist_ok=True)
    trainer = Automate_Training(dataset_path,real_test_data_path, models_list, save_dir, saved_model_dir)
    trainer.train_models(num_epochs = 120, batch_size = 32)
 