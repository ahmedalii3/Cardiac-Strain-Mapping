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


class automate_testing:
    def __init__(self, model, real_test_data_path, model_name):
        self.model = model
        self.model_name = model_name
        self.real_test_data_path = real_test_data_path
        self.model= model
        self.load_data()

    def load_data(self):
        data = {}
        files_real_test = os.listdir(self.real_test_data_path)
        for file in files_real_test:
            if file == "patient_4d_frame_1.npy":
                real_moving_image = np.load(os.path.join(self.real_test_data_path, file))
            elif file == "patient_4d_frame_13.npy":
                real_fixed_image = np.load(os.path.join(self.real_test_data_path, file))
            elif file == "patient_4d_frame_13_mask.npy":
                mask_image = np.load(os.path.join(self.real_test_data_path, file))
        
        data['moving'] = real_moving_image
        data['fixed'] = real_fixed_image

        real_moving_image_expanded = tf.expand_dims(real_moving_image, axis=0)
        real_fixed_image_expanded = tf.expand_dims(real_fixed_image, axis=0)
        
        predicted_deformation_field = self.model.predict([real_moving_image_expanded, real_fixed_image_expanded])
    
        data['displacements'] = predicted_deformation_field[0]

        x_displacement_predicted = predicted_deformation_field[0, :, :, 0]  
        y_displacement_predicted = predicted_deformation_field[0, :, :, 1]
        fixed_image_try = real_fixed_image
        moving_image_try = real_moving_image
        original_map = fixed_image_try - moving_image_try
        # absolute value of the original map
        original_map = np.abs(original_map)
        
        moving_image_try = tf.expand_dims(moving_image_try, axis=-1)
        
        warped_image = self.apply_displacement(moving_image_try, x_displacement_predicted, y_displacement_predicted)
        data['warped'] = warped_image
        mask_image = mask_image == 1
        data['mask'] = mask_image
       
        
        self.create_interactive_plots(data, self.model_name, None)

   
    
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
        mask = data['mask']

        # Calculate strain using the displacement fields
        result = self.limit_strain_range(disp[..., 0], disp[..., 1])
        dx, dy, initial_strain_tensor, final_strain_tensor, max_initial_strain, max_strain, min_initial_strain, min_strain = result

        # Create a figure with 3 rows and 3 columns
        fig, axes = plt.subplots(3, 3, figsize=(30, 25), constrained_layout=True)
        fig.suptitle(f"Sample {sample_idx} Analysis", fontsize=28, y=1.02)

        # --- First Row: Core Images ---
        images = [moving, fixed, warped]
        titles = ["Moving Image", "Fixed Image", "Warped Image"]

        Current_Row=0

        for i, (img, title) in enumerate(zip(images, titles)):
            axes[Current_Row, i].imshow(img, cmap='gray')
            axes[Current_Row, i].set_title(title, fontsize=26)
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

        axes[Current_Row+1, 2].imshow(rgb_wrpd_fxd)
        axes[Current_Row+1, 2].set_title("Warped (Red) over Fixed (RGB)", fontsize=20)
        axes[Current_Row+1, 2].axis('off')

        rgb_mvg_fxd = np.stack([
            moving_norm,      # Red channel
            fixed_norm,      # Green channel
            fixed_norm        # Blue channel
        ], axis=-1)

        axes[Current_Row+2, 2].imshow(rgb_mvg_fxd)
        axes[Current_Row+2, 2].set_title("Moving (Red) over Fixed (RGB)", fontsize=26)
        axes[Current_Row+2, 2].axis('off')


        # --- Second Row: Strain Analysis (Heatmaps) ---
        Current_Row=2
        # Auto-adjust color limits for E1 and E2 strains
        strain_min = min(np.min(final_strain_tensor['E1']), np.min(final_strain_tensor['E2']))
        strain_max = max(np.max(final_strain_tensor['E1']), np.max(final_strain_tensor['E2']))
        abs_max = max(abs(strain_min), abs(strain_max))
        vmin, vmax = -abs_max, abs_max  # Symmetric colormap
        vmin, vmax = -0.5, 0.5  # Symmetric colormap

        strain_images = [final_strain_tensor['E1']*mask, final_strain_tensor['E2']*mask]
        strain_titles = ["Final E1 Strain", "Final E2 Strain"]


        for i, (strain_img, title) in enumerate(zip(strain_images, strain_titles)):
            im = axes[Current_Row, i].imshow(strain_img, cmap='jet', vmin=vmin, vmax=vmax)
            axes[Current_Row, i].set_title(title, fontsize=26)
            axes[Current_Row, i].axis('off')
            self.add_colorbar(fig, axes[Current_Row, i], im, label="Strain (unitless)")

        # Warped Difference Image (Use Signed Differences)
        # diff = fixed - warped
        # im6 = axes[Current_Row, 2].imshow(diff, cmap='bwr', vmin=-np.max(np.abs(diff)), vmax=np.max(np.abs(diff)))
        # axes[Current_Row, 2].set_title("Warped Difference", fontsize=16)
        # axes[Current_Row, 2].axis('off')
        # self.add_colorbar(fig, axes[Current_Row, 2], im6, label="Intensity Difference")

        # axes[Current_Row, 3].axis('off')
        # axes[Current_Row, 4].axis('off')



        # --- Third Row: Strain Overlays on Fixed Image ---
        Current_Row=1
        overlay_titles = ["E1 Strain Overlay", "E2 Strain Overlay"]

        for i, (strain_img, title) in enumerate(zip(strain_images, overlay_titles)):
            # Display fixed image in grayscale
            axes[Current_Row, i].imshow(fixed, cmap='gray', alpha=0.95)
            # Overlay strain with semi-transparency
            im_overlay = axes[Current_Row, i].imshow(strain_img, cmap='jet', alpha=0.5, vmin=vmin, vmax=vmax)
            axes[Current_Row, i].set_title(title, fontsize=26)
            axes[Current_Row, i].axis('off')
            self.add_colorbar(fig, axes[Current_Row, i], im_overlay, label="Strain (unitless)")

        # Compute local absolute error
        error_map = np.abs(fixed_norm - warped_norm)

        # im = axes[Current_Row, 3].imshow(error_map, cmap='hot')
        # axes[Current_Row, 3].set_title("F-W Local Registration Error Heatmap", fontsize=16)
        # axes[Current_Row, 3].axis('off')
        # self.add_colorbar(fig, axes[Current_Row, 3], im, label="Absolute Intensity Difference")

        # error_map = np.abs(fixed_norm - moving_norm)
        # im = axes[Current_Row, 4].imshow(error_map, cmap='hot')
        # axes[Current_Row, 4].set_title("F-M Local Registration Error Heatmap", fontsize=16)
        # axes[Current_Row, 4].axis('off')
        # self.add_colorbar(fig, axes[Current_Row, 4], im, label="Absolute Intensity Difference")



        axes[Current_Row, 2].axis('off')

        
        # Save figure
        save_path = os.path.join("/Users/ahmed_ali/Documents/GitHub/GP-2025-Strain/Results_Supervised", f"sample_{self.model_name}_analysis.png")
        # save_path ="/Users/ahmed_ali/Documents/GitHub/GP-2025-Strain/Results_Supervised"
        plt.savefig(save_path, bbox_inches='tight', dpi=300)  # Increased DPI for higher resolution
        # plt.show()

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
        cbar.ax.set_ylabel(label, fontsize=26)
        cbar.ax.tick_params(labelsize=20)


# Example usage:

if __name__ == "__main__":
    # Define the model and paths
    model = tf.keras.models.load_model("/Users/ahmed_ali/Documents/GitHub/GP-2025-Strain/Best_models/Unet_3Dense_5Kernel_with_mask (1).keras", custom_objects={'MaskLoss': MaskLoss, 'MAELoss': MAELoss, 'Unet_2Dense_5Kernel': Unet_2Dense_5Kernel, 'Residual_Unet': Residual_Unet, 'Unet': Unet, 'Unet_7Kernel': Unet_7Kernel, 'Unet_5Kernel': Unet_5Kernel, 'Unet_3Dense': Unet_3Dense, 'Unet_1Dense': Unet_1Dense, 'Unet_2Dense': Unet_2Dense, 'Unet_1Dense_7Kernel': Unet_1Dense_7Kernel, 'Unet_1Dense_5Kernel': Unet_1Dense_5Kernel, 'Unet_2Dense_7Kernel': Unet_2Dense_7Kernel, 'Unet_2Dense_5Kernel': Unet_2Dense_5Kernel, 'Unet_3Dense_7Kernel': Unet_3Dense_7Kernel, 'Unet_3Dense_5Kernel': Unet_3Dense_5Kernel, 'Residual_Unet_1D': Residual_Unet_1D, 'Residual_Unet_2D': Residual_Unet_2D, 'Residual_Unet_3D': Residual_Unet_3D, 'Residual_Unet_1D_7K': Residual_Unet_1D_7K, 'Residual_Unet_1D_5K': Residual_Unet_1D_5K, 'Residual_Unet_2D_7K': Residual_Unet_2D_7K, 'Residual_Uner_2D_5K': Residual_Unet_2D_5K, 'Residual_Unet_3D_7K': Residual_Unet_3D_7K, 'Residual_Unet_3D_5K': Residual_Unet_3D  })

    real_test_data_path = "/Users/ahmed_ali/Documents/GitHub/GP-2025-Strain/Code/FrameWork/real_test_data"

    model_name = "Unet_3Dense_5Kernel_with_mask"  # Name of the model for saving results

    # Create an instance of automate_testing
    tester = automate_testing(model, real_test_data_path, model_name)
    
    # The data will be loaded and processed in the constructor
