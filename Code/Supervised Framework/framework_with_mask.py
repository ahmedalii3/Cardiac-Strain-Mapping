import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
import matplotlib.pyplot as plt
from keras.layers import Input, concatenate, add, Multiply, Lambda
from keras.models import Model
import os
from scipy.ndimage import distance_transform_edt, grey_closing, binary_closing, gaussian_filter
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
    """
    A class to automate the training, evaluation, and visualization of image registration models.
    Handles loading of displacement, frame, and mask data, strain calculations, model training,
    and visualization of results for both simulated and real test datasets with weighted masks.
    """
    def __init__(self, dataset_path, real_test_data_path, models_list, save_dir, saved_model_dir):

        """
        Initialize the Automate_Training class with paths and model configurations.

        Args:
            dataset_path (str): Path to dataset containing 'Displacement', 'Frames', and 'Masks' directories.
            real_test_data_path (str): Path to real test data (e.g., patient data).
            models_list (list): List of models to be trained.
            save_dir (str): Directory to save visualization outputs and results.
            saved_model_dir (str): Directory to save trained model checkpoints.
        """
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

    def create_weighted_mask(self,mask, decay_distance=10, binary_closing_size=5, grey_closing_size=5):
        """
        Generate a weighted mask with:
        - Myocardium (label 1) = weight 2.0
        - Smoothly decaying weights outward from the myocardium (controlled by ⁠ dilation_extent ⁠).

        Args:
            mask (np.ndarray): Input mask with labels {0, 1, 2}.
            dilation_extent (int): Number of dilation iterations (higher = wider decay).
            sigma (float): Smoothness of the decay (Gaussian blur).
        
        Returns:
            weighted_mask (np.ndarray): Weighted mask with values ≥1.0.
        """
        # ratio my cardium to back ground
        mask = mask
        ratio = (np.sum(mask == 0) + np.sum(mask == 2)) / (np.sum(mask == 1) )
        if ratio > 20:
            ratio = 20
        # Extract myocardium (label 1)
        mask[mask == 2] = 0

        myocardium = (mask == 1).astype(np.float32)
        
        # Step 1: Ensure binary mask
        binary_mask = (mask > 0).astype(np.uint8)

        # Step 2: Apply binary closing first to smooth the mask
        binary_structure = np.ones((binary_closing_size, binary_closing_size))
        cleaned_binary_mask = binary_closing(binary_mask, structure=binary_structure).astype(np.uint8)

        # Step 3: Compute distance transform outside the cleaned mask
        distance_outside = distance_transform_edt(1 - cleaned_binary_mask)

        # Step 4: Normalize distances to [1, 0] range for fading
        smooth_mask = np.clip(1 - distance_outside / decay_distance, 0, 1)

        # Step 5: Apply grayscale closing only outside the mask
    #    mask_inside = cleaned_binary_mask == 1
    #    mask_outside = cleaned_binary_mask == 0

        smooth_mask_final = grey_closing(smooth_mask, structure=np.ones((grey_closing_size, grey_closing_size)))

        # Step 6: Combine: inside stays 1, outside is smoothed fading
        #smooth_mask_final = smooth_mask.copy()
        #smooth_mask_final[mask_outside] = smooth_mask_outside[mask_outside]
        #smooth_mask_final[mask_inside] = 1  # Enforce inside = 1 exactly
        # Get mask of values strictly between 0 and 1
        mask_between = (smooth_mask_final > 0) 

        # Copy smooth_mask_final
        # fade = smooth_mask_final.copy()

        # Apply logistic function only where values are in (0, 1)
        # fade[mask_between] = (1 + 10*np.exp(-10* (1 - 0.5))) / (1 + 10*np.exp(-10 * (fade[mask_between] - 0.5)))
        fade = (1 - np.exp(1.2 * smooth_mask_final)) / (1 - np.exp(1.2))

        # Explicitly set inside and outside
        # fade[smooth_mask_final >= 1] = 1
        fade[smooth_mask_final <= 0] = 0

        smoothed_mask = fade * ratio
        smoothed_mask = smoothed_mask + 1
        
        return smoothed_mask
    
    def load_data(self):
        """
        Load and preprocess data from dataset directory, including masks, splitting into
        training (90%), validation (5%), and test (5%) sets. Organizes frames into fixed
        and moving images, displacement fields into x and y components, and applies weighted masks.
        """
        folders_in_train_simulator_directory = os.listdir(self.dataset_path)
        for i, directory in enumerate(folders_in_train_simulator_directory):
            if directory == "Displacement":
                Displacement_directory = os.path.join(self.dataset_path, directory)
            elif directory == "Frames":
                Frames_directory = os.path.join(self.dataset_path, directory)
            elif directory == "Masks":
                Masks_directory = os.path.join(self.dataset_path, directory)
     

        files_in_Masks_directory        = os.listdir(Masks_directory)
        files_in_Displacement_directory = os.listdir(Displacement_directory)
        files_in_Frames_directory       = os.listdir(Frames_directory)
        first_image_in_Frames_directory = os.path.join(Frames_directory, files_in_Frames_directory[0])
        image = np.load(first_image_in_Frames_directory)
        first_image_in_Displacement_directory = os.path.join(Displacement_directory, files_in_Displacement_directory[0])
        image = np.load(first_image_in_Displacement_directory)
        files_in_Masks_directory.sort()
        files_in_Displacement_directory.sort()
        files_in_Frames_directory.sort()
        first_frame_dict = {}
        second_frame_dict = {}
        masks_dict = {}
        for mask in files_in_Masks_directory:
            mask_path = os.path.join(Masks_directory, mask)
            image = np.load(mask_path)
            if image.any() !=1:
                continue 
            # dilate the mask
            image = self.create_weighted_mask(image)
            # ratio back ground to myocardium
                 
            id = mask.split('_')[:-1]
            id = '_'.join(id)
            masks_dict[id] = image

        for file in files_in_Frames_directory:
            file_path = os.path.join(Frames_directory, file)
            if file_path.endswith(".DS_Store"):
                continue
            image = np.load(file_path, allow_pickle=True)
            #convert image to float
            # image = image.astype(np.float32)
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
        mask_image = [] 
        for key in first_frame_dict.keys():
            if key not in masks_dict.keys() or key not in second_frame_dict.keys() or key not in x_displacement_dict.keys() or key not in y_displacement_dict.keys():
                continue
            moving_image.append(first_frame_dict[key])
            fixed_image.append(second_frame_dict[key])
            x_image.append(x_displacement_dict[key])
            y_image.append(y_displacement_dict[key])
            mask_image.append(masks_dict[key])

        displacement_array = np.array([x_image, y_image, mask_image])
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
        
        

    def apply_displacement( self,image, x_displacement, y_displacement):
        """
        Apply displacement fields to an image using remapping to generate a warped image.

        Args:
            image (tf.Tensor): Input image to be warped.
            x_displacement (np.ndarray): X-component of displacement field.
            y_displacement (np.ndarray): Y-component of displacement field.

        Returns:
            np.ndarray: Warped image after applying displacement.
        """
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
        """
        Calculate principal strains from displacement fields, ensuring strains are within a threshold.

        Args:
            x_displacement (np.ndarray): X-component of displacement field.
            y_displacement (np.ndarray): Y-component of displacement field.

        Returns:
            tuple: Principal strains (Ep1All, Ep2All, Ep3All).
        """
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
        """
        Visualize model outputs, including loss plots, evaluation metrics, and image comparisons
        for training, test, and real test datasets, incorporating weighted masks.

        Args:
            model_name (str): Name of the model.
            history (History): Training history object from model.fit().
            model (Model): Trained Keras model.
            inc_history (bool): Whether to include training history plots.

        Returns:
            None: Saves various plots and evaluation results to the model folder.
        """
        #Create folder for the model
        #mkdir
        data = {}
        model_name = model_name + "_with_mask"
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
     

        data['moving'] = real_moving_image
        data['fixed'] = real_fixed_image

        real_moving_image_expanded = tf.expand_dims(real_moving_image, axis=0)
        real_fixed_image_expanded = tf.expand_dims(real_fixed_image, axis=0)
        
        predicted_deformation_field = model.predict([real_moving_image_expanded, real_fixed_image_expanded])
    
        data['displacements'] = predicted_deformation_field[0]

        x_displacement_predicted = predicted_deformation_field[0, :, :, 0]  
        y_displacement_predicted = predicted_deformation_field[0, :, :, 1]
       

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

        model_name_train = model_name + "_train" + "with_mask"
        self.create_interactive_plots(data, model_name_train, model_folder)

        

         ########################################### plot the direction test  ########################################################################
        test_sample = 67
        
        moving_image_try = self.moving_images_test[test_sample]
        fixed_image_try = self.fixed_images_test[test_sample]
        data['moving'] = moving_image_try
        data['fixed'] = fixed_image_try
        moving_image_try = tf.expand_dims(moving_image_try, axis=0)
        fixed_image_try = tf.expand_dims(fixed_image_try, axis=0)
        predicted_deformation_field = model.predict([moving_image_try, fixed_image_try])
        data['displacements'] = predicted_deformation_field[0]
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
        model_name_test = model_name + "_test_withmask"
        self.create_interactive_plots(data, model_name_test, model_folder)


        

        ############# plot x_displacement over train #############
        moving_image   = self.moving_images_test[test_sample]
        x_displacement = self.y_test[test_sample,...,0]
        plt.imshow(moving_image, cmap= 'gray')
        plt.imshow(x_displacement, cmap= 'jet', alpha = 0.5)
        displacement_plot_path = os.path.join(model_folder, f"displacement_plot_test_simulated{model_name}.png")
        plt.savefig(displacement_plot_path)
        plt.close()

        

    def train_models(self, num_epochs = 10, batch_size = 32):
        """
            Train multiple models on the dataset with specified epochs and batch size, using
            custom loss and metric functions (MaskLoss, MAELoss) for weighted mask integration.

            Args:
                num_epochs (int): Number of training epochs (default: 10).
                batch_size (int): Batch size for training (default: 32).

            Returns:
                None: Trains models and saves results and visualizations.
            """
        for model in self.models_list:
            
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, clipvalue=1.0)
            fixed_input = Input(shape=(128, 128, 1), name="fixed_image")
            moving_input = Input(shape=(128, 128, 1), name="moving_image")
            modelnet = model
            model_name = model.__class__.__name__
            self.model_names.append(model_name)
            out_def = modelnet([moving_input, fixed_input])
            model = Model(inputs=[moving_input, fixed_input], outputs=out_def)

            model.compile(optimizer=optimizer, loss=MaskLoss(), metrics=[MAELoss()])

            file_name = f"{model_name}_with_mask.keras"  
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
            
            self.visualise_outputs(model_name, history, model)

       
    


# main function
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
    
    # model_name = "Unet_loaded"
    # history = None
    trainer.train_models(num_epochs = 120, batch_size = 32)
    # model = tf.keras.models.load_model("/Users/ahmed_ali/Library/CloudStorage/GoogleDrive-ahmed.rajab502@eng-st.cu.edu.eg/My Drive/Models/Unet.keras", custom_objects={'MaskLoss': MaskLoss, 'MAELoss': MAELoss, 'Unet': Unet})
    # trainer.visualise_outputs(model_name, history, model, inc_history = False)

    