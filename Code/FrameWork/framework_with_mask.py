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
import sys
from mpl_toolkits.axes_grid1 import make_axes_locatable
sys.path.append(os.path.abspath("Code/Supervised_deformation/Models"))
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

    def create_weighted_mask(self,mask, dilation_extent=10, sigma=2):
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
        # Extract myocardium (label 1)
        myocardium = (mask == 1).astype(np.float32)
        
        # Initialize dilated mask and process mask
        dilated_mask = myocardium.copy()
        process_mask = myocardium.copy()
        
        # Kernel for dilation (7x7 ellipse)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        
        # Gradually reduce the added weight per iteration
        initial_value = 0.9
        step_size = initial_value / dilation_extent  # Controls decay per iteration
        
        for i in range(dilation_extent):
            old_process_mask = process_mask.copy()
            process_mask = cv2.dilate(process_mask, kernel)
            
            # Identify newly added pixels (boundary of the dilated region)
            added_region = (process_mask - old_process_mask).astype(np.float32)

            # Ensure added_region has the same number of dimensions as dilated_mask
            added_region = added_region[..., np.newaxis] if added_region.ndim < dilated_mask.ndim else added_region

            
            # Compute weight for this iteration (decays linearly with iterations)
            current_weight = initial_value - i * step_size
            
            # Update dilated mask with decaying weights
            dilated_mask += added_region * current_weight
        
        # Smooth the dilation
        smoothed_mask = gaussian_filter(dilated_mask, sigma=sigma)
        
        # Make sure the myocardium part is exactly 1 and has no dilation
        smoothed_mask[myocardium.astype(bool)] = 1.0

        # Add 1 to the mask
        smoothed_mask += 1.0
        
        return smoothed_mask
    
    def load_data(self):
        folders_in_train_simulator_directory = os.listdir(self.dataset_path)
        for i, directory in enumerate(folders_in_train_simulator_directory):
            if directory == "Displacement_loc":
                Displacement_directory = os.path.join(self.dataset_path, directory)
            elif directory == "Frames_loc":
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
        first_frame_dict = {}
        second_frame_dict = {}
        masks_dict = {}
        for mask in files_in_Masks_directory:
            mask_path = os.path.join(Masks_directory, mask)
            image = np.load(mask_path)   
            # dilate the mask
            image = self.create_weighted_mask(image)      
            id = mask.split('_')[:-1]
            id = '_'.join(id)
            masks_dict[id] = image

        for file in files_in_Frames_directory:
            file_path = os.path.join(Frames_directory, file)
            image = np.load(file_path)
            #convert image to float
            image = image.astype(np.float32)
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
            image = np.load(file_path)

            
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
            if key not in masks_dict.keys():
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

        train_index = 5
        valid_index = 7
        self.fixed_images_train = fixed_image_array[:train_index]
        self.fixed_images_valid = fixed_image_array[train_index:valid_index]
        self.fixed_images_test = fixed_image_array[valid_index:10]
        self.moving_images_train = moving_image_array[:train_index]
        self.moving_images_valid = moving_image_array[train_index:valid_index]
        self.moving_images_test = moving_image_array[valid_index:10]

        self.y_train = displacement_array[:train_index]
        self.y_valid = displacement_array[train_index:valid_index]
        self.y_test = displacement_array[valid_index:10]

     
    
        
        

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
        deltaT = 1  # Time step
        StrainEpPeak = 0.2  # Strain limit
        CineNoFrames = 2  # Number of frames

        # Simulated displacement field (random example, replace with actual data)
        FrameDisplX = x_displacement
        FrameDisplY =  y_displacement
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
            print(Ep3All.shape)
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

    def visualise_outputs(self, model_name, history, model):
        #Create folder for the model
        #mkdir
        model_folder = os.path.join(self.save_dir, model_name)
        os.makedirs(model_folder, exist_ok=True)
        self.all_losses.append(history.history['loss'])
        self.all_val_losses.append(history.history['val_loss'])

        np.save(os.path.join(self.save_dir, "all_losses.npy"), self.all_losses)
        np.save(os.path.join(self.save_dir, "all_val_losses.npy"), self.all_val_losses)
        
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
        ############# save the model results ############
        # Save the evaluation results
        results_path = os.path.join(model_folder, f"evaluation_results{model_name}.txt")

        with open(results_path, "w") as f:
            f.write(f"Test Loss: {test_loss:.4f}\n")
            f.write(f"Test MAE: {test_mae:.4f}\n")
        
        #############plot direction real test ############
        files_real_test = os.listdir(self.real_test_data_path)
        for file in files_real_test:
            if file == "patient_053_frame_0_slice3.npy":
                moving_image = np.load(os.path.join(self.real_test_data_path, file))
            elif file == "patient_053_frame_4_slice3.npy":
                fixed_image = np.load(os.path.join(self.real_test_data_path, file))
            elif file == "patient_053_frame_4_slice3_mask.npy":
                mask_image = np.load(os.path.join(self.real_test_data_path, file))


        moving_image = cv2.normalize(moving_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        fixed_image = cv2.normalize(fixed_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        print(moving_image.shape)
        print(fixed_image.shape)
        moving_image_expanded = tf.expand_dims(moving_image, axis=0)
        fixed_image_expanded = tf.expand_dims(fixed_image, axis=0)
        print(moving_image_expanded.shape)
        print(fixed_image_expanded.shape)
        predicted_deformation_field = model.predict([moving_image_expanded, fixed_image_expanded])
        x_displacement_predicted = predicted_deformation_field[0, :, :, 0]  
        y_displacement_predicted = predicted_deformation_field[0, :, :, 1]
        for i in range(0, 128, 10):
            for j in range(0, 128, 10):
                plt.arrow(j, i, x_displacement_predicted[i, j], y_displacement_predicted[i, j], head_width=0.5, head_length=0.7, fc='b', ec='b')
        plt.title('Predicted Displacement')
        direction_plot_path = os.path.join(model_folder, f"direction_plot_real_test{model_name}.png")
        plt.savefig(direction_plot_path)
        plt.close()
        ############# plot strain over the real data ############
        Ep1All, Ep2All, Ep3All = self.calculate_strain(x_displacement_predicted, y_displacement_predicted)
        fig, ax = plt.subplots(1, 4, figsize=(15, 5))
        ax[0].imshow(moving_image, cmap='gray')  
        ax[0].set_title('Moving Image')

        ax[1].imshow(fixed_image, cmap='gray')
        ax[1].set_title('Fixed Image')

        ax[2].imshow(fixed_image, cmap='gray')
        im1 = ax[2].imshow(Ep1All, cmap='coolwarm', alpha=0.5)
        
        divider1 = make_axes_locatable(ax[2])
        cax1 = divider1.append_axes("right", size="5%", pad=0.1)
        fig.colorbar(im1, cax=cax1, shrink=0.8)
        ax[2].set_title('Ep1All')

        ax[3].imshow(fixed_image, cmap='gray')
        im2 = ax[3].imshow(Ep2All, cmap='coolwarm', alpha=0.5)
        
        divider2 = make_axes_locatable(ax[3])
        cax2 = divider2.append_axes("right", size="5%", pad=0.1)
        fig.colorbar(im2, cax=cax2, shrink=0.8)
        ax[3].set_title('Ep2All')

        for i in range (4):
            ax[i].axis('off')
        strain_plot_path = os.path.join(model_folder, f"strain_plot_real_test{model_name}.png")
        plt.savefig(strain_plot_path)
        plt.close()

        ############ plot moving and fixed and warped images real test ########

        fixed_image_try = fixed_image
        moving_image_try = moving_image
        original_map = fixed_image_try - moving_image_try
        # absolute value of the original map
        original_map = np.abs(original_map)
        
        moving_image_try = tf.expand_dims(moving_image_try, axis=-1)
        print(moving_image_try.shape)
        warped_image = self.apply_displacement(moving_image_try, x_displacement_predicted, y_displacement_predicted)
        difference_image = fixed_image_try - warped_image
        #absolute value of the difference image
        difference_image = np.abs(difference_image)

        vmin, vmax = np.min(original_map), np.max(original_map)
        
        fig, ax = plt.subplots(1, 5, figsize=(18, 5))  # Increase figure width

        ax[0].imshow(fixed_image_try)
        ax[0].set_title('Fixed Image')

        ax[1].imshow(moving_image_try)
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
        ax[4].imshow(masked_overlay, cmap='grey', alpha=0.5)
        
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
        ############# plot the direction test  ############
        moving_image_try = self.moving_images_test[0]
        fixed_image_try = self.fixed_images_test[0]
        moving_image_try = tf.expand_dims(moving_image_try, axis=0)
        fixed_image_try = tf.expand_dims(fixed_image_try, axis=0)
        predicted_deformation_field = model.predict([moving_image_try, fixed_image_try])
        x_displacement_predicted = predicted_deformation_field[0, :, :, 0]
        y_displacement_predicted = predicted_deformation_field[0, :, :, 1]
        x_displacement = self.y_test[0, :, :, 0]
        y_displacement = self.y_test[0, :, :, 1]
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
        
        ############### plot magnitude of the displacement test ########
        magnitude_actual = np.sqrt(x_displacement ** 2 + y_displacement ** 2)
        magnitude_predicted = np.sqrt(x_displacement_predicted ** 2 + y_displacement_predicted ** 2)
        magnitude_difference = np.sqrt(difference_x ** 2 + difference_y ** 2)
        fig, ax = plt.subplots(1, 3, figsize=(10, 5))
        ax[0].imshow(magnitude_actual, cmap='hot')
        ax[0].set_title('Actual Magnitude')
        ax[1].imshow(magnitude_predicted, cmap='hot')
        ax[1].set_title('Predicted Magnitude')
        ax[2].imshow(magnitude_difference, cmap='hot')
        ax[2].set_title('Difference')
        magnitude_plot_path = os.path.join(model_folder, f"magnitude_plot_test{model_name}.png")
        plt.savefig(magnitude_plot_path)
        plt.close()


        ######### plot the moving and fixed and warped images test ########
        fixed_image_try = self.fixed_images_test[0]
        moving_image_try = self.moving_images_test[0]
        original_map = fixed_image_try - moving_image_try
        # absolute value of the original map
        original_map = np.abs(original_map)
        
        moving_image_try = tf.expand_dims(moving_image_try, axis=-1)
        print(moving_image_try.shape)
        warped_image = self.apply_displacement(moving_image_try, x_displacement_predicted, y_displacement_predicted)
        difference_image = fixed_image_try - warped_image
        #absolute value of the difference image
        difference_image = np.abs(difference_image)

        vmin, vmax = np.min(original_map), np.max(original_map)
        
        fig, ax = plt.subplots(1, 5, figsize=(18, 5))  # Increase figure width

        ax[0].imshow(fixed_image_try)
        ax[0].set_title('Fixed Image')

        ax[1].imshow(moving_image_try)
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
        
        im2 = ax[4].imshow(difference_image, cmap='hot', vmin=vmin, vmax=vmax)
        # mask is the y test where values are 2
        mask = self.y_test[0, :, :, 2]
        masked_overlay = np.ma.masked_where(mask < 2, mask)


        
        # Overlay the mask on top of the difference image
        ax[4].imshow(masked_overlay, cmap='jet', alpha=0.5, interpolation='none')

        # overlay almask with alpha = 0.5
        divider2 = make_axes_locatable(ax[4])
        cax2 = divider2.append_axes("right", size="5%", pad=0.1)
        fig.colorbar(im2, cax=cax2, shrink=0.8)
        ax[4].set_title('Final Difference')

        for i in range (5):
            ax[i].axis('off')
        # Adjust spacing between subplots
        fig.subplots_adjust(wspace=0.4)
        warped_image_path = os.path.join(model_folder, f"warped_image_test{model_name}.png")
        plt.savefig(warped_image_path)
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

            model.compile(optimizer=optimizer, loss=MaskLoss(), metrics=[MAELoss()])

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
                patience=35,
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

        all_losses = np.array(self.all_losses)
        all_val_losses = np.array(self.all_val_losses)
        #save np file for all losses
        np.save(os.path.join(self.save_dir, "all_losses.npy"), all_losses)
        np.save(os.path.join(self.save_dir, "all_val_losses.npy"), all_val_losses)
        # Plot the training and validation loss
        for i, loss in enumerate(all_losses):
            plt.plot(loss, label=self.model_names[i])

        plt.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=8, ncol=2)  
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('ALL Loss During Training')


        # Save the figure
        all_loss_plot_path = os.path.join(self.save_dir, "all_loss_plot.png")
        plt.savefig(all_loss_plot_path, bbox_inches="tight")  # Ensures full image is saved
        plt.close()
        # Plot the training and validation loss
        for i, loss in enumerate(all_val_losses):
            plt.plot(loss, label=self.model_names[i])
            plt.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=8, ncol=2)  
        
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('ALL Validation Loss During Training')
       
        all_val_loss_plot_path = os.path.join(self.save_dir, "all_val_loss_plot.png")
        plt.savefig(all_val_loss_plot_path)
        plt.close()
        
    


            

            
            
            




# main function
if __name__ == '__main__':
    # check for the gpu
    # physical_devices = tf.config.experimental.list_physical_devices('GPU')
    # print("Num GPUs Available: ", len(physical_devices))

    dataset_path = "/Users/ahmed_ali/Documents/GitHub/GP-2025-Strain/Data/simulated_data_4000_loc"
    real_test_data_path = "/Users/ahmed_ali/Documents/GitHub/GP-2025-Strain/Code/FrameWork/real_test_data"
    current_script = Path(__file__)
    models_list = [ Residual_Unet(),Unet(), Unet_7Kernel(), Unet_5Kernel(), Unet_3Dense(), Unet_1Dense(), Unet_2Dense(), Unet_1Dense_7Kernel(), Unet_1Dense_5Kernel(), Unet_2Dense_7Kernel(), Unet_2Dense_5Kernel(), Unet_3Dense_7Kernel(), Unet_3Dense_5Kernel(), Residual_Unet_1D(), Residual_Unet_2D(), Residual_Unet_3D(), Residual_Unet_1D_7K(), Residual_Unet_1D_5K(), Residual_Unet_2D_7K(), Residual_Unet_2D_5K(), Residual_Unet_3D_7K(), Residual_Unet_3D_5K()]
    models_list=[Unet()]
    save_dir = current_script.parent / "Saved"
    saved_model_dir = current_script.parent / "Models"
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(saved_model_dir, exist_ok=True)
    trainer = Automate_Training(dataset_path,real_test_data_path, models_list, save_dir, saved_model_dir)
    trainer.train_models(num_epochs = 2, batch_size = 32)

    