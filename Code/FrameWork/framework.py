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
from pathlib import Path
import os
import cv2
import sys
sys.path.append(os.path.abspath("Code/Supervised_deformation/Models"))
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
    def __init__(self, dataset_path, models_list, save_dir, saved_model_dir):
        self.dataset_path = dataset_path
        self.models_list = models_list
        self.save_dir = save_dir
        self.saved_model_dir = saved_model_dir
        self.fixed_images_train = None
        self.fixed_images_test = None
        self.moving_images_train = None
        self.moving_images_test = None
        self.y_train = None
        self.y_test = None
        self.load_data()
    
    def load_data(self):
        folders_in_train_simulator_directory = os.listdir(self.dataset_path)
        Displacement_directory = os.path.join(self.dataset_path, folders_in_train_simulator_directory[0])
        Frames_directory = os.path.join(self.dataset_path, folders_in_train_simulator_directory[1])
        files_in_Displacement_directory = os.listdir(Displacement_directory)
        files_in_Frames_directory = os.listdir(Frames_directory)
        first_image_in_Frames_directory = os.path.join(Frames_directory, files_in_Frames_directory[0])
        image = np.load(first_image_in_Frames_directory)
        first_image_in_Displacement_directory = os.path.join(Displacement_directory, files_in_Displacement_directory[0])
        image = np.load(first_image_in_Displacement_directory)
        first_frame_dict = {}
        second_frame_dict = {}
        for file in files_in_Frames_directory:
            file_path = os.path.join(Frames_directory, file)
            image = np.load(file_path)
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
        for key in first_frame_dict.keys():
            moving_image.append(first_frame_dict[key])
            fixed_image.append(second_frame_dict[key])
            x_image.append(x_displacement_dict[key])
            y_image.append(y_displacement_dict[key])

        displacement_array = np.array([x_image, y_image])
        displacement_array = np.transpose(displacement_array, (1, 2, 3,0))
        fixed_image_array = np.array(fixed_image)
        moving_image_array = np.array(moving_image)
        self.fixed_images_train = fixed_image_array[:300]
        self.fixed_images_valid = fixed_image_array[300:304]
        self.fixed_images_test = fixed_image_array[304:]
        self.moving_images_train = moving_image_array[:300]
        self.moving_images_valid = moving_image_array[300:304]
        self.moving_images_test = moving_image_array[304:]

        self.y_train = displacement_array[:300]
        self.y_valid = displacement_array[300:304]
        self.y_test = displacement_array[304:]

        

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

    def train_models(self, num_epochs = 10, batch_size = 32):
       
        for model in self.models_list:
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, clipvalue=1.0)
            fixed_input = Input(shape=(128, 128, 1), name="fixed_image")
            moving_input = Input(shape=(128, 128, 1), name="moving_image")
            modelnet = model
            model_name = model.__class__.__name__
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
                monitor='loss',
                patience=15,
                mode='min',
                verbose=1
            )
            history = model.fit(
                [self.moving_images_train, self.fixed_images_train],
                self.y_train,
                batch_size=batch_size,
                epochs = num_epochs,
                validation_data=([self.moving_images_valid, self.fixed_images_valid], self.y_valid),
                callbacks=[checkpoint_callback, stopping_callback],
            )
            #Create folder for the model
            #mkdir
            model_folder = os.path.join(self.save_dir, model_name)
            os.makedirs(model_folder, exist_ok=True)

            
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
            
            ############# plot the direction train  ############
            moving_image_try = self.moving_images_train[94]
            fixed_image_try = self.fixed_images_train[94]
            moving_image_try = tf.expand_dims(moving_image_try, axis=0)
            fixed_image_try = tf.expand_dims(fixed_image_try, axis=0)
            predicted_deformation_field = model.predict([moving_image_try, fixed_image_try])
            x_displacement_predicted = predicted_deformation_field[0, :, :, 0]
            y_displacement_predicted = predicted_deformation_field[0, :, :, 1]
            x_displacement = self.y_train[94, :, :, 0]
            y_displacement = self.y_train[94, :, :, 1]
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            for i in range(0, 128, 10):  # Adjust the step for better visualization
                for j in range(0, 128, 10):
                    ax[0].arrow(j, i, x_displacement[i, j], y_displacement[i, j], head_width=0.5, head_length=0.7, fc='b', ec='b')

            for i in range(0, 128, 10):  # Adjust the step for better visualization
                for j in range(0, 128, 10):
                    ax[1].arrow(j, i, x_displacement_predicted[i, j], y_displacement_predicted[i, j], head_width=0.5, head_length=0.7, fc='b', ec='b')
            ax[0].set_title('Actual Displacement')
            ax[1].set_title('Predicted Displacement')
            direction_plot_path = os.path.join(model_folder, f"direction_plot_train{model_name}.png")
            plt.savefig(direction_plot_path)
            plt.close()

               ######### plot the moving and fixed and warped images train ########
            fixed_image_try = self.fixed_images_train[94]
            moving_image_try = self.moving_images_train[94]
            moving_image_try = tf.expand_dims(moving_image_try, axis=-1)
            print(moving_image_try.shape)
            warped_image = self.apply_displacement(moving_image_try, x_displacement_predicted, y_displacement_predicted)
            difference_image = fixed_image_try - warped_image
            fig, ax = plt.subplots(1, 4, figsize=(10, 5))
            ax[0].imshow(fixed_image_try)
            ax[0].set_title('Fixed Image')
            ax[1].imshow(moving_image_try)
            ax[1].set_title('Moving Image')
            ax[2].imshow(warped_image)
            ax[2].set_title('Warped Image')
            im = ax[3].imshow(difference_image, cmap='jet')
            fig.colorbar(im, ax=ax[3])  # Correct way to add colorbar
            ax[3].set_title('Difference Image')
            warped_image_path = os.path.join(model_folder, f"warped_image_train{model_name}.png")
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
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            for i in range(0, 128, 10):  # Adjust the step for better visualization
                for j in range(0, 128, 10):
                    ax[0].arrow(j, i, x_displacement[i, j], y_displacement[i, j], head_width=0.5, head_length=0.7, fc='b', ec='b')

            for i in range(0, 128, 10):  # Adjust the step for better visualization
                for j in range(0, 128, 10):
                    ax[1].arrow(j, i, x_displacement_predicted[i, j], y_displacement_predicted[i, j], head_width=0.5, head_length=0.7, fc='b', ec='b')
            ax[0].set_title('Actual Displacement')
            ax[1].set_title('Predicted Displacement')
            direction_plot_path = os.path.join(model_folder, f"direction_plot_test{model_name}.png")
            plt.savefig(direction_plot_path)
            plt.close()

         


            ######### plot the moving and fixed and warped images test ########
            fixed_image_try = self.fixed_images_test[0]
            moving_image_try = self.moving_images_test[0]
            moving_image_try = tf.expand_dims(moving_image_try, axis=-1)
            print(moving_image_try.shape)
            warped_image = self.apply_displacement(moving_image_try, x_displacement_predicted, y_displacement_predicted)
            difference_image = fixed_image_try - warped_image

            fig, ax = plt.subplots(1, 4, figsize=(10, 5))
            ax[0].imshow(fixed_image_try)
            ax[0].set_title('Fixed Image')
            ax[1].imshow(moving_image_try)
            ax[1].set_title('Moving Image')
            ax[2].imshow(warped_image)
            ax[2].set_title('Warped Image')
            im = ax[3].imshow(difference_image, cmap='jet')
            fig.colorbar(im, ax=ax[3])  # Correct way to add colorbar
            ax[3].set_title('Difference Image')
            warped_image_path = os.path.join(model_folder, f"warped_image_test{model_name}.png")
            plt.savefig(warped_image_path)
            plt.close()
            
            




# main function
if __name__ == '__main__':
    dataset_path = "/Users/ahmed_ali/Documents/GitHub/GP-2025-Strain/Data/Simulated_data_localized"
    current_script = Path(__file__)
    models_list = [ Residual_Unet(),Unet(), Unet_7Kernel(), Unet_5Kernel(), Unet_3Dense(), Unet_1Dense(), Unet_2Dense(), Unet_1Dense_7Kernel(), Unet_1Dense_5Kernel(), Unet_2Dense_7Kernel(), Unet_2Dense_5Kernel(), Unet_3Dense_7Kernel(), Unet_3Dense_5Kernel(), Residual_Unet_1D(), Residual_Unet_2D(), Residual_Unet_3D(), Residual_Unet_1D_7K(), Residual_Unet_1D_5K(), Residual_Unet_2D_7K(), Residual_Unet_2D_5K(), Residual_Unet_3D_7K(), Residual_Unet_3D_5K()]
    save_dir = current_script.parent / "Saved"
    saved_model_dir = current_script.parent / "Models"
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(saved_model_dir, exist_ok=True)
    trainer = Automate_Training(dataset_path, models_list, save_dir, saved_model_dir)
    trainer.train_models(num_epochs = 2, batch_size = 32)


