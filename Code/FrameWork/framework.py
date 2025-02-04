import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
import matplotlib.pyplot as plt
from keras.layers import Input, concatenate, add, Multiply, Lambda
from keras.layers import Conv3D, MaxPooling3D, MaxPooling2D, UpSampling2D,UpSampling3D, Conv2D
from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.models import Model
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.initializers import HeUniform
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, Concatenate, Add, Multiply, BatchNormalization, Activation
from tensorflow.keras.models import Model

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
            image = image.astype(np.float32)
            #normalize image
            image = image / 255.0
            print(np.max(image))
            print(np.min(image))
            print(image.shape)
            frame_id = file.split('_')[-1].split('.')[0]
            print(frame_id)
            id = file.split('_')[:-1]
            id = '_'.join(id)
            
            if frame_id == "1" :
                first_frame_dict[id] = image
            else:
                second_frame_dict[id] = image
        fixed_image = []
        moving_image = []
        for key in first_frame_dict.keys():
            moving_image.append(first_frame_dict[key])
            fixed_image.append(second_frame_dict[key])

        x_displacement_dict = {}
        y_displacement_dict = {}
        for file in files_in_Displacement_directory:
            file_path = os.path.join(Displacement_directory , file)
            image = np.load(file_path)

            print(image.shape)
            frame_id = file.split('_')[-1].split('.')[0]
            print(frame_id)
            id = file.split('_')[:-1]
            id = '_'.join(id)
            print(id)
            if frame_id == "x" :
                x_displacement_dict[id] = image
            else:
                y_displacement_dict[id] = image

        x_image = []
        y_image = []
        for key in x_displacement_dict.keys():
            x_image.append(x_displacement_dict[key])
            y_image.append(y_displacement_dict[key])

        displacement_array = np.array([x_image, y_image])
        displacement_array = np.transpose(displacement_array, (1, 2, 3,0))
        fixed_image_array = np.array(fixed_image)
        moving_image_array = np.array(moving_image)
        self.fixed_images_train = fixed_image_array[:300]
        self.fixed_images_test = fixed_image_array[300:]
        self.moving_images_train = moving_image_array[:300]
        self.moving_images_test = moving_image_array[300:]
        self.y_train = displacement_array[:300]
        self.y_test = displacement_array[300:]

    def train_models(self, num_epochs = 10, batch_size = 32):
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, clipvalue=1.0)
        for model in self.models_list:
            fixed_input = Input(shape=(128, 128, 1), name="fixed_image")
            moving_input = Input(shape=(128, 128, 1), name="moving_image")
            modelnet = model
            out_def = modelnet([moving_input, fixed_input])
            model = Model(inputs=[moving_input, fixed_input], outputs=out_def)

            model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

            file_name = f"{model}.keras"  # Name files as 1.keras, 2.keras, etc.
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
            history = model.fit(
                [self.moving_images_train, self.fixed_images_train],
                self.y_train,
                batch_size=batch_size,
                epochs = num_epochs,
                validation_data=([self.moving_images_test, self.fixed_images_test], self.y_test),
                callbacks=[checkpoint_callback]
            )
            # Plot the training and validation loss
            plt.plot(history.history['loss'], label='Training Loss')
            plt.plot(history.history['val_loss'], label='Validation Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.title('Loss During Training')
            plt.legend()
            loss_plot_path = os.path.join(self.save_dir, f"loss_plot{model}.png")
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
            results_path = os.path.join(self.save_dir, f"evaluation_results{model}.txt")

            with open(results_path, "w") as f:
                f.write(f"Test Loss: {test_loss:.4f}\n")
                f.write(f"Test MAE: {test_mae:.4f}\n")



