import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.cm as cm
from scipy.ndimage import gaussian_filter1d
import cv2
from PIL import Image as im 

from wave_genrator import Wave_Generator
from strain_validation import limit_strain_range, plot_strain_results

logging.getLogger().setLevel(logging.INFO)  # Allow printing info level logs
os.chdir(os.path.dirname(__file__)) #change working directory to current directory


DISPLACMENET_MULTIPLAYER = 5e7
GAUSSIAN_SIGMA = 50

class Create_Displacement_Masks:
    def __init__(self, path, save_mode=False):
        self.image_path = path
        self.image = None
        self.wave = Wave_Generator()
        self.param = self.wave.param
        self.H0, self.W, self.Grid_Sign = self.wave.initialize_wave()
        self.displaced_image_stack = []
        self.iterator = 0
        self.mask = None
        self.frame_count = 0
        self.save_mode = save_mode
        self.finished = False
        self.plot_strain = False

    def apply_displacement(self, image, x_displacement, y_displacement):
        # Prepare meshgrid for remap
        height, width, _ = image.shape
        x, y = np.meshgrid(np.arange(width), np.arange(height))

        # Apply displacement (scale the displacements for more visible effect)
        x_new = (x + x_displacement).astype(np.float32)
        y_new = (y + y_displacement).astype(np.float32)

        # Warp the image using remap for both x and y displacements
        displaced_image = cv2.remap(image, x_new, y_new, interpolation=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT)
        return displaced_image
    
    def save_stack(self):
        import os
        from PIL import Image

        output_dir = "displaced_images_test"
        os.makedirs(output_dir, exist_ok=True)
        np.savez_compressed("displaced_images/displaced_images.npz", *self.displaced_image_stack)
        # np.savez_compressed("strain_values.npz", *self.strain_values)

    def plot(self):
        print("Creating displacement masks...")
        fig = plt.figure(figsize=(20, 10))
        self.displaced_image_stack = []
        fig.set_facecolor('#eeeeef')
        # Setup subplots
        ax_wave = fig.add_subplot(2, 5, 4, projection='3d')  # 3D wave plot
        ax_zx = fig.add_subplot(2, 5, 2)
        ax_zy = fig.add_subplot(2, 5, 3)
        ax_image = fig.add_subplot(1, 5, 1)
        ax_displaced = fig.add_subplot(1, 5, 4)
        ax_x_displacement = fig.add_subplot(1, 5, 2)
        ax_y_displacement = fig.add_subplot(1, 5, 3)

        # Load the initial image
        image = self.load_image()

        # Initialize the wave grid
        x = np.linspace(self.param['xLim'][0], self.param['xLim'][1], self.param['meshsize'])
        y = np.linspace(self.param['yLim'][0], self.param['yLim'][1], self.param['meshsize'])
        X, Y = np.meshgrid(x, y)

        # Plot the initial data
        Z = self.wave.calc_wave(self.H0, self.W, 0, self.Grid_Sign)
        Z = gaussian_filter1d(Z, sigma=GAUSSIAN_SIGMA, axis=0)
        Zx, Zy = np.gradient(Z)

        binarized_image = np.where(image > 128, 1, 0)
        x_displacement = np.clip(Zx * 50, -20, 20).astype(np.float32)  # Scale by 50 for more visible effect
        y_displacement = np.clip(Zy * 50, -20, 20).astype(np.float32)
        x_displacement = y_displaced_image = 0
        displaced_image = self.apply_displacement(image, x_displacement, y_displacement)
        x_displaced_image = self.apply_displacement(image, x_displacement, 0)
        y_displaced_image = self.apply_displacement(image, 0, y_displacement)
        binarized_image = np.where(displaced_image > 128, 1, 0)

        # Initial plots
        wave_surf = ax_wave.plot_surface(X, Y, Z, cmap=cm.coolwarm)
        zx_img = ax_zx.imshow(Zx, cmap='coolwarm')
        zy_img = ax_zy.imshow(Zy, cmap='coolwarm')
        image_plot = ax_image.imshow(image)
        ax_x_displacement.imshow(image)
        ax_y_displacement.imshow(image)
        displaced_image_plot = ax_displaced.imshow(displaced_image,cmap='viridis')
        x_displaced_image_plot = ax_x_displacement.imshow(x_displaced_image)
        y_displaced_image_plot = ax_y_displacement.imshow(y_displaced_image)
        ax_displaced.set_facecolor('#eeeeef')

        # Function to update the plots
        def update(frame):
            nonlocal displaced_image, x_displaced_image, y_displaced_image # To ensure displaced_image is updated across frames
            # print(frame)
            self.frame_count += 1
            if self.frame_count >= 30:
                self.finished = True
            Z = self.wave.calc_wave(self.H0, self.W, frame, self.Grid_Sign)
            Z = gaussian_filter1d(Z, sigma=GAUSSIAN_SIGMA, axis=0)
            Zx, Zy = np.gradient(Z)

            Zx_disp = np.clip(Zx * 50, -20, 20).astype(np.float32) * DISPLACMENET_MULTIPLAYER
            Zy_dsip = np.clip(Zy * 50, -20, 20).astype(np.float32) * DISPLACMENET_MULTIPLAYER

            print(f"Max Value in Zx before strain validation: {np.max(Zx_disp)}")
            result = limit_strain_range(Zx_disp, Zy_dsip, strain_lower_bound=0, strain_upper_bound=0.3)
            Zx_disp, Zy_dsip, initial_strain, final_strain, max_initial, max_final, min_initial, min_final = result

            if(self.plot_strain): 
                plot_strain_results(
                    initial_strain, final_strain, 
                    min_initial, max_initial, 
                    min_final, max_final,
                    strain_lower_bound = 0, strain_upper_bound = 0.1
                    )
                self.plot_strain = False

            # Apply the displacements to the previously displaced image (cumulative effect)
            displaced_image = self.apply_displacement(displaced_image, Zx_disp, Zy_dsip)
            x_displaced_image = self.apply_displacement(x_displaced_image, Zx_disp, 0)
            y_displaced_image = self.apply_displacement(y_displaced_image, 0, Zy_dsip)

            binarized_image = np.where(displaced_image > 128, 1, 0)
            self.displaced_image_stack.append(binarized_image)         

            # Update the 3D wave plot
            ax_wave.cla()  # Remove previous surface to avoid plotting over it
            ax_wave.plot_surface(X, Y, Z, cmap=cm.coolwarm)
            

            # Update the x and y displacement images
            zx_img.set_data(Zx)
            zy_img.set_data(Zy)

            # Update the displaced image plot
            displaced_image_plot.set_data(displaced_image)
            x_displaced_image_plot.set_data(x_displaced_image)
            y_displaced_image_plot.set_data(y_displaced_image)

            # self.save_stack()
            print(f"Finished? : {self.finished}")

            return wave_surf, zx_img, zy_img, displaced_image_plot


        # Animate the wave and displacements
        ani = FuncAnimation(fig, update, frames=np.linspace(0, 31, 31), blit=False, repeat=False)
        plt.tight_layout()
        #add padding between subplots
        plt.subplots_adjust(wspace=0.5, hspace=0.5)
        
        if self.save_mode:
            ani.save('trash/wave_displacement.mp4', writer='ffmpeg')
        else:
            plt.show()
        

    def load_image(self):
        # Load the image array from the .npy file
        array = np.load(self.image_path)
        array = array[1]
        array = np.where(array > 1, 1, 0)
        
        # Extract the first image if the array has extra dimensions
        if len(array.shape) > 3 or (len(array.shape) == 3 and array.shape[0] not in [1, 3]):
            image_array = array[0]  # Adjust indexing as needed for your specific data
        else:
            image_array = array

        # Ensure the array is in uint8 format (0-255 range) for proper visualization
        if image_array.dtype != np.uint8:
            image_array = cv2.normalize(image_array, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Handle grayscale or RGB format
        if len(image_array.shape) == 2:  # Grayscale
            self.image = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)  # Convert to RGB for consistency
        elif len(image_array.shape) == 3 and image_array.shape[2] in [3, 4]:  # RGB or RGBA
            self.image = image_array[:, :, :3]  # Discard alpha if present
        else:
            raise ValueError("Unexpected image shape, unable to load the image properly.")

        return self.image
    
    def check_status(self):
        if self.finished:
            return True
        else:
            return False
displacer = Create_Displacement_Masks(path="/Users/osama/GP-2025-Strain/Data/ACDC/train_numpy/patient001/patient001_frame01_slice_5_ACDC.npy")
displacer.plot()