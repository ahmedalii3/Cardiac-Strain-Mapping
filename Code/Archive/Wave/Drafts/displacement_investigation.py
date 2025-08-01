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

logging.getLogger().setLevel(logging.INFO)  # Allow printing info level logs
os.chdir(os.path.dirname(__file__)) #change working directory to current directory



class Apply_Displacement:
    def __init__(self):

        # self.image_path = "SheppLogan_Phantom.svg.png"
        self.image_path = '/Users/osama/GP-2025-Strain/Data/ACDC/train_numpy/patient001/patient001_frame01_slice_3_ACDC.npy'
        self.image = None
        self.wave = Wave_Generator()
        self.param = self.wave.param
        self.H0, self.W, self.Grid_Sign = self.wave.initialize_wave()
        self.displaced_image_stack = []
        self.iterator = 0
        self.mask = None

        self.frame_count = 0

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

    def plot(self):
        fig = plt.figure(figsize=(20, 10))
        self.displaced_image_stack = []

        # Setup subplots
        ax_wave = fig.add_subplot(2, 5, 4, projection='3d')  # 3D wave plot
        ax_zx = fig.add_subplot(2, 5, 2)
        ax_zy = fig.add_subplot(2, 5, 3)
        ax_image = fig.add_subplot(2, 5, 4)
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
        Z = gaussian_filter1d(Z, sigma=50, axis=0)

        # self.masks = np.load('displaced_images/displaced_images.npz')
        self.masks = np.load('dilated_masks/dilated_masks.npz')
        displaced_mask = self.masks['arr_0']
        # displaced_mask = cv2.normalize(displaced_mask, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        # displaced_mask = cv2.cvtColor(displaced_mask, cv2.COLOR_RGB2GRAY)
        displaced_mask = displaced_mask[:,:,0]
        displaced_mask = displaced_mask.astype(np.float64)
        # displaced_mask = displaced_mask / 255
        Zx, Zy = np.gradient(Z) * displaced_mask

        natural_Zx, natural_Zy = np.gradient(Z)

        binarized_image = np.where(image > 128, 1, 0)
        self.displaced_image_stack.append(binarized_image)

        x_displacement = np.clip(Zx * 50, -20, 20).astype(np.float32)  # Scale by 50 for more visible effect
        y_displacement = np.clip(Zy * 50, -20, 20).astype(np.float32)
        x_displacement = y_displacement = 0
        displaced_image = self.apply_displacement(image, x_displacement, y_displacement)
        x_displaced_image = self.apply_displacement(image, x_displacement, 0)
        y_displaced_image = self.apply_displacement(image, 0, y_displacement)

        binarized_image = np.where(displaced_image > 128, 1, 0)
        self.displaced_image_stack.append(binarized_image)

        # Initial plots
        wave_surf = ax_wave.plot_surface(X, Y, Z, cmap=cm.coolwarm)

        Zx_max_abs_value = max(abs(Zx.min()), abs(Zx.max()))
        
        zx_img = ax_zx.imshow(natural_Zx, cmap='hsv', vmin=-Zx_max_abs_value, vmax=Zx_max_abs_value)
        zy_img = ax_zy.imshow(Zx, cmap='hsv', vmin=-Zx_max_abs_value, vmax=Zx_max_abs_value)
        image_plot = ax_image.imshow(natural_Zx - Zx, cmap='hsv', vmin=-Zx_max_abs_value, vmax=Zx_max_abs_value)
        
        Zy_max_abs_value = max(abs(Zy.min()), abs(Zy.max()))
        x_displaced_image_plot = ax_x_displacement.imshow(natural_Zy, cmap='hsv', vmin=-Zy_max_abs_value, vmax=Zy_max_abs_value)
        y_displaced_image_plot = ax_y_displacement.imshow(Zy, cmap='hsv', vmin=-Zy_max_abs_value, vmax=Zy_max_abs_value)
        displaced_image_plot = ax_displaced.imshow(natural_Zy - Zy,cmap='hsv', vmin=-Zy_max_abs_value, vmax=Zy_max_abs_value)

        # ax_wave.set_title('3D Wave Surface')
        ax_zx.set_title('Wave Displacement (Zx)')
        ax_zy.set_title('Dilated Zx Displacement')
        ax_image.set_title('Wave Zx - Dilated Zx')
        ax_displaced.set_title('Wave Zy - Dilated Zy')
        ax_x_displacement.set_title('Wave Displacement (Zy)')
        ax_y_displacement.set_title('dilated Zy Displacement')

        ax_wave.set_visible(False)

        # Function to update the plots
        def update(frame):
            nonlocal displaced_image, x_displaced_image, y_displaced_image, image_plot # To ensure displaced_image is updated across frames
            Z = self.wave.calc_wave(self.H0, self.W, frame, self.Grid_Sign)
            Z = gaussian_filter1d(Z, sigma=50, axis=0)

            #get the frame in dispalce_images            
            displaced_mask = self.masks[f'arr_{self.frame_count}']
            self.frame_count += 1
            # print(f"arr_{int(frame)}")
            # displaced_mask = cv2.normalize(displaced_mask, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            # displaced_mask = cv2.cvtColor(displaced_mask, cv2.COLOR_RGB2GRAY)
            displaced_mask = displaced_mask[:,:,0]
            displaced_mask = displaced_mask.astype(np.float64)
            Zx, Zy = np.gradient(Z) * displaced_mask

            natural_Zx, natural_Zy = np.gradient(Z)

            Zx_disp = np.clip(Zx * 50, -20, 20).astype(np.float32) *4e7
            Zy_dsip = np.clip(Zy * 50, -20, 20).astype(np.float32) *4e7

            # Apply the displacements to the previously displaced image (cumulative effect)
            displaced_image = self.apply_displacement(displaced_image, Zx_disp, Zy_dsip)
            x_displaced_image = self.apply_displacement(x_displaced_image, Zx_disp, 0)
            y_displaced_image = self.apply_displacement(y_displaced_image, 0, Zy_dsip)

            binarized_image = np.where(displaced_image > 128, 1, 0)
            self.displaced_image_stack.append(binarized_image)            

            # Update the 3D wave plot
            ax_wave.cla()  # Remove previous surface to avoid plotting over it
            ax_wave.plot_surface(X, Y, Z, cmap=cm.coolwarm)
            ax_wave.set_title('3D Wave Surface')
            ax_wave.set_xlim(self.param['xLim'])
            ax_wave.set_ylim(self.param['yLim'])
            ax_wave.set_zlim(self.param['zLim'])

            ax_wave.set_visible(False)

            # Update the x and y displacement images
            zx_img.set_data(natural_Zx)
            zy_img.set_data(Zx)
            image_plot.set_data(natural_Zx - Zx)

            # Update the displaced image plot            
            displaced_image_plot.set_data(natural_Zy - Zy)
            x_displaced_image_plot.set_data(natural_Zy)
            y_displaced_image_plot.set_data(Zy)
            

            return wave_surf, zx_img, zy_img, displaced_image_plot


        # Animate the wave and displacements
        ani = FuncAnimation(fig, update, frames=np.linspace(0, 30, 30), blit=False, repeat=False)
        plt.tight_layout()
        #add padding between subplots
        plt.subplots_adjust(wspace=0.5, hspace=0.5)
        
        plt.show()

    

    def load_image(self):
        # Load the image array from the .npy file
        array = np.load(self.image_path)
        array = array[0]
        
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

# Initialize and run the plot with wave displacements
displaced_image = Apply_Displacement()
displaced_image.plot()