import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.cm as cm
from scipy.ndimage import gaussian_filter1d
import cv2
from PIL import Image as im 
import time
np.random.seed(int(time.time()))  # Dynamically set seed based on current time
import matplotlib.colors as mcolors

from wave_genrator import Wave_Generator
from strain_validation import limit_strain_range, plot_strain_results

logging.getLogger().setLevel(logging.INFO)  # Allow printing info level logs
os.chdir(os.path.dirname(__file__)) #change working directory to current directory

DISPLACMENET_MULTIPLAYER = 5e7
GAUSSIAN_SIGMA = 50

def save_if_not_exists(file_paths):
    """Check if any of the files exist"""
    for path in file_paths:
        if os.path.exists(path + '.npy'):
            return False
    return True


class Wave_Displacer:
    def __init__(self, path, save_mode=False):
        self.image_path = path
        self.image = None
        self.wave = Wave_Generator()
        self.param = self.wave.param
        self.H0, self.W, self.Grid_Sign = self.wave.initialize_wave()
        self.displaced_image_stack = []
        self.iterator = 0
        self.mask = None
        self.save_mode = save_mode
        self.frame_1 = None
        self.frame_2 = None
        self.displacement_x = None
        self.displacement_y = None
        self.frame_count = 0
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

    def plot(self):
        fig = plt.figure(figsize=(20, 10))
        self.displaced_image_stack = []

        # Setup subplots
        ax_wave = fig.add_subplot(2, 5, 4, projection='3d')  # 3D wave plot
        ax_wave.set_facecolor("#eeeeef")
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

        # self.masks = np.load('displaced_images/displaced_images.npz')
        self.masks = np.load('dilated_masks/dilated_masks.npz')
        displaced_mask = self.masks['arr_0']
        displaced_mask = displaced_mask[:,:,0]
        displaced_mask = displaced_mask.astype(np.float64)
        Zx, Zy = np.gradient(Z) * displaced_mask
        binarized_image = np.where(image > 128, 1, 0)
        self.displaced_image_stack.append(binarized_image)
        x_displacement = np.clip(Zx * 50, -20, 20).astype(np.float32)
        y_displacement = np.clip(Zy * 50, -20, 20).astype(np.float32)
        x_displacement = y_displacement = 0
        displaced_image = self.apply_displacement(image, x_displacement, y_displacement)
        x_displaced_image = self.apply_displacement(image, x_displacement, 0)
        y_displaced_image = self.apply_displacement(image, 0, y_displacement)
        binarized_image = np.where(displaced_image > 128, 1, 0)
        self.displaced_image_stack.append(binarized_image)

        # Initial plots
        wave_surf = ax_wave.plot_surface(X, Y, Z, cmap=cm.coolwarm) 
        vmin = min(Zx.min(), Zy.min())
        vmax = max(Zx.max(), Zy.max())
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        zx_img = ax_zx.imshow(Zx, cmap='RdBu', norm=norm)
        zy_img = ax_zy.imshow(Zy, cmap='RdBu', norm=norm)
        image_plot = ax_image.imshow(image)
        ax_x_displacement.imshow(image)
        ax_y_displacement.imshow(image)
        displaced_image_plot = ax_displaced.imshow(displaced_image,cmap='viridis')
        x_displaced_image_plot = ax_x_displacement.imshow(x_displaced_image)
        y_displaced_image_plot = ax_y_displacement.imshow(y_displaced_image)

        ax_wave.set_title('3D Wave Surface', pad= 10)
        ax_wave.set_axis_off()
        ax_zx.set_title('X Displacement (Zx)', pad= 10)
        ax_zx.set_axis_off()
        ax_zy.set_title('Y Displacement (Zy)', pad= 10)
        ax_zy.set_axis_off()
        ax_image.set_title('Original Image', pad= 10)
        ax_image.set_axis_off()
        ax_displaced.set_title('Displaced Image', pad= 10)
        ax_displaced.set_axis_off()
        ax_x_displacement.set_title('X Displaced Image', pad= 10)
        ax_x_displacement.set_axis_off()
        ax_y_displacement.set_title('Y Displaced Image', pad= 10)
        ax_y_displacement.set_axis_off()
        fig.colorbar(zx_img, ax=ax_zx, shrink=0.4)
        fig.colorbar(zy_img, ax=ax_zy, shrink=0.4)

        # Function to update the plots
        def update(frame):
            nonlocal displaced_image, x_displaced_image, y_displaced_image # To ensure displaced_image is updated across frames
            Z = self.wave.calc_wave(self.H0, self.W, frame, self.Grid_Sign)
            Z = gaussian_filter1d(Z, sigma=GAUSSIAN_SIGMA, axis=0)

            #get the frame in dispalce_images 
            displaced_mask = self.masks[f'arr_{self.frame_count}']
            self.frame_count += 1
            if self.frame_count > 30:
                self.finished = True
            displaced_mask = displaced_mask[:,:,0]
            displaced_mask = displaced_mask.astype(np.float64)
            
            Zx, Zy = np.gradient(Z)
            Zx_disp = np.clip(Zx * 50, -20, 20).astype(np.float32) *DISPLACMENET_MULTIPLAYER
            Zy_dsip = np.clip(Zy * 50, -20, 20).astype(np.float32) *DISPLACMENET_MULTIPLAYER
            Zx_disp = Zx_disp * displaced_mask
            Zy_dsip = Zy_dsip * displaced_mask
            
            # Limit the strain range
            result = limit_strain_range(Zx_disp, Zy_dsip, stretch=False, strain_upper_bound=0.4)
            Zx_disp, Zy_dsip, initial_strain, final_strain, max_initial, max_final, min_initial, min_final = result

            if(self.plot_strain):
                plot_strain_results(
                    initial_strain, final_strain, 
                    min_initial, max_initial, 
                    min_final, max_final,
                    strain_lower_bound = 0, strain_upper_bound = 0.1
                    )
                self.plot_strain = False


            # Zx_disp = Zx_disp * displaced_mask
            # Zy_dsip = Zy_dsip * displaced_mask

            # Apply the displacements to the previously displaced image (cumulative effect)
            self.frame_1 = displaced_image
            displaced_image = self.apply_displacement(displaced_image, Zx_disp, Zy_dsip)
            x_displaced_image = self.apply_displacement(x_displaced_image, Zx_disp, 0)
            y_displaced_image = self.apply_displacement(y_displaced_image, 0, Zy_dsip)
            self.frame_2 = displaced_image
            self.displacement_x = Zx_disp
            self.displacement_y = Zy_dsip
            binarized_image = np.where(displaced_image > 128, 1, 0)
            self.displaced_image_stack.append(binarized_image)            

            # Update the 3D wave plot
            ax_wave.cla()  # Remove previous surface to avoid plotting over it
            ax_wave.plot_surface(X, Y, Z, cmap=cm.coolwarm)
            ax_wave.set_title('3D Wave Surface')
            ax_wave.set_xlim(self.param['xLim'])
            ax_wave.set_ylim(self.param['yLim'])
            ax_wave.set_zlim(self.param['zLim'])
            ax_wave.set_axis_off()

            # Update the x and y displacement images
            zx_img.set_data(Zx * displaced_mask)
            zy_img.set_data(Zy * displaced_mask)

            # Update the displaced image plot            
            displaced_image_plot.set_data(displaced_image)
            x_displaced_image_plot.set_data(x_displaced_image)
            y_displaced_image_plot.set_data(y_displaced_image)
            
            if self.save_mode:
                #generate random probability to save the frame
               if np.random.rand() > 0.5:
                base_name = os.path.basename(self.image_path)
                base_name = os.path.splitext(base_name)[0]
                
                # Create file paths
                frame1_path = f"Saved_test/Frames/{base_name}_#{self.frame_count}_1"
                frame2_path = f"Saved_test/Frames/{base_name}_#{self.frame_count}_2"
                disp_x_path = f"Saved_test/Displacements/{base_name}_#{self.frame_count}_x"
                disp_y_path = f"Saved_test/Displacements/{base_name}_#{self.frame_count}_y"
                
                # Check if any of the files exist
                if save_if_not_exists([frame1_path, frame2_path, disp_x_path, disp_y_path]):
                    # Save all files if none exist
                    np.save(frame1_path, self.frame_1)
                    np.save(frame2_path, self.frame_2)
                    np.save(disp_x_path, self.displacement_x)
                    np.save(disp_y_path, self.displacement_y)
                    print(f"Successfully saved files for {base_name}_#{self.frame_count}")
                else:
                    print(f"Skipped saving: One or more files already exist for {base_name}_#{self.frame_count}")
            return wave_surf, zx_img, zy_img, displaced_image_plot


        # Animate the wave and displacements
        ani = FuncAnimation(fig, update, frames=np.linspace(0, 30, 30), blit=False, repeat=False)
        plt.tight_layout()
        #add padding between subplots
        plt.subplots_adjust(wspace=0.5, hspace=0.5)
        #check if animation is finished first

        if self.save_mode:
            #check if file exists
            trash_path = "trash/wave_displacement.mp4"
            if os.path.exists(trash_path):
                os.remove(trash_path)
            ani.save(trash_path, writer='ffmpeg')            
        else:
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
    def check_status(self):
        if self.finished:
            print("Applied displacement to all frames")
            return True
            
        else:
            return False
        

displacer = Wave_Displacer(path="/Users/osama/GP-2025-Strain/Code/Wave/Images/patient001_frame01_slice_5_ACDC.npy")
displacer.plot()