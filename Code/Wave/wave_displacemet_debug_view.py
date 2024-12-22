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
        height, width= image.shape[:2]
        x, y = np.meshgrid(np.arange(width), np.arange(height))

        # Apply displacement (scale the displacements for more visible effect)
        x_new = (x + x_displacement).astype(np.float32)
        y_new = (y + y_displacement).astype(np.float32)

        # Warp the image using remap for both x and y displacements
        displaced_image = cv2.remap(image, x_new, y_new, interpolation=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT)
        return displaced_image

    def plot(self):
        fig = plt.figure(figsize=(60,42))  # Adjust the figure size to accommodate the larger grid
        self.displaced_image_stack = []

        # Setup subplots in a 6x4 grid
        ax_image = fig.add_subplot(6, 4, 5)            # Row 1, Col 1
        ax_displaced = fig.add_subplot(6, 4, 6)        # Row 1, Col 2
        ax_x_displacement = fig.add_subplot(6, 4, 7)   # Row 1, Col 3
        ax_y_displacement = fig.add_subplot(6, 4, 8)   # Row 1, Col 4

        ax_wave = fig.add_subplot(6, 4, 4, projection='3d')  # 3D wave plot (Row 2, Col 1)
        ax_zx = fig.add_subplot(6, 4, 2)              # Row 2, Col 2
        ax_zy = fig.add_subplot(6, 4, 3)              # Row 2, Col 3

        # Add other plots to fill the grid
        # ax_extra1 = fig.add_subplot(6, 4, 8)          # Row 2, Col 4
        ax_extra2 = fig.add_subplot(6, 4, 9)          # Row 3, Col 1
        ax_extra3 = fig.add_subplot(6, 4, 10)         # Row 3, Col 2
        ax_extra4 = fig.add_subplot(6, 4, 11)         # Row 3, Col 3
        ax_extra5 = fig.add_subplot(6, 4, 12)         # Row 3, Col 4
        ax_extra6 = fig.add_subplot(6, 4, 13)         # Row 4, Col 1
        ax_extra7 = fig.add_subplot(6, 4, 14)         # Row 4, Col 2
        ax_extra8 = fig.add_subplot(6, 4, 15)         # Row 4, Col 3
        ax_extra9 = fig.add_subplot(6, 4, 16)         # Row 4, Col 4
        # ax_extra10 = fig.add_subplot(6, 4, 17)        # Row 5, Col 1
        ax_extra11 = fig.add_subplot(6, 4, 18)        # Row 5, Col 2
        ax_extra12 = fig.add_subplot(6, 4, 19)        # Row 5, Col 3
        # ax_extra13 = fig.add_subplot(6, 4, 20)        # Row 5, Col 4
        ax_extra14 = fig.add_subplot(6, 4, 21)        # Row 6, Col 1
        ax_extra15 = fig.add_subplot(6, 4, 22)        # Row 6, Col 2
        ax_extra16 = fig.add_subplot(6, 4, 23)        # Row 6, Col 3
        ax_extra17 = fig.add_subplot(6, 4, 24)        # Row 6, Col 4



    


        # Load the initial image
        image = self.load_image()

        # Initialize the wave grid
        x = np.linspace(self.param['xLim'][0], self.param['xLim'][1], self.param['meshsize'])
        y = np.linspace(self.param['yLim'][0], self.param['yLim'][1], self.param['meshsize'])
        X, Y = np.meshgrid(x, y)

        # Plot the initial data
        Z = self.wave.calc_wave(self.H0, self.W, 0, self.Grid_Sign)
        Z = gaussian_filter1d(Z, sigma=50, axis=0)

        self.masks = np.load('dilated_masks/dilated_masks.npz')
        displaced_mask = self.masks['arr_0']
        displaced_mask = cv2.normalize(displaced_mask, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        displaced_mask = cv2.cvtColor(displaced_mask, cv2.COLOR_RGB2GRAY)
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
        zx_img = ax_zx.imshow(natural_Zx, cmap='coolwarm')
        zy_img = ax_zy.imshow(natural_Zy, cmap='coolwarm')
        image_plot = ax_image.imshow(image)
        ax_x_displacement.imshow(image)
        ax_y_displacement.imshow(image)
        displaced_image_plot = ax_displaced.imshow(displaced_image,cmap='viridis')
        x_displaced_image_plot = ax_x_displacement.imshow(x_displaced_image)
        y_displaced_image_plot = ax_y_displacement.imshow(y_displaced_image)

        # ax_wave.set_title('3D Wave Surface')
        # ax_zx.set_title('X Displacement (Zx)')
        # ax_zy.set_title('Y Displacement (Zy)')
        # ax_image.set_title('Original Image')
        # ax_displaced.set_title('Displaced Image')
        # ax_x_displacement.set_title('X Displaced Image')
        # ax_y_displacement.set_title('Y Displaced Image')


        # image_plot = ax_extra1.imshow(image)
        mask_display = np.load("displaced_images/displaced_images.npz")['arr_0']
        mask_display = mask_display.astype(np.float64)
        
        mask_display_x = mask_display_y = mask_display
        
        original_mask_plot = ax_extra5.imshow(mask_display)
        original_mask_plot_x = ax_extra3.imshow(mask_display_x)
        original_mask_plot_y = ax_extra4.imshow(mask_display_y)
        image_plot = ax_extra2.imshow(mask_display)

        dilated_mask_display = displaced_mask
        dilated_mask_display_x = dilated_mask_display_y = dilated_mask_display
        dilated_mask_plot = ax_extra9.imshow(dilated_mask_display, cmap='grey')
        dilated_mask_plot_x = ax_extra7.imshow(dilated_mask_display_x, cmap='grey')
        dilated_mask_plot_y = ax_extra8.imshow(dilated_mask_display_y, cmap='grey')
        image_plot = ax_extra6.imshow(dilated_mask_display, cmap='grey')


        # image_plot = ax_extra10.imshow(image)
        zx_plot = ax_extra11.imshow(Zx, cmap='coolwarm')
        zy_plot = ax_extra12.imshow(Zy, cmap='coolwarm')
        # image_plot = ax_extra13.imshow(image)

        image_plot = ax_extra14.imshow(image)
        final_image = final_image_x = final_image_y = image
        final_plot = ax_extra15.imshow(image)
        final_plot_x = ax_extra16.imshow(image)
        finak_plot_y = ax_extra17.imshow(image)

        ax_wave.set_axis_off()
        
        ax_zx.set_axis_off()
        ax_zy.set_axis_off()
        ax_image.set_axis_off()
        ax_displaced.set_axis_off()
        ax_x_displacement.set_axis_off()
        ax_y_displacement.set_axis_off()
        # ax_extra1.set_axis_off()
        ax_extra2.set_axis_off()
        ax_extra3.set_axis_off()
        ax_extra4.set_axis_off()
        ax_extra5.set_axis_off()
        ax_extra6.set_axis_off()
        ax_extra7.set_axis_off()
        ax_extra8.set_axis_off()
        ax_extra9.set_axis_off()
        # ax_extra10.set_axis_off()
        ax_extra11.set_axis_off()
        ax_extra12.set_axis_off()
        # ax_extra13.set_axis_off()
        ax_extra14.set_axis_off()
        ax_extra15.set_axis_off()
        ax_extra16.set_axis_off()
        ax_extra17.set_axis_off()

        # Function to update the plots
        def update(frame):
            nonlocal displaced_image, x_displaced_image, y_displaced_image, mask_display, mask_display_x
            nonlocal mask_display_y, dilated_mask_display, dilated_mask_display_x, dilated_mask_display_y
            nonlocal zx_plot, zy_plot, final_plot, final_plot_x, finak_plot_y, final_image, final_image_x, final_image_y # To ensure displaced_image is updated across frames
            Z = self.wave.calc_wave(self.H0, self.W, frame, self.Grid_Sign)
            Z = gaussian_filter1d(Z, sigma=50, axis=0)

            #get the frame in dispalce_images            
            displaced_mask = self.masks[f'arr_{self.frame_count}']
            self.frame_count += 1
            # print(f"arr_{int(frame)}")
            displaced_mask = cv2.normalize(displaced_mask, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            displaced_mask = cv2.cvtColor(displaced_mask, cv2.COLOR_RGB2GRAY)
            Zx, Zy = np.gradient(Z) * displaced_mask

            natural_Zx, natural_Zy = np.gradient(Z)

            Zx_disp = np.clip(Zx * 50, -20, 20).astype(np.float32) *1e5
            Zy_dsip = np.clip(Zy * 50, -20, 20).astype(np.float32) *1e5

            natural_Zx_disp = np.clip(natural_Zx * 50, -20, 20).astype(np.float32) *17500000
            natural_Zy_dsip = np.clip(natural_Zy * 50, -20, 20).astype(np.float32) *17500000

            # Apply the displacements to the previously displaced image (cumulative effect)
            displaced_image = self.apply_displacement(displaced_image, natural_Zx_disp, natural_Zy_dsip)
            x_displaced_image = self.apply_displacement(x_displaced_image, natural_Zx_disp, 0)
            y_displaced_image = self.apply_displacement(y_displaced_image, 0, natural_Zy_dsip)

            mask_display = self.apply_displacement(mask_display, natural_Zx_disp, natural_Zy_dsip)
            mask_display_x = self.apply_displacement(mask_display, natural_Zx_disp, 0)
            mask_display_y = self.apply_displacement(mask_display, 0, natural_Zy_dsip)

            dilated_mask_display = self.apply_displacement(dilated_mask_display, Zx_disp, Zy_dsip)
            dilated_mask_display_x = self.apply_displacement(dilated_mask_display, Zx_disp, 0)
            dilated_mask_display_y = self.apply_displacement(dilated_mask_display, 0, Zy_dsip)

            final_image = self.apply_displacement(final_image, Zx_disp, Zy_dsip)
            final_image_x = self.apply_displacement(final_image, Zx_disp, 0)
            final_image_y = self.apply_displacement(final_image, 0, Zy_dsip)



            # binarized_image = np.where(displaced_image > 128, 1, 0)
            # self.displaced_image_stack.append(binarized_image)            

            # Update the 3D wave plot
            ax_wave.cla()  # Remove previous surface to avoid plotting over it
            ax_wave.plot_surface(X, Y, Z, cmap=cm.coolwarm)
            # ax_wave.set_title('3D Wave Surface')
            ax_wave.set_xlim(self.param['xLim'])
            ax_wave.set_ylim(self.param['yLim'])
            ax_wave.set_zlim(self.param['zLim'])
            ax_wave.set_axis_off()

            # Update the x and y displacement images
            zx_plot.set_data(Zx)
            zy_plot.set_data(Zy)
            zx_img.set_data(natural_Zx)
            zy_img.set_data(natural_Zy)

            # Update the displaced image plot            
            displaced_image_plot.set_data(displaced_image)
            x_displaced_image_plot.set_data(x_displaced_image)
            y_displaced_image_plot.set_data(y_displaced_image)

            
            original_mask_plot.set_data(mask_display)
            original_mask_plot_x.set_data(mask_display_x)
            original_mask_plot_y.set_data(mask_display_y)
            
            dilated_mask_plot.set_data(dilated_mask_display)
            dilated_mask_plot_x.set_data(dilated_mask_display_x)
            dilated_mask_plot_y.set_data(dilated_mask_display_y)

            final_plot.set_data(final_image)
            final_plot_x.set_data(final_image_x)
            finak_plot_y.set_data(final_image_y)
            
            

            return wave_surf, zx_img, zy_img, displaced_image_plot


        # Animate the wave and displacements
        ani = FuncAnimation(fig, update, frames=np.linspace(0, 30, 30), blit=False, repeat=True)
        plt.tight_layout()
        #add padding between subplots
        # plt.subplots_adjust(wspace=0.5, hspace=0.5)
        mng = plt.get_current_fig_manager()
        mng.full_screen_toggle()
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