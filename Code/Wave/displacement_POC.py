import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.cm as cm
import cv2

from wave_genrator import Wave_Generator

logging.getLogger().setLevel(logging.INFO)  # Allow printing info level logs
os.chdir(os.path.dirname(__file__)) #change working directory to current directory



class Apply_Displacement:
    def __init__(self):
        self.image_path = "SheppLogan_Phantom.svg.png"
        self.image = None
        self.wave = Wave_Generator()
        self.param = self.wave.param
        self.H0, self.W, self.Grid_Sign = self.wave.initialize_wave()

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

        # Setup subplots
        
        ax_zx = fig.add_subplot(3, 4, 2)
        ax_zy = fig.add_subplot(3, 4, 3)
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
        Zx, Zy = np.gradient(Z)

        x_displacement = np.clip(Zx * 50, -20, 20).astype(np.float32)  # Scale by 50 for more visible effect
        y_displacement = np.clip(Zy * 50, -20, 20).astype(np.float32)
        displaced_image = self.apply_displacement(image, x_displacement, y_displacement)
        x_displaced_image = self.apply_displacement(image, x_displacement, 0)
        y_displaced_image = self.apply_displacement(image, 0, y_displacement)


        height, width, _ = image.shape
        x, y = np.meshgrid(np.arange(width), np.arange(height))
        slope_x = 1.0
        slope_y = 1.0

        # Case 1 -> nothing
        # slope_x_target = 1
        # slope_y_target = 1

        # Case 2 -> enlrage 
        slope_x_target = 2
        slope_y_target = 2
        
        # Case 3 -> shrink
        # slope_x_target = 0.5
        # slope_y_target = 0.5

        # Smoothing factor for gradual change
        smoothing_factor = 0.01

        # Gradually adjust the slope towards the target
        slope_x += (slope_x_target - slope_x) * smoothing_factor
        slope_y += (slope_y_target - slope_y) * smoothing_factor

        # Calculate displacement fields for x and y independently
        x_displacement = ((x - width / 2) / slope_x) - (x - width / 2)
        y_displacement = ((y - height / 2) / slope_y) - (y - height / 2)
        # Plot displacement for the middle row and middle column
        row_idx = height // 2  # Middle row for X displacement
        col_idx = width // 2  
        zx_img = ax_zx.plot(-x_displacement[row_idx, :])
        ax_zx.set_xlabel('X Coordinate')
        ax_zx.set_ylabel('Displacement')
        zy_img = ax_zy.plot(-y_displacement[:, col_idx])
        ax_zy.set_xlabel('Y Coordinate')
        ax_zy.set_ylabel('Displacement')


        image_plot = ax_image.imshow(image)
        ax_x_displacement.imshow(image)
        ax_y_displacement.imshow(image)
        displaced_image_plot = ax_displaced.imshow(displaced_image)
        x_displaced_image_plot = ax_x_displacement.imshow(x_displaced_image)
        y_displaced_image_plot = ax_y_displacement.imshow(y_displaced_image)

        ax_zx.set_title('X Displacement (Zx)')
        ax_zy.set_title('Y Displacement (Zy)')
        ax_image.set_title('Original Image')
        ax_displaced.set_title('Displaced Image')
        ax_x_displacement.set_title('X Displaced Image')
        ax_y_displacement.set_title('Y Displaced Image')

        # Function to update the plots
        def update(frame):

            # Apply the displacements to the previously displaced image (cumulative effect)
            nonlocal displaced_image, x_displaced_image, y_displaced_image  # To ensure displaced_image is updated across frames
            # displaced_image = self.apply_displacement(displaced_image, Zx, Zy)
            displaced_image = self.apply_displacement(displaced_image, x_displacement, y_displacement)
            x_displaced_image = self.apply_displacement(x_displaced_image, x_displacement, 0)
            y_displaced_image = self.apply_displacement(y_displaced_image, 0, y_displacement)


            # Update the x and y displacement images
            

            # Update the displaced image plot
            displaced_image_plot.set_data(displaced_image)
            x_displaced_image_plot.set_data(x_displaced_image)
            y_displaced_image_plot.set_data(y_displaced_image)

            return _, zx_img, zy_img, displaced_image_plot


        # Animate the wave and displacements
        ani = FuncAnimation(fig, update, frames=np.linspace(0, 100, 400), blit=False, repeat=False)
        plt.tight_layout()
        #add padding between subplots
        plt.subplots_adjust(wspace=0.5, hspace=0.5)
        
        plt.show()

    def load_image(self):
        self.image = cv2.imread(self.image_path)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        return self.image

# Initialize and run the plot with wave displacements
displaced_image = Apply_Displacement()
displaced_image.plot()
