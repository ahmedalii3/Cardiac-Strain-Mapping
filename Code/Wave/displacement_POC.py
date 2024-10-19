import os
import logging
from matplotlib import cm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import cv2

logging.getLogger().setLevel(logging.INFO) # Allow printing info level logs
os.chdir(os.path.dirname(__file__)) #change working directory to current directory


class wave_POC:
    def __init__(self):
        self.image_path  = "SheppLogan_Phantom.svg.png"
        self.image = None
    def apply_displacement(self, image, x_displacement, y_displacement):
        displaced_image = np.roll(image, shift=(y_displacement, x_displacement), axis=(0, 1))
        return displaced_image
    
    def plot_wave_POC(self):
        fig = plt.figure(figsize=(20, 10))
        ax1 = fig.add_subplot(1, 5, 1)  # Original Image
        ax3 = fig.add_subplot(1, 5, 2)
        ax4 = fig.add_subplot(1, 5, 3)
        x = np.linspace(-10, 10, 220)
        y = np.linspace(-10, 10, 220)
        X, Y = np.meshgrid(x, y)
        ax1.imshow(self.load_image())
        ax3.imshow(self.load_image())
        ax4.imshow(self.load_image())

        def update(frame):
            Z = self.load_image()
            x_displacement = frame*2
            y_displacement = frame*2

            # Apply the displacement to the image
            new_image = self.apply_displacement(Z, x_displacement, 0)

            ax3.cla()
            ax3.imshow(new_image)

            new_image2 = self.apply_displacement(Z, 0, y_displacement)
            ax4.cla()
            ax4.imshow(new_image2)
            return frame
        

        ani = FuncAnimation(fig, update, frames=5, blit=False)
        plt.show()
       

    def load_image(self):
        self.image = cv2.imread(self.image_path)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        return self.image
  


wave_POC = wave_POC()
wave_POC.plot_wave_POC()