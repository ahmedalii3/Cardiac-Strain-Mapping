import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# Function to load and prepare the image
def load_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    return image

# Updated Gerstner wave parameters
wave_params = {
    'amplitude': 5,
    'wavelength': 50,
    'speed': 1,
    'direction_x': np.radians(45),  # Wave moves at 45 degrees for X
    'direction_y': np.radians(135), # Wave moves at 135 degrees for Y (phase shift)
}

# Gerstner wave function to generate distinct displacement maps
def generate_gerstner_wave_displacement(width, height, frame, params):
    A = params['amplitude']
    L = params['wavelength']
    speed = params['speed']
    
    direction_x = params['direction_x']
    direction_y = params['direction_y']

    x, y = np.meshgrid(np.arange(width), np.arange(height))
    
    # Time-dependent phase for X and Y with distinct directions
    phase_x = (2 * np.pi / L) * (x * np.cos(direction_x) + y * np.sin(direction_x)) - (speed * frame)
    phase_y = (2 * np.pi / L) * (x * np.cos(direction_y) + y * np.sin(direction_y)) - (speed * frame)
    
    displacement_x = A * np.cos(phase_x)
    displacement_y = A * np.sin(phase_y)

    # Calculate the wave surface Z based on the displacement
    Z = np.sqrt(displacement_x**2 + displacement_y**2)  # Create a wave height map
    return displacement_x, displacement_y, Z

# Function to apply the displacement map to the image
def apply_displacement(image, x_displacement, y_displacement):
    height, width, _ = image.shape
    x, y = np.meshgrid(np.arange(width), np.arange(height))

    # Compute new coordinates based on displacement
    x_new = (x + x_displacement).astype(np.float32)
    y_new = (y + y_displacement).astype(np.float32)

    # Warp the image using the displacement map
    displaced_image = cv2.remap(image, x_new, y_new, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return displaced_image

# Function to normalize displacement maps for display
def normalize_map(disp_map):
    normalized = (disp_map - np.min(disp_map)) / (np.max(disp_map) - np.min(disp_map))
    return normalized

# Function to create and display an animation of the displacement effect
def animate_displacement(image_path, frames=10, interval=200):
    # Load the image
    image = load_image(image_path)
    height, width, _ = image.shape

    fig = plt.figure(figsize=(20, 10))
    
    # Create subplots
    ax1 = fig.add_subplot(1, 5, 1)  # Original Image
    ax2 = fig.add_subplot(1, 5, 2)  # Displaced Image
    ax3 = fig.add_subplot(1, 5, 3)  # X Displacement Map
    ax4 = fig.add_subplot(1, 5, 4)  # Y Displacement Map
    ax5 = fig.add_subplot(1, 5, 5, projection='3d')  # 3D Wave Surface

    # Display original image
    ax1.imshow(image)
    ax1.set_title("Original Image")
    ax1.axis('off')

    # Prepare placeholders for the animation
    displaced_im = ax2.imshow(image)
    ax2.set_title("Displaced Image")
    ax2.axis('off')

    x_disp_im = ax3.imshow(np.zeros_like(image[:, :, 0]), cmap='viridis', vmin=0, vmax=1)
    ax3.set_title("X Displacement Map")
    ax3.axis('off')

    y_disp_im = ax4.imshow(np.zeros_like(image[:, :, 0]), cmap='viridis', vmin=0, vmax=1)
    ax4.set_title("Y Displacement Map")
    ax4.axis('off')

    # Set up 3D plot
    x = np.linspace(0, width, width)
    y = np.linspace(0, height, height)
    X, Y = np.meshgrid(x, y)
    wave_surface = ax5.plot_surface(X, Y, np.zeros((height, width)), cmap='Blues', edgecolor='none')
    ax5.set_title("Wave Surface")
    ax5.set_xlabel('X')
    ax5.set_ylabel('Y')
    ax5.set_zlabel('Z (Height)')
    ax5.set_xlim(0, width)
    ax5.set_ylim(0, height)

    # Update function for each frame of the animation
    def update(frame):
        # Generate displacement maps using the Gerstner wave model
        x_displacement, y_displacement, Z = generate_gerstner_wave_displacement(width, height, frame, wave_params)

        # Apply displacement to the image
        displaced_image = apply_displacement(image, x_displacement, y_displacement)

        # Normalize and update the displacement maps
        x_disp_im.set_array(normalize_map(x_displacement))
        y_disp_im.set_array(normalize_map(y_displacement))

        # Update the displaced image in the animation
        displaced_im.set_array(displaced_image)

        # Update the wave surface in 3D
        ax5.cla()  # Clear the axis
        ax5.plot_surface(X, Y, Z, cmap='Blues', edgecolor='none')

        return [displaced_im, x_disp_im, y_disp_im]

    # Create an animation with 'frames' number of steps and display interval
    anim = FuncAnimation(fig, update, frames=np.arange(0, frames), interval=interval, blit=True)
    
    plt.show()

# Run the animation with the updated Gerstner wave displacement
animate_displacement('SheppLogan_Phantom.svg.png', frames=10, interval=200)
