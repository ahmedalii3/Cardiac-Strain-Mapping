import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D

# Base wave parameters
num_waves = 5  # Number of wave components
amplitude_range = [0.5, 1.5]  # Range of amplitudes for the waves
wavelength_range = [4.0, 8.0]  # Range of wavelengths
speed_range = [0.5, 1.5]  # Range of speeds
direction_range = [0, np.pi]  # Range of directions (0 to 180 degrees)
phase_range = [0, 2 * np.pi]  # Random phase shifts

# Randomly generate wave properties
amplitudes = np.random.uniform(*amplitude_range, num_waves)
wavelengths = np.random.uniform(*wavelength_range, num_waves)
speeds = np.random.uniform(*speed_range, num_waves)
directions = np.random.uniform(*direction_range, num_waves)
phases = np.random.uniform(*phase_range, num_waves)
frequencies = speeds / wavelengths  # Frequencies of the waves
wave_numbers = 2 * np.pi / wavelengths  # Wave numbers (k)

# Create a grid
x = np.linspace(-10, 10, 100)
y = np.linspace(-10, 10, 100)
X, Y = np.meshgrid(x, y)

# Function to calculate the combined wave height with multiple wave components
def wave_height(t):
    Z = np.zeros_like(X)
    for i in range(num_waves):
        # Calculate contribution from each wave
        Z += amplitudes[i] * np.cos(
            wave_numbers[i] * (X * np.cos(directions[i]) + Y * np.sin(directions[i])) 
            - frequencies[i] * t + phases[i]
        )
    return Z

# Create a figure for the 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Set up 3D plot limits and labels
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)
ax.set_zlim(-3, 3)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Wave Height')

# Initial wave surface plot
Z = wave_height(0)
surface = ax.plot_surface(X, Y, Z, cmap='Blues', rstride=1, cstride=1, linewidth=0, antialiased=True)

# Animation function to update the 3D surface
def animate(t):
    Z = wave_height(t)
    ax.clear()  # Clear the previous surface
    ax.set_xlim(-10, 10)  # Reset limits
    ax.set_ylim(-10, 10)
    ax.set_zlim(-3, 3)
    ax.set_xlabel('X')  # Reset labels
    ax.set_ylabel('Y')
    ax.set_zlabel('Wave Height')
    surface = ax.plot_surface(X, Y, Z, cmap='Blues', rstride=1, cstride=1, linewidth=0, antialiased=True)
    return surface,

# Create the animation
anim = animation.FuncAnimation(fig, animate, frames=200, interval=50)

# Display the animation
plt.show()