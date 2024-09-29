import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

# Wave Parameters
amplitude = 1.0
wavelength = 5.0
speed = 1.0
direction = np.pi / 4  # 45 degrees
frequency = speed / wavelength
k = 2 * np.pi / wavelength  # wave number

# Create a grid
x = np.linspace(-10, 10, 100)
y = np.linspace(-10, 10, 100)
X, Y = np.meshgrid(x, y)

# Function to calculate wave height (Gerstner wave)
def wave_height(t):
    return amplitude * np.cos(k * (X * np.cos(direction) + Y * np.sin(direction)) - frequency * t)

# Animation function
fig, ax = plt.subplots()
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)
line = ax.imshow(wave_height(0), extent=[-10, 10, -10, 10], origin='lower', cmap='Blues')

def animate(t):
    Z = wave_height(t)
    line.set_array(Z)
    return line,

anim = animation.FuncAnimation(fig, animate, frames=200, interval=50)

plt.show()
