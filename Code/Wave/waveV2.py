import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.cm as cm
import cv2
import random

# Default parameters
param = {
    'meshsize': 128,
    'patchsize': 200,
    'windSpeed': 100,
    'winddir': 90,
    'rng': 13,
    'A': 1e-7,
    'g': 9.81,
    'xLim': [-10, 10],
    'yLim': [-10, 10],
    'zLim': [-1e-4 * 2, 1e-4 * 2]
}

presetModes = {
    'names': ['Calm', 'Windy lake', 'Swell', 'T-Storm', 'Tsunami', 'Custom'],
    'winddir': [90, 90, 220, 90, 90],
    'patchsize': [150, 500, 180, 128, 128],
    'meshsize': [128, 480, 215, 128, 128]
}
def initialize_wave(param):
    np.random.seed(param['rng'])

    gridSize = (param['meshsize'], param['meshsize'])

    meshLim = np.pi * param['meshsize'] / param['patchsize']
    N = np.linspace(-meshLim, meshLim, param['meshsize'])
    M = np.linspace(-meshLim, meshLim, param['meshsize'])
    Kx, Ky = np.meshgrid(N, M)

    K = np.sqrt(Kx**2 + Ky**2)
    W = np.sqrt(K * param['g'])

    windx, windy = np.cos(np.radians(param['winddir'])), np.sin(np.radians(param['winddir']))

    P = phillips(Kx, Ky, [windx, windy], param['windSpeed'], param['A'], param['g'])
    H0 = 1 / np.sqrt(2) * (np.random.randn(*gridSize) + 1j * np.random.randn(*gridSize)) * np.sqrt(P)

    Grid_Sign = signGrid(param['meshsize'])

    return H0, W, Grid_Sign


def calc_wave(H0, W, time, Grid_Sign):
    wt = np.exp(1j * W * time)
    Ht = H0 * wt + np.conj(np.rot90(H0, 2)) * np.conj(wt)
    Z = np.real(np.fft.ifft2(Ht) * Grid_Sign)

    # Generate displacement maps
    displacement_x = np.gradient(Z, axis=1)
    displacement_y = np.gradient(Z, axis=0)

    return Z, displacement_x, displacement_y

def phillips(Kx, Ky, windDir, windSpeed, A, g):
    K_sq = Kx**2 + Ky**2
    L = windSpeed**2 / g
    k_norm = np.sqrt(K_sq)
    WK = Kx / k_norm * windDir[0] + Ky / k_norm * windDir[1]
    P = A / K_sq**2 * np.exp(-1.0 / (K_sq * L**2)) * WK**2
    P[K_sq == 0] = 0
    P[WK < 0] = 0
    return P

def signGrid(n):
    x, y = np.meshgrid(np.arange(n), np.arange(n))
    sgn = np.ones((n, n))
    sgn[(x + y) % 2 == 0] = -1
    return sgn


def apply_displacement(image, displacement_x, displacement_y, apply_x=True, apply_y=True):
    h, w = image.shape[:2]
    map_x, map_y = np.meshgrid(np.arange(w), np.arange(h))

    # print(f"Displacement X min: {np.min(displacement_x)}, max: {np.max(displacement_x)}")
    # print(f"Displacement Y min: {np.min(displacement_y)}, max: {np.max(displacement_y)}")

    # Scale up the displacement values to make them more noticeable
    displacement_x_resized = cv2.resize(displacement_x, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32) * 1e4
    displacement_y_resized = cv2.resize(displacement_y, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32) * 1e4

    # print(f"Displacement X resized min: {np.min(displacement_x_resized)}, max: {np.max(displacement_x_resized)}")
    # print(f"Displacement Y resized min: {np.min(displacement_y_resized)}, max: {np.max(displacement_y_resized)}")

    # Apply displacement
    if apply_x and apply_y:
        # Apply both x and y displacements
        map_x = (map_x + displacement_x_resized).astype(np.float32)
        map_y = (map_y + displacement_y_resized).astype(np.float32)
        warped_image = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    elif apply_x:
        # Only apply x displacement; leave y unchanged
        map_x = (map_x + displacement_x_resized).astype(np.float32)
        map_y = map_y.astype(np.float32)  # Identity mapping for y
        warped_image = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    elif apply_y:
        # Only apply y displacement; leave x unchanged
        map_x = map_x.astype(np.float32)  # Identity mapping for x
        map_y = (map_y + displacement_y_resized).astype(np.float32)
        warped_image = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    else:
        # No displacement, return the original image
        warped_image = image

    return warped_image

def init_gui(param, presetModes, image_path):
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    ax1, ax2, ax3, ax4, ax5, ax6 = axs.flatten()
    # ax1, ax2, ax3, ax4, ax5, ax6, ax7 = axs.flatten()

    x = np.linspace(param['xLim'][0], param['xLim'][1], param['meshsize'])
    y = np.linspace(param['yLim'][0], param['yLim'][1], param['meshsize'])
    X, Y = np.meshgrid(x, y)

    H0, W, Grid_Sign = initialize_wave(param)

    Z, displacement_x, displacement_y = calc_wave(H0, W, 0, Grid_Sign)

    ax1 = fig.add_subplot(231, projection='3d')
    surf = ax1.plot_surface(X, Y, Z, cmap=cm.coolwarm)

    # Set axis limits
    ax1.set_xlim(param['xLim'])
    ax1.set_ylim(param['yLim'])
    ax1.set_zlim(param['zLim'])

    # Add padding
    fig.tight_layout(pad=2.0)

    # Display the x displacement map
    ax2.imshow(displacement_x, cmap='gray')
    ax2.set_title('X Displacement')
    ax2.axis('off')

    # Display the y displacement map
    ax3.imshow(displacement_y, cmap='gray')
    ax3.set_title('Y Displacement')
    ax3.axis('off')

    # Load and display the original image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Image not found or failed to load at path: {image_path}")
        exit()

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    ax4.imshow(image_rgb)
    ax4.set_title('Original Image')
    ax4.axis('off')

    # Apply X displacement map to the image
    x_warped_image = apply_displacement(image, displacement_x, displacement_y, apply_x=True, apply_y=False)
    ax5.imshow(x_warped_image)
    ax5.set_title('Displacement X Warped Image')
    ax5.axis('off')

    # Apply Y displacement map to the image
    y_warped_image = apply_displacement(image, displacement_x, displacement_y, apply_x=False, apply_y=True)
    ax6.imshow(y_warped_image)
    ax6.set_title('Displacement Y Warped Image')
    ax6.axis('off')

    # # Apply both X and Y displacement maps to the image
    # x_y_warped_image = apply_displacement(image, displacement_x, displacement_y, apply_x=True, apply_y=True)
    # ax7.imshow(x_y_warped_image)
    # ax7.set_title('Displacement X + Y Warped Image')
    # ax7.axis('off')

    def update(frame):
        # Recalculate wave and displacements
        Z, displacement_x, displacement_y = calc_wave(H0, W, frame, Grid_Sign)
        ax1.clear()
        surf = ax1.plot_surface(X, Y, Z, cmap=cm.coolwarm)
        ax1.set_xlim(param['xLim'])
        ax1.set_ylim(param['yLim'])
        ax1.set_zlim(param['zLim'])

        # Update displacement maps
        ax2.imshow(displacement_x, cmap='gray')
        ax3.imshow(displacement_y, cmap='gray')


        # Apply both X and Y displacements to the image
        x_warped_image = apply_displacement(image, displacement_x, displacement_y, apply_x=True, apply_y=False)
        y_warped_image = apply_displacement(image, displacement_x, displacement_y, apply_x=False, apply_y=True)
        x_y_warped_image = apply_displacement(image, displacement_x, displacement_y, apply_x=True, apply_y=True)

        # Update plots
        ax5.imshow(x_warped_image)
        ax5.set_title('Displacement X Warped Image')

        ax6.imshow(y_warped_image)
        ax6.set_title('Displacement Y Warped Image')

        # ax7.imshow(x_y_warped_image)
        # ax7.set_title('Displacement X + Y Warped Image')

        return surf,

    ani = FuncAnimation(fig, update, frames=np.linspace(0, 10, 100), blit=False)
    plt.show()

init_gui(param, presetModes, 'SheppLogan_Phantom.svg.png')