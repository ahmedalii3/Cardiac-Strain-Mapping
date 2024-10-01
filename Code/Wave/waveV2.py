import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.cm as cm
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
    return Z

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


def init_gui(param, presetModes):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    x = np.linspace(param['xLim'][0], param['xLim'][1], param['meshsize'])
    y = np.linspace(param['yLim'][0], param['yLim'][1], param['meshsize'])
    X, Y = np.meshgrid(x, y)

    H0, W, Grid_Sign = initialize_wave(param)

    Z = calc_wave(H0, W, 0, Grid_Sign)

    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm)

    # Set axis limits
    ax.set_xlim(param['xLim'])
    ax.set_ylim(param['yLim'])
    ax.set_zlim(param['zLim'])

    def update(frame):
        Z = calc_wave(H0, W, frame, Grid_Sign)
        ax.clear()
        ax.plot_surface(X, Y, Z, cmap=cm.coolwarm)
        # Set axis limits again after clearing
        ax.set_xlim(param['xLim'])
        ax.set_ylim(param['yLim'])
        ax.set_zlim(param['zLim'])
        return surf,
    

    ani = FuncAnimation(fig, update, frames=np.linspace(0, 10, 100), blit=False)
    plt.show()

init_gui(param, presetModes)