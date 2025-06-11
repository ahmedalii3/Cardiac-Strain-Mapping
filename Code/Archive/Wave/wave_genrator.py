import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.cm as cm

class Wave_Generator():
    def __init__(self):
        
        # Default self.parameters
        self.param = {
            'meshsize': 512,
            'patchsize': 50,
            'windSpeed': 50,
            'winddir': 45,
            'rng': 13,
            'A': 1e-7,
            'g': 9.81,
            'xLim': [-10, 10],
            'yLim': [-10, 10],
            'zLim': [-1e-5, 1e-5]
        }

        self.presetModes = {
            'names': ['Calm', 'Windy lake', 'Swell', 'T-Storm', 'Tsunami', 'Custom'],
            'winddir': [90, 90, 220, 90, 90],
            'patchsize': [150, 500, 180, 128, 128],
            'meshsize': [128, 480, 215, 128, 128]
        }
    def initialize_wave(self):
        np.random.seed(self.param['rng'])

        gridSize = (self.param['meshsize'], self.param['meshsize'])

        meshLim = np.pi * self.param['meshsize'] / self.param['patchsize']
        N = np.linspace(-meshLim, meshLim, self.param['meshsize'])
        M = np.linspace(-meshLim, meshLim, self.param['meshsize'])
        Kx, Ky = np.meshgrid(N, M)

        K = np.sqrt(Kx**2 + Ky**2)
        W = np.sqrt(K * self.param['g'])

        windx, windy = np.cos(np.radians(self.param['winddir'])), np.sin(np.radians(self.param['winddir']))

        P = self.phillips( Kx, Ky, [windx, windy], self.param['windSpeed'], self.param['A'], self.param['g'])
        H0 = 1 / np.sqrt(2) * (np.random.randn(*gridSize) + 1j * np.random.randn(*gridSize)) * np.sqrt(P)

        Grid_Sign = self.signGrid(self.param['meshsize'])

        return H0, W, Grid_Sign

    def calc_wave(self,H0, W, time, Grid_Sign):
        wt = np.exp(1j * W * time)
        Ht = H0 * wt + np.conj(np.rot90(H0, 2)) * np.conj(wt)
        Z = np.real(np.fft.ifft2(Ht) * Grid_Sign)
        return Z

    def phillips(self, Kx, Ky, windDir, windSpeed, A, g):
        K_sq = Kx**2 + Ky**2
        L = windSpeed**2 / g
        k_norm = np.sqrt(K_sq)
        WK = Kx / k_norm * windDir[0] + Ky / k_norm * windDir[1]
        P = A / K_sq**2 * np.exp(-1.0 / (K_sq * L**2)) * WK**2
        P[K_sq == 0] = 0
        P[WK < 0] = 0
        return P

    def signGrid(self, n):
        x, y = np.meshgrid(np.arange(n), np.arange(n))
        sgn = np.ones((n, n))
        sgn[(x + y) % 2 == 0] = -1
        return sgn


    def init_gui(self):
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        fig.patch.set_facecolor('#eeeeef')
        x = np.linspace(self.param['xLim'][0], self.param['xLim'][1], self.param['meshsize'])
        y = np.linspace(self.param['yLim'][0], self.param['yLim'][1], self.param['meshsize'])
        X, Y = np.meshgrid(x, y)
        ax.set_axis_off()
        ax.set_facecolor("#eeeeef")

        H0, W, Grid_Sign = self.initialize_wave()

        Z = self.calc_wave(H0, W, 0, Grid_Sign)

        surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm)

        # Set axis limits
        ax.set_xlim(self.param['xLim'])
        ax.set_ylim(self.param['yLim'])
        ax.set_zlim(self.param['zLim'])

        def update(frame):
            Z = self.calc_wave( H0, W, frame, Grid_Sign)
            ax.clear()
            ax.plot_surface(X, Y, Z, cmap=cm.coolwarm)
            # Set axis limits again after clearing
            ax.set_xlim(self.param['xLim'])
            ax.set_ylim(self.param['yLim'])
            ax.set_zlim(self.param['zLim'])
            ax.set_axis_off()
            ax.set_facecolor("#eeeeef")
            return surf,
        

        ani = FuncAnimation(fig, update, frames=np.linspace(0, 10, 100), blit=False)
        plt.show()

# wave = Wave_Generator()
# wave.init_gui()