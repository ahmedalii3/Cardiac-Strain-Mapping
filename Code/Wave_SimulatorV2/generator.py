import os
import numpy as np
from pathlib import Path
from helper import load_image
from scipy.ndimage import center_of_mass
from wave_simulation import (
    StrainWaveParams, 
    initialize_wave, 
    phillips_spectrum, 
    run_wave_simulation, 
    adjust_displacement_for_strain, 
    generate_radial_logistic_label_from_shifted_grids, 
    adjust_displacement_with_ring, 
    animate_deformed_masked_mri
)
import json

class ConfigLoader:
    def __init__(self, config_path):
        self.config_path = config_path
        self.config = self.load_config()
        
    def load_config(self):
        with open(self.config_path, "r") as file:
            return json.load(file)
            
    def get_parameters(self):
        return {
            'wind_dir': self.config["wind_dir"],
            'wind_speed': self.config["wind_speed"],
            'num_frames': self.config["num_frames"],
            'FOV': self.config["FOV"],
            'sizeImage': self.config["sizeImage"],
            'steepness': self.config["radial_logistic"]["steepness"],
            'cutoff': self.config["radial_logistic"]["cutoff"],
            'StrainEpPeak': self.config["strain_validation"]["StrainEpPeak"],
            'max_iterations': self.config["strain_validation"]["max_iterations"],
            'inner_radius': self.config["strain_validation"]["inner_radius"],
            'outer_radius': self.config["strain_validation"]["outer_radius"],
            'strain_tolerance': self.config["strain_validation"]["strain_tolerance"],
            'DATASET_SIZE': self.config["generator"]["DATASET_SIZE"],
            'patinet_start': self.config["generator"]["patients_start"],
            'patinet_end': self.config["generator"]["patients_end"],
            'folder': self.config["generator"]["folder"]
        }

class DirectoryManager:
    def __init__(self):
        self.current_script = Path(__file__)
        self.project_root = self.current_script.parent.parent.parent
        self.data_dir = self.project_root / "Data" / "ACDC"
        self.saved_displacements = self.current_script.parent / "generatedData" / "Displacements"
        self.saved_frames = self.current_script.parent / "generatedData" / "Frames"
        
    def setup_directories(self, folder):
        self.data_dir = self.data_dir / folder
        for directory in [self.saved_displacements, self.saved_frames]:
            directory.mkdir(parents=True, exist_ok=True)
        os.chdir(os.path.dirname(__file__))
        
    def get_patient_path(self, patient_number):
        return self.data_dir / f"patient{patient_number}"
    
    def get_npy_path(self, patient_number, frame_number, slice_number):
        patient_folder = self.get_patient_path(patient_number)
        return patient_folder / f"patient{patient_number}_frame{frame_number}_slice_{slice_number}_ACDC.npy"

class WaveSimulator:
    def __init__(self, params):
        self.params = params
        
    def simulate_phillips_spectrum(self):
        params = StrainWaveParams(
            wind_dir_vector=(np.cos(np.radians(self.params['wind_dir'])), 
                           np.sin(np.radians(self.params['wind_dir']))),
            wind_speed=self.params['wind_speed']
        )
        H0, W, Grid_Sign = initialize_wave(params)
        
        mesh_lim = np.pi * params.meshsize / params.patchsize
        N = np.linspace(-mesh_lim, mesh_lim, params.meshsize)
        M = np.linspace(-mesh_lim, mesh_lim, params.meshsize)
        Kx, Ky = np.meshgrid(N, M)
        P = phillips_spectrum(Kx, Ky, params.wind_dir_vector, params.wind_speed, 
                            params.A, params.g)
        return run_wave_simulation(num_frames=self.params['num_frames'])

    def scale_displacements(self, simulation_results):
        deltaX = self.params['FOV'] / self.params['sizeImage']
        deltaY = self.params['FOV'] / self.params['sizeImage']
        MaxFrameDispl = 5 * self.params['FOV'] / self.params['sizeImage']
        MaxFrameDisplX = MaxFrameDispl
        MinFrameDisplX = -MaxFrameDispl
        MaxFrameDisplY = MaxFrameDispl
        MinFrameDisplY = -MaxFrameDispl

        FrameDisplX = simulation_results["FrameDisplX"].copy()
        FrameDisplY = simulation_results["FrameDisplY"].copy()
        UseGlobalScaling = False

        if UseGlobalScaling:
            MinWaveX, MaxWaveX = np.min(FrameDisplX), np.max(FrameDisplX)
            MinWaveY, MaxWaveY = np.min(FrameDisplY), np.max(FrameDisplY)
            FrameDisplX = ((FrameDisplX - MinWaveX) / (MaxWaveX - MinWaveX)) * (MaxFrameDisplX - MinFrameDisplX) + MinFrameDisplX
            FrameDisplY = ((FrameDisplY - MinWaveY) / (MaxWaveY - MinWaveY)) * (MaxFrameDisplY - MinFrameDisplY) + MinFrameDisplY
        else:
            for iframe in range(self.params['num_frames']):
                MinWaveX, MaxWaveX = np.min(FrameDisplX[:, :, iframe]), np.max(FrameDisplX[:, :, iframe])
                MinWaveY, MaxWaveY = np.min(FrameDisplY[:, :, iframe]), np.max(FrameDisplY[:, :, iframe])
                FrameDisplX[:, :, iframe] = ((FrameDisplX[:, :, iframe] - MinWaveX) / (MaxWaveX - MinWaveX)) * (MaxFrameDisplX - MinFrameDisplX) + MinFrameDisplX
                FrameDisplY[:, :, iframe] = ((FrameDisplY[:, :, iframe] - MinWaveY) / (MaxWaveY - MinWaveY)) * (MaxFrameDisplY - MinFrameDisplY) + MinFrameDisplY

        simulation_results["FrameDisplX"] = FrameDisplX
        simulation_results["FrameDisplY"] = FrameDisplY
        
        if self.params['sizeImage'] % 2 != 0:
            outputXCoord = np.arange(-(self.params['sizeImage'] - 1) / 2, (self.params['sizeImage'] - 1) / 2 + 1) * deltaX
            outputYCoord = np.arange(-(self.params['sizeImage'] - 1) / 2, (self.params['sizeImage'] - 1) / 2 + 1) * deltaY
        else:
            outputXCoord = np.arange(-self.params['sizeImage'] / 2, self.params['sizeImage'] / 2) * deltaX
            outputYCoord = np.arange(-self.params['sizeImage'] / 2, self.params['sizeImage'] / 2) * deltaY
            
        simulation_results["outputXCoord"] = outputXCoord
        simulation_results["outputYCoord"] = outputYCoord
        return simulation_results, deltaX, deltaY

class StrainCalculator:
    @staticmethod
    def compute_strain(FrameDisplX, FrameDisplY, deltaX, deltaY):
        UXx, UXy = np.gradient(FrameDisplX, deltaX, deltaY, axis=(0, 1))
        UYx, UYy = np.gradient(FrameDisplY, deltaX, deltaY, axis=(0, 1))
        
        ExxAll = (2 * UXx - (UXx**2 + UYx**2)) / 2
        ExyAll = (UXy + UYx - (UXx * UXy + UYx * UYy)) / 2
        EyyAll = (2 * UYy - (UXy**2 + UYy**2)) / 2
        
        ThetaEp = 0.5 * np.arctan2(2 * ExyAll, ExxAll - EyyAll)
        Ep1All = (ExxAll + EyyAll) / 2 + np.sqrt(((ExxAll - EyyAll) / 2)**2 + ExyAll**2)
        Ep2All = (ExxAll + EyyAll) / 2 - np.sqrt(((ExxAll - EyyAll) / 2)**2 + ExyAll**2)
        Ep3All = 1 / ((1 + Ep1All) * (1 + Ep2All)) - 1
        return Ep1All, Ep2All, Ep3All, ThetaEp

class PolarConverter:
    def __init__(self, params, Mask):
        self.params = params
        self.Mask = Mask
        
    def convert_to_polar(self, FrameDisplX, FrameDisplY, deltaX, deltaY, outputXCoord, outputYCoord):
        heartCenter = (np.array(center_of_mass(self.Mask)) - np.array(self.Mask.shape) / 2)[:2][::-1]
        FrameDisplRads = FrameDisplX.copy()
        FrameDisplThta = FrameDisplY.copy()
        
        NY, NX, num_frames = FrameDisplX.shape
        xMat = np.tile(outputXCoord, (NY, 1))
        yMat = np.tile(outputYCoord[:, np.newaxis], (1, NX))
        x0 = heartCenter[1] * deltaX
        y0 = heartCenter[0] * deltaY
        
        xMat_shft = xMat - y0
        yMat_shft = yMat - x0
        
        R = np.sqrt(xMat_shft**2 + yMat_shft**2)
        Theta = np.arctan2(yMat_shft, xMat_shft)
        
        R = np.repeat(R[:, :, np.newaxis], self.params['num_frames'], axis=2)
        Theta = np.repeat(Theta[:, :, np.newaxis], self.params['num_frames'], axis=2)
        
        ThetaScale = ((R + 0.001) / np.max(R + 0.001)) * 3
        ThetaScale[ThetaScale > 2] = 2
        
        RadiaScale = (R + np.finfo(float).eps) / np.max(R)
        RadiaScale *= RadiaScale
        RadiaScale[np.abs(RadiaScale) < 0.8] = 0.8 * np.sign(RadiaScale[np.abs(RadiaScale) < 0.8])
        
        x = np.linspace(-1, 1, self.params['sizeImage'])
        y = np.linspace(-1, 1, self.params['sizeImage'])
        X, Y = np.meshgrid(x, y)
        
        frame = 0
        ud_r = np.ones_like(R[:, :, frame])
        ud_theta = np.ones_like(R[:, :, frame])
        
        ud_r_scaled = ud_r / RadiaScale[:, :, frame]
        ud_theta_scaled = ud_theta * ThetaScale[:, :, frame]
        
        ud_x = ud_r_scaled * np.cos(Theta[:, :, frame]) - ud_theta_scaled * np.sin(Theta[:, :, frame])
        ud_y = ud_r_scaled * np.sin(Theta[:, :, frame]) + ud_theta_scaled * np.cos(Theta[:, :, frame])
        
        label_map = generate_radial_logistic_label_from_shifted_grids(
            xMat_shft, yMat_shft, steepness=self.params['steepness'], cutoff=self.params['cutoff']
        )
        
        label_map_3d = np.repeat(label_map[:, :, np.newaxis], self.params['num_frames'], axis=2)
        label_map_3d[label_map_3d < 0.04] = 0
        
        u_r = FrameDisplRads / RadiaScale * label_map_3d
        u_theta = FrameDisplThta * ThetaScale * label_map_3d
        
        FrameDisplXOrgAdjPlr = u_r * np.cos(Theta) - u_theta * np.sin(Theta)
        FrameDisplYOrgAdjPlr = u_r * np.sin(Theta) + u_theta * np.cos(Theta)
        
        return FrameDisplXOrgAdjPlr, FrameDisplYOrgAdjPlr, xMat_shft, yMat_shft

def main():
    config_loader = ConfigLoader("/Users/osama/GP-2025-Strain/Code/Wave_SimulatorV2/config.json")
    params = config_loader.get_parameters()
    
    dir_manager = DirectoryManager()
    dir_manager.setup_directories(params['folder'])
    
    size = 0
    processed_combinations = set()
    wave_simulator = WaveSimulator(params)
    strain_calculator = StrainCalculator()
    
    while size < params['DATASET_SIZE']:
        patient_number = str(np.random.randint(params['patinet_start'], params['patinet_end'])).zfill(3)
        slice_number = np.random.randint(0, 11)
        for frame_number in range(1, 31):
            combination = (patient_number, frame_number, slice_number)
            if combination not in processed_combinations:
                npy_file = dir_manager.get_npy_path(patient_number, frame_number, slice_number)
                if npy_file.exists():
                    print(f"Patient: {patient_number}, Frame: {frame_number}, Slice: {slice_number}")
                    processed_combinations.add(combination)            
                    Mask = load_image(npy_file, isMask=True)
                    Image = load_image(npy_file)
                    
                    simulation_results = wave_simulator.simulate_phillips_spectrum()
                    simulation_results, deltaX, deltaY = wave_simulator.scale_displacements(simulation_results)
                    
                    FrameDisplX_Org = simulation_results["FrameDisplX"].copy()
                    FrameDisplY_Org = simulation_results["FrameDisplY"].copy()
                    
                    FrameDisplX_Org_adjusted = FrameDisplX_Org.copy()
                    FrameDisplY_Org_adjusted = FrameDisplY_Org.copy()
                    
                    FrameDisplX_Org_adjusted, FrameDisplY_Org_adjusted, Ep1All, Ep2All, Ep3All, strain_history = adjust_displacement_for_strain(
                        FrameDisplX_Org_adjusted, FrameDisplY_Org_adjusted, deltaX, deltaY, params['StrainEpPeak'],
                        strain_tolerance=params['strain_tolerance'], max_iterations=params['max_iterations']
                    )
                    
                    polar_converter = PolarConverter(params, Mask)
                    FrameDisplXOrgAdjPlr, FrameDisplYOrgAdjPlr, xMat_shft, yMat_shft = polar_converter.convert_to_polar(
                        FrameDisplX_Org_adjusted, FrameDisplY_Org_adjusted, deltaX, deltaY,
                        simulation_results["outputXCoord"], simulation_results["outputYCoord"]
                    )
                    
                    Ep1All, Ep2All, Ep3All, ThetaEp = strain_calculator.compute_strain(
                        FrameDisplXOrgAdjPlr, FrameDisplYOrgAdjPlr, deltaX, deltaY
                    )
                    
                    FrameDisplXOrgAdjPlrAdj = FrameDisplXOrgAdjPlr.copy()
                    FrameDisplYOrgAdjPlrAdj = FrameDisplYOrgAdjPlr.copy()
                    
                    FrameDisplXOrgAdjPlrAdj, FrameDisplYOrgAdjPlrAdj, Ep1All, Ep2All, Ep3All, strain_history = adjust_displacement_with_ring(
                        FrameDisplXOrgAdjPlrAdj, FrameDisplYOrgAdjPlrAdj, deltaX, deltaY, params['StrainEpPeak'],
                        xMat_shft, yMat_shft, inner_radius=params['inner_radius'], outer_radius=params['outer_radius'],
                        strain_tolerance=params['strain_tolerance'], max_iterations=params['max_iterations']
                    )
                    
                    ani_masked = animate_deformed_masked_mri(
                        Image, Mask, FrameDisplXOrgAdjPlrAdj, FrameDisplYOrgAdjPlrAdj,
                        output_filename="deformed_mri_masked_polar_rescaled.mp4",
                        save_file=True, save_mode=True, patinet_file_name=npy_file
                    )
                    
                    file_count = len([f for f in dir_manager.saved_displacements.glob('*') 
                                    if f.is_file() and f.name != ".DS_Store"])
                    size = file_count
                    print(f"Size: {size}")
    
    print("All done!")

if __name__ == "__main__":
    main()