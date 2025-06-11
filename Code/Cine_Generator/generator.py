import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, Set
from helper.helper import load_image
from scipy.ndimage import center_of_mass
from helper.wave_simulation import (
    run_wave_simulation, 
    adjust_displacement_for_strain, 
    generate_radial_logistic_label_from_shifted_grids, 
    adjust_displacement_with_ring, 
    animate_deformed_masked_mri,
    compute_strains
)

# Constants
os.chdir(os.path.dirname(__file__))
SIZE_IMAGE = 512
FOV = 512


class ConfigLoader:
    """Handles loading and parsing of configuration files."""
    
    def __init__(self, config_path: str):
        """
        Initialize ConfigLoader with a configuration file path.

        Args:
            config_path (str): Path to the configuration file
        """
        self.config_path = config_path
        self.config = self.load_config()
        
    def load_config(self) -> Dict[str, Any]:
        """
        Load configuration from JSON file.

        Returns:
            Dict[str, Any]: Configuration dictionary
        """
        with open(self.config_path, "r") as file:
            return json.load(file)
            
    def get_parameters(self) -> Dict[str, Any]:
        """
        Get parameters based on configuration type.

        Returns:
            Dict[str, Any]: Dictionary containing simulation parameters
        """
        if "parameters" in self.config_path:
            return {
                'wind_dir': self.config["wind_dir"],
                'wind_speed': self.config["wind_speed"],
                'num_frames': int(np.random.uniform(self.config["num_frames"][0], self.config["num_frames"][1])),
                'FOV': FOV,
                'sizeImage': SIZE_IMAGE,            
                'StrainEpPeak': self.config["strain_validation"]["StrainEpPeak"],                
                'inner_radius': int(np.random.uniform(self.config["strain_validation"]["inner_radius"][0], self.config["strain_validation"]["inner_radius"][1])),
                'outer_radius': int(np.random.uniform(self.config["strain_validation"]["outer_radius"][0], self.config["strain_validation"]["outer_radius"][1])),
            }
        return {
            'no_of_cines': self.config["generator"]["no_of_cines"],
            'patinet_start': self.config["generator"]["patients_start"],
            'patinet_end': self.config["generator"]["patients_end"],
            'folder': self.config["generator"]["folder"],
        }


class DirectoryManager:
    """Manages directory structure and file paths for the simulation."""
    
    def __init__(self, data_path: str = None):
        """
        Initialize DirectoryManager with optional data path.

        Args:
            data_path (str, optional): Custom data directory path
        """
        self.current_script = Path(__file__)
        self.project_root = self.current_script.parent.parent.parent
        self.data_dir = Path(data_path) if data_path else self.project_root / "Data" / "ACDC" / "train_numpy"
        self.saved_displacements = self.current_script.parent / "generatedData" / "Displacements"
        self.saved_frames = self.current_script.parent / "generatedData" / "Frames"
        self.saved_cines = self.current_script.parent / "generatedData" / "Cines"
        self.setup_directories()
        
    def setup_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        for directory in [self.saved_displacements, self.saved_frames, self.saved_cines]:
            directory.mkdir(parents=True, exist_ok=True)
        os.chdir(os.path.dirname(__file__))
        
    def get_patient_path(self, patient_number: str) -> Path:
        """
        Get path for a specific patient.

        Args:
            patient_number (str): Patient identifier

        Returns:
            Path: Path to patient directory
        """
        return self.data_dir / f"patient{patient_number}"
    
    def get_npy_path(self, patient_number: str, frame_number: str, slice_number: int) -> Path:
        """
        Get path for a specific patient's frame and slice.

        Args:
            patient_number (str): Patient identifier
            frame_number (str): Frame number
            slice_number (int): Slice number

        Returns:
            Path: Path to the .npy file
        """
        patient_folder = self.get_patient_path(patient_number)
        return patient_folder / f"patient{patient_number}_frame{frame_number}_slice_{slice_number}_ACDC.npy"


class WaveSimulator:
    """Handles wave simulation and displacement calculations."""
    
    def __init__(self, params: Dict[str, Any]):
        """
        Initialize WaveSimulator with parameters.

        Args:
            params (Dict[str, Any]): Simulation parameters
        """
        self.params = params
        
    def simulate_phillips_spectrum(self) -> Dict[str, Any]:
        """
        Run wave simulation using Phillips spectrum.

        Returns:
            Dict[str, Any]: Simulation results
        """
        return run_wave_simulation(self.params)

    def scale_displacements(self, simulation_results: Dict[str, Any]) -> Tuple[Dict[str, Any], float, float]:
        """
        Scale displacement values to appropriate ranges.

        Args:
            simulation_results (Dict[str, Any]): Raw simulation results

        Returns:
            Tuple[Dict[str, Any], float, float]: Scaled results and delta values
        """
        deltaX = self.params['FOV'] / self.params['sizeImage']
        deltaY = self.params['FOV'] / self.params['sizeImage']
        MaxFrameDispl = 5 * self.params['FOV'] / self.params['sizeImage']
        MaxFrameDisplX = MaxFrameDispl
        MinFrameDisplX = -MaxFrameDispl
        MaxFrameDisplY = MaxFrameDispl
        MinFrameDisplY = -MaxFrameDispl

        FrameDisplX = simulation_results["FrameDisplX"].copy()
        FrameDisplY = simulation_results["FrameDisplY"].copy()

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
    """Calculates strain values from displacement fields."""
    
    @staticmethod
    def compute_strain(FrameDisplX: np.ndarray, FrameDisplY: np.ndarray, deltaX: float, deltaY: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute strain components from displacement fields.

        Args:
            FrameDisplX (np.ndarray): X-displacement field
            FrameDisplY (np.ndarray): Y-displacement field
            deltaX (float): X-direction spatial step
            deltaY (float): Y-direction spatial step

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Principal strains and angle
        """
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
    """Converts Cartesian coordinates to polar coordinates for displacement fields."""
    
    def __init__(self, params: Dict[str, Any], Mask: np.ndarray):
        """
        Initialize PolarConverter with parameters and mask.

        Args:
            params (Dict[str, Any]): Simulation parameters
            Mask (np.ndarray): Binary mask
        """
        self.params = params
        self.Mask = Mask
        
    def convert_to_polar(self, FrameDisplX: np.ndarray, FrameDisplY: np.ndarray, 
                        deltaX: float, deltaY: float, outputXCoord: np.ndarray, 
                        outputYCoord: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert displacement fields to polar coordinates.

        Args:
            FrameDisplX (np.ndarray): X-displacement field
            FrameDisplY (np.ndarray): Y-displacement field
            deltaX (float): X-direction spatial step
            deltaY (float): Y-direction spatial step
            outputXCoord (np.ndarray): X-coordinate grid
            outputYCoord (np.ndarray): Y-coordinate grid

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Polar displacement fields and coordinate grids
        """
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
            xMat_shft, yMat_shft, steepness=50, cutoff=0.04
        )
        
        label_map_3d = np.repeat(label_map[:, :, np.newaxis], self.params['num_frames'], axis=2)
        label_map_3d[label_map_3d < 0.04] = 0
        
        u_r = FrameDisplRads / RadiaScale * label_map_3d
        u_theta = FrameDisplThta * ThetaScale * label_map_3d
        
        FrameDisplXOrgAdjPlr = u_r * np.cos(Theta) - u_theta * np.sin(Theta)
        FrameDisplYOrgAdjPlr = u_r * np.sin(Theta) + u_theta * np.cos(Theta)
        
        return FrameDisplXOrgAdjPlr, FrameDisplYOrgAdjPlr, xMat_shft, yMat_shft


def main() -> None:
    """Main function to run the wave simulation and generate cines."""
    config_parameter_loader = ConfigLoader("config_parameters.json")
    config_generator_loader = ConfigLoader("config_generator.json")
    generator_params = config_generator_loader.get_parameters()
    
    dir_manager = DirectoryManager(generator_params['folder'])
    processed_combinations: Set[Tuple[str, str, int]] = set()
    strain_calculator = StrainCalculator()
    
    for _ in range(generator_params['no_of_cines']):
        params = config_parameter_loader.get_parameters()
        wave_simulator = WaveSimulator(params)
        patient_number = str(np.random.randint(generator_params['patinet_start'], generator_params['patinet_end'])).zfill(3)
        slice_number = np.random.randint(1, 6)
        
        for frame_number in range(1, 31):
            frame_number = str(frame_number).zfill(2)
            
            combination = (patient_number, frame_number, slice_number)
            if combination not in processed_combinations:
                npy_file = dir_manager.get_npy_path(patient_number, frame_number, slice_number)
                if npy_file.exists():
                    print(f"Patient: {patient_number}, Frame: {frame_number}, Slice: {slice_number}")
                    processed_combinations.add(combination)            
                    Mask = load_image(npy_file, is_mask=True)
                    Image = load_image(npy_file)
                    
                    simulation_results = wave_simulator.simulate_phillips_spectrum()
                    simulation_results, deltaX, deltaY = wave_simulator.scale_displacements(simulation_results)
                    
                    FrameDisplX_Org = simulation_results["FrameDisplX"].copy()
                    FrameDisplY_Org = simulation_results["FrameDisplY"].copy()
                    
                    FrameDisplX_Org_adjusted = FrameDisplX_Org.copy()
                    FrameDisplY_Org_adjusted = FrameDisplY_Org.copy()
                    
                    FrameDisplX_Org_adjusted, FrameDisplY_Org_adjusted, Ep1All, Ep2All, Ep3All, strain_history = adjust_displacement_for_strain(
                        FrameDisplX_Org_adjusted, FrameDisplY_Org_adjusted, deltaX, deltaY, params['StrainEpPeak'],
                        strain_tolerance=0.01, max_iterations=20
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
                        strain_tolerance=0.01, max_iterations=20
                    )
                    
                    if Image.ndim == 3 and Image.shape[2] == 3:
                        Image = Image[:, :, 0]

                    if Mask.ndim == 3 and Mask.shape[2] == 3:
                        Mask = Mask[:, :, 0]

                    _, Image_deformed_all, Mask_deformed_all, T3DDispX_masked_all, T3DDispY_masked_all, MaskFadedDefrmd_all = animate_deformed_masked_mri(
                        Image, Mask, FrameDisplXOrgAdjPlrAdj, FrameDisplYOrgAdjPlrAdj, 
                        save_file=True, save_mode=True, 
                        patinet_file_name=npy_file, 
                        output_filename=f"cine_patient{patient_number}_frame{frame_number}_slice_{slice_number}_ACDC.mp4"
                    )
                    
                    Ep1All_dilated_mask, Ep2All_dilated_mask, Ep3All_dilated_mask = compute_strains(
                        T3DDispX_masked_all, T3DDispY_masked_all, deltaX, deltaY
                    )

                    # Save the cine results as a .npy file
                    cine_file_name = dir_manager.saved_cines / f"cine_patient{patient_number}_frame{frame_number}_slice_{slice_number}_ACDC.npy" 
                    np.save(cine_file_name, {
                        "Image": Image_deformed_all,
                        "Mask": Mask_deformed_all,
                        "T3DDispX_masked": T3DDispX_masked_all,
                        "T3DDispY_masked": T3DDispY_masked_all,
                        "MaskFadedDefrmd": MaskFadedDefrmd_all,
                        "Ep1All_dilated_mask": Ep1All_dilated_mask,
                        "Ep2All_dilated_mask": Ep2All_dilated_mask,
                        "Ep3All_dilated_mask": Ep3All_dilated_mask,
                        "wind_dir_x": simulation_results['wind_dir_x'],
                        "wind_dir_y": simulation_results['wind_dir_y'],
                        "wind_speed_x": simulation_results['wind_speed_x'],
                        "wind_speed_y": simulation_results['wind_speed_y'],                    
                        "StrainEpPeak": params['StrainEpPeak'],
                        "inner_radius": params['inner_radius'],
                        "outer_radius": params['outer_radius'],                        
                        "no_of_frames": params['num_frames'],
                    })
                    break
    
    print(f"Generated {generator_params['no_of_cines']} cines from patient number {generator_params['patinet_start']} to patient number {generator_params['patinet_end']} in folder {generator_params['folder']}")


if __name__ == "__main__":
    main()