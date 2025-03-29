import os
import numpy as np
from pathlib import Path
from helper import load_image
from scipy.ndimage import center_of_mass
from wave_simulation import StrainWaveParams, initialize_wave, phillips_spectrum, run_wave_simulation, adjust_displacement_for_strain, generate_radial_logistic_label_from_shifted_grids, adjust_displacement_with_ring, animate_deformed_masked_mri
import json

os.chdir(os.path.dirname(__file__)) #change working directory to current directory

# Load parameters from JSON
with open("/Users/osama/GP-2025-Strain/Code/Wave_SimulatorV2/config.json", "r") as file:
    config = json.load(file)

# Access parameters
wind_dir = config["wind_dir"]
wind_speed = config["wind_speed"]
num_frames = config["num_frames"]
FOV = config["FOV"]
sizeImage = config["sizeImage"]

steepness = config["radial_logistic"]["steepness"]
cutoff = config["radial_logistic"]["cutoff"]

StrainEpPeak = config["strain_validation"]["StrainEpPeak"]
max_iterations = config["strain_validation"]["max_iterations"]
inner_radius = config["strain_validation"]["inner_radius"]
outer_radius = config["strain_validation"]["outer_radius"]
strain_tolerance = config["strain_validation"]["strain_tolerance"]

DATASET_SIZE = config["generator"]["DATASET_SIZE"]
patinet_start = config["generator"]["patients_start"]
patinet_end = config["generator"]["patients_end"]
folder = config["generator"]["folder"]


# Get path to current script (create_dataset.py)
current_script = Path(__file__)

# Calculate project root (GP-2025-Strain directory)
project_root = current_script.parent.parent.parent  # Goes up from Code/Wave to main project
data_dir = project_root / "Data" / "ACDC" / folder

# Define important directories
saved_displacements = current_script.parent / "generatedData" / "Displacements"
saved_frames = current_script.parent / "generatedData" / "Frames"

# Create directories if they don't exist
for directory in [saved_displacements, saved_frames]:
    directory.mkdir(parents=True, exist_ok=True)


size = 0
processed_combinations = set()
while size < DATASET_SIZE:
    patient_number = str(np.random.randint(patinet_start, patinet_end)).zfill(3)
    slice_number = np.random.randint(0, 11)
    for frame_number in range(1, 31):
        combination = (patient_number, frame_number, slice_number)
        if combination not in processed_combinations:
            patient_folder = data_dir / f"patient{patient_number}"
            npy_file = patient_folder / f"patient{patient_number}_frame{frame_number}_slice_{slice_number}_ACDC.npy"
            if npy_file.exists():
                print(f"Patient: {patient_number}, Frame: {frame_number}, Slice: {slice_number}")
                processed_combinations.add(combination)
            
                ### Import Image and Mask ###
                
                Mask = load_image(npy_file, isMask=True)
                Image = load_image(npy_file)


                ### Philips spectrum ###
                params = StrainWaveParams(wind_dir_vector=(np.cos(np.radians(wind_dir)), np.sin(np.radians(wind_dir))), wind_speed = wind_speed)
                # Compute initial wave height and frequency
                H0, W, Grid_Sign = initialize_wave(params)

                # Compute the Phillips spectrum for visualization
                mesh_lim = np.pi * params.meshsize / params.patchsize
                N = np.linspace(-mesh_lim, mesh_lim, params.meshsize)
                M = np.linspace(-mesh_lim, mesh_lim, params.meshsize)
                Kx, Ky = np.meshgrid(N, M)
                P = phillips_spectrum(Kx, Ky, params.wind_dir_vector, params.wind_speed, params.A, params.g)

                ### Simulate wave displacements ###
                simulation_results = run_wave_simulation(num_frames=num_frames)

                # Extract displacement frames and wind directions
                FrameDisplX = simulation_results["FrameDisplX"]
                FrameDisplY = simulation_results["FrameDisplY"]
                wind_dir_x = simulation_results["wind_dir_x"]
                wind_dir_y = simulation_results["wind_dir_y"]
                wind_speed_x = simulation_results["wind_speed_x"]
                wind_speed_y = simulation_results["wind_speed_y"]


                ### Fixing scaling code ###
                deltaX = FOV / sizeImage
                deltaY = FOV / sizeImage
                # Define displacement limits based on FOV and image size
                MaxFrameDispl = 5 * FOV / sizeImage  # mm
                MaxFrameDisplX = MaxFrameDispl
                MinFrameDisplX = -MaxFrameDispl
                MaxFrameDisplY = MaxFrameDispl
                MinFrameDisplY = -MaxFrameDispl

                # Extract displacement arrays from simulation results
                FrameDisplX = simulation_results["FrameDisplX"].copy()
                FrameDisplY = simulation_results["FrameDisplY"].copy()

                # Define whether to use global or per-frame scaling
                UseGlobalScaling = False  # Set to True for global scaling


                if UseGlobalScaling:
                    # Global scaling: Normalize based on all frames
                    MinWaveX, MaxWaveX = np.min(FrameDisplX), np.max(FrameDisplX)
                    MinWaveY, MaxWaveY = np.min(FrameDisplY), np.max(FrameDisplY)

                    FrameDisplX = ((FrameDisplX - MinWaveX) / (MaxWaveX - MinWaveX)) * (MaxFrameDisplX - MinFrameDisplX) + MinFrameDisplX
                    FrameDisplY = ((FrameDisplY - MinWaveY) / (MaxWaveY - MinWaveY)) * (MaxFrameDisplY - MinFrameDisplY) + MinFrameDisplY

                else:
                    # Per-frame scaling: Normalize each frame individually
                    for iframe in range(num_frames):
                        MinWaveX, MaxWaveX = np.min(FrameDisplX[:, :, iframe]), np.max(FrameDisplX[:, :, iframe])
                        MinWaveY, MaxWaveY = np.min(FrameDisplY[:, :, iframe]), np.max(FrameDisplY[:, :, iframe])

                        FrameDisplX[:, :, iframe] = ((FrameDisplX[:, :, iframe] - MinWaveX) / (MaxWaveX - MinWaveX)) * (MaxFrameDisplX - MinFrameDisplX) + MinFrameDisplX
                        FrameDisplY[:, :, iframe] = ((FrameDisplY[:, :, iframe] - MinWaveY) / (MaxWaveY - MinWaveY)) * (MaxFrameDisplY - MinFrameDisplY) + MinFrameDisplY

                # Store updated displacement in simulation results
                simulation_results["FrameDisplX"] = FrameDisplX
                simulation_results["FrameDisplY"] = FrameDisplY


                if sizeImage % 2 != 0:  # Odd
                    outputXCoord = np.arange(-(sizeImage - 1) / 2, (sizeImage - 1) / 2 + 1) * deltaX
                    outputYCoord = np.arange(-(sizeImage - 1) / 2, (sizeImage - 1) / 2 + 1) * deltaY
                else:  # Even
                    outputXCoord = np.arange(-sizeImage / 2, sizeImage / 2) * deltaX
                    outputYCoord = np.arange(-sizeImage / 2, sizeImage / 2) * deltaY

                # Store coordinates
                simulation_results["outputXCoord"] = outputXCoord
                simulation_results["outputYCoord"] = outputYCoord


                ### Compute Strain ###
                FrameDisplX_Org = simulation_results["FrameDisplX"].copy()
                FrameDisplY_Org = simulation_results["FrameDisplY"].copy()
                # Compute displacement gradients
                UXx, UXy = np.gradient(FrameDisplX_Org, deltaX, deltaY, axis=(0, 1))
                UYx, UYy = np.gradient(FrameDisplY_Org, deltaX, deltaY, axis=(0, 1))

                # Compute strain components
                ExxAll = (2 * UXx - (UXx**2 + UYx**2)) / 2
                ExyAll = (UXy + UYx - (UXx * UXy + UYx * UYy)) / 2
                EyyAll = (2 * UYy - (UXy**2 + UYy**2)) / 2
                # Compute principal strains
                ThetaEp = 0.5 * np.arctan2(2 * ExyAll, ExxAll - EyyAll)
                Ep1All = (ExxAll + EyyAll) / 2 + np.sqrt(((ExxAll - EyyAll) / 2)**2 + ExyAll**2)
                Ep2All = (ExxAll + EyyAll) / 2 - np.sqrt(((ExxAll - EyyAll) / 2)**2 + ExyAll**2)

                # Compute incompressibility strain component
                Ep3All = 1 / ((1 + Ep1All) * (1 + Ep2All)) - 1

                FrameDisplX_Org_adjusted=FrameDisplX_Org.copy()
                FrameDisplY_Org_adjusted=FrameDisplY_Org.copy()
                # Run the optimized function
                FrameDisplX_Org_adjusted, FrameDisplY_Org_adjusted, Ep1All, Ep2All, Ep3All, strain_history = adjust_displacement_for_strain(
                    FrameDisplX_Org_adjusted, FrameDisplY_Org_adjusted, deltaX, deltaY, StrainEpPeak, strain_tolerance=strain_tolerance, max_iterations=max_iterations
                )

                ### Polar Conversion ###
                heartCenter = (np.array(center_of_mass(Mask)) - np.array(Mask.shape) / 2)[:2][::-1]
                # Extract displacement frames
                FrameDisplRads = FrameDisplX_Org_adjusted.copy()  # Radial displacement
                FrameDisplThta = FrameDisplY_Org_adjusted.copy()  # Angular displacement

                # Get image dimensions
                NY, NX, num_frames = FrameDisplX.shape

                # Create coordinate grids
                xMat = np.tile(outputXCoord, (NY, 1))  # X grid in mm
                yMat = np.tile(outputYCoord[:, np.newaxis], (1, NX))  # Y grid in mm
                # Define the center of the heart (polar coordinate origin)
                x0 = heartCenter[1] * deltaX  # X-center in mm
                y0 = heartCenter[0] * deltaY  # Y-center in mm

                # Shift coordinates so heart is at the center
                xMat_shft = xMat - y0
                yMat_shft = yMat - x0

                # Compute radial (R) and angular (Theta) coordinates
                R = np.sqrt(xMat_shft**2 + yMat_shft**2)
                Theta = np.arctan2(yMat_shft, xMat_shft)

                # Replicate R and Theta across frames
                R = np.repeat(R[:, :, np.newaxis], num_frames, axis=2)
                Theta = np.repeat(Theta[:, :, np.newaxis], num_frames, axis=2)
                # Compute Theta scaling
                ThetaScale = ((R + 0.001) / np.max(R + 0.001)) * 3
                ThetaScale[ThetaScale > 2] = 2

                # Compute Radial scaling
                RadiaScale = (R + np.finfo(float).eps) / np.max(R)
                RadiaScale*=RadiaScale
                RadiaScale[np.abs(RadiaScale) < 0.8] = 0.8 * np.sign(RadiaScale[np.abs(RadiaScale) < 0.8])
                # Create grid
                x = np.linspace(-1, 1, sizeImage)
                y = np.linspace(-1, 1, sizeImage)
                X, Y = np.meshgrid(x, y)

                frame=0
                # Simulate uniform unit vectors in radial and tangential directions
                ud_r = np.ones_like(R[:, :, frame])     # baseline radial motion
                ud_theta = np.ones_like(R[:, :, frame]) # baseline tangential motion


                # Apply scaling to the unit field
                ud_r_scaled = ud_r / RadiaScale[:, :, frame]
                ud_theta_scaled = ud_theta * ThetaScale[:, :, frame]

                # Convert to Cartesian
                ud_x = ud_r_scaled * np.cos(Theta[:, :, frame]) - ud_theta_scaled * np.sin(Theta[:, :, frame])
                ud_y = ud_r_scaled * np.sin(Theta[:, :, frame]) + ud_theta_scaled * np.cos(Theta[:, :, frame])

                label_map = generate_radial_logistic_label_from_shifted_grids(xMat_shft, yMat_shft, steepness=steepness, cutoff=cutoff)

                label_map_3d = np.repeat(label_map[:, :, np.newaxis], num_frames, axis=2)
                label_map_3d[label_map_3d < 0.04]=0  # e.g., threshold = 0.05

                # Compute polar displacement components
                u_r = FrameDisplRads / RadiaScale * label_map_3d
                u_theta = FrameDisplThta * ThetaScale * label_map_3d

                # Convert polar displacement to Cartesian
                FrameDisplXOrgAdjPlr = u_r * np.cos(Theta) - u_theta * np.sin(Theta)
                FrameDisplYOrgAdjPlr = u_r * np.sin(Theta) + u_theta * np.cos(Theta)

                ### Compute Strain ###

                # Compute displacement gradients
                UXx, UXy = np.gradient(FrameDisplX, deltaX, deltaY, axis=(0, 1))
                UYx, UYy = np.gradient(FrameDisplY, deltaX, deltaY, axis=(0, 1))

                # Compute strain components
                ExxAll = (2 * UXx - (UXx**2 + UYx**2)) / 2
                ExyAll = (UXy + UYx - (UXx * UXy + UYx * UYy)) / 2
                EyyAll = (2 * UYy - (UXy**2 + UYy**2)) / 2


                ThetaEp = 0.5 * np.arctan2(2 * ExyAll, ExxAll - EyyAll)
                Ep1All = (ExxAll + EyyAll) / 2 + np.sqrt(((ExxAll - EyyAll) / 2)**2 + ExyAll**2)
                Ep2All = (ExxAll + EyyAll) / 2 - np.sqrt(((ExxAll - EyyAll) / 2)**2 + ExyAll**2)

                # Compute incompressibility strain component
                Ep3All = 1 / ((1 + Ep1All) * (1 + Ep2All)) - 1

                ### Adjust displacements with ring ###

                FrameDisplXOrgAdjPlrAdj=FrameDisplXOrgAdjPlr.copy()
                FrameDisplYOrgAdjPlrAdj=FrameDisplYOrgAdjPlr.copy()
                # Run the optimized function

                FrameDisplXOrgAdjPlrAdj, FrameDisplYOrgAdjPlrAdj, Ep1All, Ep2All, Ep3All, strain_history = adjust_displacement_with_ring(
                    FrameDisplXOrgAdjPlrAdj, FrameDisplYOrgAdjPlrAdj, deltaX, deltaY, StrainEpPeak,
                    xMat_shft, yMat_shft, inner_radius=inner_radius, outer_radius=outer_radius, strain_tolerance=strain_tolerance, max_iterations=max_iterations)

                ani_masked = animate_deformed_masked_mri(Image, Mask, FrameDisplXOrgAdjPlrAdj, FrameDisplYOrgAdjPlrAdj, output_filename="deformed_mri_masked_polar_rescaled.mp4", save_file=True, save_mode=True, patinet_file_name=npy_file)

                # Count files using dynamic path
                file_count = len([f for f in saved_displacements.glob('*') 
                                if f.is_file() and f.name != ".DS_Store"])
                size = file_count
                print(f"Size: {size}")

print("All done!")