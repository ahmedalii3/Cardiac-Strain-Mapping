# Cine Image Generation Module

## Introduction

This module is a specialized component of a broader medical imaging framework designed to generate dynamic cine images from static magnetic resonance imaging (MRI) frames, specifically focusing on cardiac imaging. It processes a single MRI frame and its corresponding myocardium mask for a specified slice, producing a temporal sequence of deformed images (cine) that simulate cardiac motion. The module leverages wave-based simulation techniques, strain calculations, and polar coordinate transformations to model realistic myocardial deformations, making it suitable for applications in cardiac motion analysis, strain estimation, and validation of biomechanical models. This documentation provides a detailed, technical description of the module's architecture, methodologies, and operational workflow, tailored for scientific paper writing.

## Module Overview

The cine image generation module is a self-contained Python-based system comprising three primary files: `generator.py`, `helper.py`, and `wave_simulation.py`. These files collectively handle configuration management, directory operations, image processing, wave simulation, strain computation, and animation generation. The module operates as a closed system, accepting input in the form of a static MRI frame, a myocardium mask, and configuration parameters, and producing cine sequences stored as NumPy arrays and optional MP4 animations. It is designed to integrate seamlessly into larger medical imaging pipelines, ensuring modularity and reproducibility.

### Objectives
- **Dynamic Cine Generation**: Transform a static MRI frame into a temporal sequence of images simulating cardiac motion.
- **Biomechanical Realism**: Incorporate wave-based displacement fields and strain constraints to model myocardial deformation accurately.
- **Flexibility**: Support customizable simulation parameters via JSON configuration files.
- **Output Versatility**: Generate both numerical data (NumPy arrays) and visual outputs (MP4 animations) for analysis and visualization.

## Module Components

### 1. Configuration and Directory Management (`generator.py`)

#### ConfigLoader Class
The `ConfigLoader` class is responsible for parsing JSON configuration files that define simulation parameters. Two configuration types are supported:
- **Parameters Configuration** (`config_parameters.json`): Specifies simulation parameters such as wind direction, wind speed, number of frames, field of view (FOV), image size, peak strain, and inner/outer radii for strain adjustments. Parameters like wind direction and speed are defined as ranges, allowing randomization within specified bounds to enhance variability in simulations.
- **Generator Configuration** (`config_generator.json`): Defines operational parameters, including the number of cines to generate, patient ID ranges, and data storage directories.

The class employs a modular design, extracting parameters based on the configuration file's context, ensuring flexibility for different simulation scenarios. Randomization of parameters (e.g., frame count, radii) introduces stochasticity, which is critical for generating diverse cine sequences.

#### DirectoryManager Class
The `DirectoryManager` class manages the file system structure, ensuring robust handling of input and output paths. It dynamically constructs paths for patient-specific data, displacement fields, frames, and cine outputs. Key functionalities include:
- **Directory Setup**: Creates necessary directories (`Displacements`, `Frames`, `Cines`) if they do not exist, using the `pathlib` library for cross-platform compatibility.
- **Path Resolution**: Generates paths for patient-specific NumPy files containing MRI frames and masks, following a standardized naming convention (e.g., `patientXXX_frameYY_slice_Z_ACDC.npy`).
- **Data Organization**: Centralizes data storage under a configurable root directory, defaulting to a project-specific `Data/ACDC/train_numpy` folder.

This class ensures that the module can operate within varied directory structures, enhancing its portability across different computational environments.

### 2. Image Processing and Mask Handling (`helper.py`)

The `helper.py` file provides utility functions for loading and preprocessing MRI images and masks, as well as generating smooth, fading masks for deformation applications. These functions are critical for ensuring that input data is correctly formatted and that deformation effects are visually and numerically coherent.

#### Image Loading
The `load_image` function loads MRI frames and myocardium masks from NumPy files. It handles both image and mask data, applying preprocessing steps such as:
- **Data Extraction**: Extracts the appropriate array (image or mask) from the NumPy file, accounting for multi-dimensional arrays.
- **Normalization**: Converts image data to 8-bit unsigned integer format (uint8) using min-max normalization, ensuring compatibility with visualization and deformation pipelines.
- **Color Conversion**: Transforms grayscale images to RGB format for consistent processing, handling cases where input arrays have varying dimensions (e.g., 2D grayscale or 3D with multiple channels).
- **Mask Binarization**: For masks, applies thresholding to ensure binary values (0 or 1), which is essential for subsequent deformation and strain calculations.

The function includes robust error handling for missing files and unexpected array shapes, making it reliable for scientific applications.

#### Mask Dilation and Fading
The module implements multiple mask dilation strategies to create smooth transitions at the myocardium boundaries, which are crucial for realistic deformation effects. These functions include:
- **Dilate Mask**: Iteratively dilates a binary mask using an elliptical kernel, applying a linear fade to create a smooth transition. The fading intensity decreases with each iteration, and a Gaussian filter is applied to enhance smoothness.
- **Dilate Mask Fade**: Uses distance transform and morphological operations (binary and grayscale closing) to generate a fading mask. The fade distance is configurable, allowing control over the transition width.
- **Dilate Mask Fade Cosine**: Employs a cosine-based decay function for fading, combined with pre- and post-blurring via Gaussian filters. This method produces a perceptually smooth transition, ideal for visualization.
- **Dilate Mask Fade Smooth**: Similar to `dilate_mask_fade`, but emphasizes pre- and post-blurring for enhanced smoothness, suitable for numerical stability in strain calculations.

These functions ensure that deformation effects are confined to the myocardium and its immediate vicinity, preventing artifacts in surrounding tissues. The use of distance transforms and Gaussian filtering aligns with established image processing techniques in medical imaging, ensuring scientific rigor.

#### File Existence Check
The `save_if_not_exists` function checks for the existence of output files, preventing overwrites and ensuring data integrity. This is particularly important in batch processing scenarios, where multiple cines are generated iteratively.

### 3. Wave Simulation and Deformation (`wave_simulation.py`)

The `wave_simulation.py` file is the core of the module, implementing the wave-based simulation, strain computation, and deformation pipeline. It leverages the Phillips spectrum for wave generation, polar coordinate transformations for myocardial-specific deformations, and iterative strain adjustments to ensure biomechanical accuracy.

#### Wave Simulation
The wave simulation is based on the Phillips spectrum, a well-established model for ocean wave generation adapted here for myocardial motion. Key components include:

- **StrainWaveParams Class**: A dataclass that encapsulates simulation parameters, including grid size (default 512x512), patch size, wind speed, wind direction, random seed, scaling factor, and gravitational constant. The class ensures reproducibility by setting a random seed post-initialization.
- **Wave Initialization**: The `initialize_wave` function generates initial wave height coefficients (`H0`), deep water frequencies (`W`), and a checkerboard sign grid (`Grid_Sign`). The Phillips spectrum is computed based on wave number magnitudes, wind direction, and speed, modeling energy distribution across wave frequencies.
- **Wave Evolution**: The `calc_wave` function computes the temporal evolution of the wave field using inverse Fourier transforms, incorporating damping and optional random phase shifts to simulate energy loss and realism.
- **Displacement Computation**: The `compute_displacement_frames` function generates displacement fields for X and Y directions, applying Gaussian smoothing and an interaction factor to couple the directions. Displacement values are clipped to ensure physical plausibility.
- **Simulation Orchestration**: The `run_wave_simulation` function orchestrates the simulation, randomizing wind direction and speed within configured ranges, generating displacement fields, and inverting Y-direction displacements to align with anatomical conventions.

The use of the Phillips spectrum, while unconventional for cardiac modeling, provides a robust framework for generating smooth, periodic displacement fields that mimic the cyclic nature of cardiac motion. The randomization of wind parameters introduces variability, simulating inter-patient differences.

#### Strain Computation and Adjustment
Strain calculations are central to ensuring that deformations reflect realistic myocardial mechanics. The module implements the following strain-related functionalities:

- **Strain Calculation**: The `compute_strains` function computes Eulerian strain tensor components (`Exx`, `Exy`, `Eyy`) from displacement field gradients, deriving principal strains (`Ep1`, `Ep2`) and an incompressibility strain (`Ep3`) using the determinant rule. The `enforce_full_principal_strain_order` function ensures that strains are ordered (`Ep1 >= Ep2 >= Ep3`) at each voxel, maintaining physical consistency.
- **Strain Adjustment**: The `adjust_displacement_for_strain` function iteratively scales displacement fields to match a target peak strain (`StrainEpPeak`). It uses a tolerance-based convergence criterion and clips scaling factors to prevent excessive adjustments. The process involves recomputing strains after each scaling iteration, ensuring that the maximum strain across all frames converges to the target.
- **Ring-Based Adjustment**: The `adjust_displacement_with_ring` function refines strain adjustments by applying a ring-shaped mask defined by inner and outer radii. This excludes central and peripheral regions from peak strain calculations, focusing on the myocardial wall where strains are most relevant.
- **Center-Exclusion Adjustment**: The `adjust_displacement_ignore_center` function excludes a circular region around the heart center, further refining strain localization.

These functions employ vectorized operations for efficiency and incorporate numerical safeguards (e.g., avoiding division by zero) to ensure stability. The iterative adjustment process, with a maximum of 20 iterations and a strain tolerance of 0.01, balances computational efficiency with accuracy.

#### Polar Coordinate Transformation
The `PolarConverter` class transforms Cartesian displacement fields into polar coordinates, aligning deformations with the radial and circumferential geometry of the myocardium. Key steps include:
- **Center of Mass Calculation**: Determines the heart center using the myocardium mask's center of mass, adjusted for grid alignment.
- **Coordinate Transformation**: Converts displacement fields to radial (`u_r`) and angular (`u_theta`) components, scaling them based on radial distance and angular position.
- **Logistic Labeling**: Applies a radial logistic function to create a high-pass label map, confining deformations to the myocardial region and ensuring smooth transitions at boundaries.
- **Reconversion**: Transforms polar displacements back to Cartesian coordinates for application to the MRI frame.

This approach ensures that deformations respect the anatomical structure of the heart, with radial and circumferential strains reflecting realistic cardiac mechanics.

#### Animation and Deformation
The module generates both numerical and visual outputs through a series of animation functions:

- **Deformed MRI Animation**: The `animate_deformed_mri` and `animate_deformed_masked_mri` functions apply displacement fields to warp the MRI image and mask, using OpenCV's `remap` function with Lanczos interpolation for high-quality results. The latter function incorporates a fading mask to smooth deformation boundaries and stores deformed images, masks, and masked displacements for all frames.
- **Strain Visualization**: Functions like `animate_strain_histograms`, `animate_wave_displacement`, and `animate_strain_maps` create animations of strain distributions, displacement fields, and strain maps, respectively. These use Matplotlib's `FuncAnimation` with FFmpeg for MP4 output, providing visual insights into the simulation dynamics.
- **Comprehensive Cine Animation**: The `animate_deformation_cines` function produces a multi-panel animation displaying deformed images, masks, displacement fields, strain maps, and fading masks, offering a holistic view of the cine sequence.

These functions ensure that outputs are both numerically precise (stored as NumPy arrays) and visually interpretable (saved as MP4 files), facilitating scientific analysis and presentation.

## Workflow

The module's operational workflow is orchestrated by the `main` function in `generator.py`, which integrates all components into a cohesive pipeline:

1. **Configuration Loading**: Loads simulation and generator parameters from JSON files, initializing randomization for patient IDs, slice numbers, and frame counts.
2. **Directory Setup**: Configures input and output directories, ensuring all necessary paths are created.
3. **Patient and Frame Selection**: Randomly selects a patient ID, frame number, and slice number within configured ranges, ensuring unique combinations to avoid redundancy.
4. **Image and Mask Loading**: Loads the MRI frame and myocardium mask from a NumPy file, preprocessing them for deformation.
5. **Wave Simulation**: Generates displacement fields using the Phillips spectrum, scaling them to match the image's field of view and resolution.
6. **Strain Adjustment**: Applies iterative strain adjustments, first globally and then with a ring-based mask, to align deformations with the target peak strain.
7. **Polar Transformation**: Converts displacements to polar coordinates, applying a logistic label map to focus deformations on the myocardium.
8. **Deformation and Animation**: Warps the MRI image and mask using the adjusted displacement fields, generating a cine sequence and optional MP4 animation.
9. **Output Storage**: Saves the cine sequence as a NumPy file containing deformed images, masks, displacement fields, strain maps, and simulation parameters. Optionally saves individual frames and displacements for detailed analysis.

The workflow is designed for batch processing, generating multiple cines by iterating over different patient-frame-slice combinations. The use of randomization ensures diversity in the generated sequences, while the modular structure allows for easy integration into larger pipelines.

## Technical Considerations

### Performance and Scalability
- **Vectorized Operations**: The module extensively uses NumPy for vectorized computations, minimizing computational overhead in strain calculations and displacement transformations.
- **Memory Efficiency**: Displacement fields and strain maps are stored as 3D arrays, with memory usage optimized by processing one cine at a time.
- **Scalability**: The module supports configurable grid sizes and frame counts, allowing adaptation to different computational resources and imaging resolutions.

### Numerical Stability
- **Gradient Computations**: Strain calculations use NumPy's `gradient` function with spatial step sizes (`deltaX`, `deltaY`), ensuring accurate derivative approximations.
- **Convergence Safeguards**: Iterative strain adjustments include clipping of scaling factors and a maximum iteration limit, preventing numerical divergence.
- **Floating-Point Precision**: Small constants (e.g., `np.finfo(float).eps`) are added to denominators to avoid division-by-zero errors.

### Reproducibility
- Random seeds are set for wave simulations, ensuring reproducible results when desired.
- Configuration files centralize parameter management, facilitating consistent experimental setups.

### Limitations
- **Simplified Biomechanics**: The use of the Phillips spectrum, while effective for generating smooth displacement fields, simplifies the complex viscoelastic properties of myocardial tissue.
- **2D Assumption**: The module operates on 2D slices, neglecting out-of-plane motion, which may limit its applicability to full 3D cardiac modeling.
- **Parameter Sensitivity**: The quality of cine sequences depends on the choice of parameters (e.g., wind speed, strain peak), requiring careful tuning for specific applications.

## Scientific Applications

The module is well-suited for several applications in cardiac imaging research:
- **Strain Analysis**: Provides detailed strain maps (`Ep1`, `Ep2`, `Ep3`) for validating biomechanical models of myocardial deformation.
- **Motion Simulation**: Generates realistic cine sequences for training machine learning models in cardiac motion tracking or segmentation.
- **Data Augmentation**: Produces synthetic cine data to augment limited clinical datasets, enhancing robustness in diagnostic algorithms.
- **Visualization**: Offers high-quality animations for presenting cardiac motion dynamics in scientific publications and conferences.

The module's outputs, including numerical arrays and MP4 animations, are directly usable in scientific workflows, supporting both quantitative analysis and qualitative visualization.

## Conclusion

The cine image generation module is a robust, modular component for transforming static MRI frames into dynamic cine sequences, with a focus on biomechanical realism and scientific applicability. By integrating wave-based simulation, strain computation, polar transformations, and advanced image processing, it provides a comprehensive tool for cardiac motion analysis. Its design emphasizes flexibility, reproducibility, and integration into larger medical imaging frameworks, making it a valuable asset for researchers in biomedical engineering and cardiology. Future enhancements could include 3D modeling, incorporation of patient-specific biomechanical properties, and optimization for real-time processing.