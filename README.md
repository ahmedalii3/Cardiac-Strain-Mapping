# GP-2025-Strain

Investigate novel ideas of unsupervised learning methods for calculating regional cardiac function (displacement and strain)

1. [Supervised Models Framework Instructions](#Supervised-Models-Framework-Instructions)
2. [Unsupervised Models Framework Instructions](#Unsupervised-Models-Framework-Instructions)
3. [Simulator Instructions](#Simulator-Instructions)
4. [Software Instructions](#Software-Instructions)

To update the README file for the **GP-2025-Strain** project to include your contribution to the unsupervised framework, we need to add a section describing the `VoxelMorph_WithMask_and_WithoutMask&SimulatedData_version_Localization.ipynb` framework, its functionality, setup instructions, and integration with the project. Below is a proposed section to add to the README, tailored to your unsupervised framework based on the provided Jupyter notebook. This section assumes it will be added after the existing "Supervised Models Framework Instructions" section and before the "Software Instructions" section.

---

# Supervised Models Framework Instructions

The `Automate_Training` framework automates the training, evaluation, and visualization of image registration models, particularly for medical imaging applications. It supports multiple U-Net-based architectures, processes displacement and frame data, calculates principal strains, and generates detailed visualizations for simulated and real test datasets. This README provides step-by-step instructions for setting up and using the framework.

### Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Data Preparation](#data-preparation)
5. [Running the Framework](#running-the-framework)
6. [Output Description](#output-description)
7. [Troubleshooting](#troubleshooting)
8. [Notes](#notes)

### Overview

The `Automate_Training` class is designed to:

- Load and preprocess image frames and displacement fields from a dataset.
- Train multiple U-Net-based models for image registration using TensorFlow.
- Compute principal strains from displacement fields.
- Visualize results, including loss curves, displacement fields, strain maps, and image comparisons.
- Support both simulated and real test data, with specific handling for patient data.

The framework uses a variety of U-Net and Residual U-Net models, with custom loss and metric functions (`MaskLoss`, `MAELoss`) for enhanced training. It is optimized for GPU usage and includes early stopping and model checkpointing.

### Prerequisites

- **Python Version**: Python 3.8 or higher.
- **Hardware**: A GPU is recommended for faster training (CUDA-compatible for TensorFlow).
- **Dependencies**:
  - `tensorflow>=2.10.0`
  - `numpy`
  - `matplotlib`
  - `opencv-python` (cv2)
  - `scipy`
  - `pathlib`
- **Operating System**: Compatible with Windows, macOS, or Linux.
- **Disk Space**: Ensure sufficient space for datasets, model checkpoints, and output visualizations (several GBs depending on dataset size).

### Installation

1. **Clone the Repository**:

   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Set Up a Virtual Environment** (recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:

   ```bash
   pip install tensorflow numpy matplotlib opencv-python scipy
   ```

4. **Verify TensorFlow GPU Support** (optional):
   Run the following Python code to check for GPU availability:

   ```python
   import tensorflow as tf
   print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
   ```

5. **Model Architectures**:
   Ensure the `Models_Arch` directory is in the same parent directory as the main script (`automate_training.py`). This directory should contain the following model files:

   - `ResidualUnet.py`
   - `Unet.py`
   - `Unet_7Kernel.py`
   - `Unet_5Kernel.py`
   - `Unet_3Dense.py`
   - `Unet_1Dense.py`
   - `Unet_2Dense.py`
   - `Unet_1Dense_7Kernel.py`
   - `Unet_1Dense_5Kernel.py`
   - `Unet_2Dense_7Kernel.py`
   - `Unet_2Dense_5Kernel.py`
   - `Unet_3Dense_7Kernel.py`
   - `Unet_3Dense_5Kernel.py`
   - `ResidualUnet_1Dense.py`
   - `ResidualUnet_2Dense.py`
   - `ResidualUnet_3Dense.py`
   - `ResidualUnet_1Dense_7Kernels.py`
   - `ResidualUnet_1Dense_5Kernels.py`
   - `ResidualUnet_2Dense_7Kernels.py`
   - `ResidualUnet_2Dense_5Kernels.py`
   - `ResidualUnet_3Dense_7Kernels.py`
   - `ResidualUnet_3Dense_5Kernels.py`
   - `mask_loss.py` (contains `MaskLoss` and `MAELoss` implementations)

   If any model file is missing, you may need to implement or remove the corresponding model from the `models_list` in the main script.

### Data Preparation

The framework expects a specific directory structure for the dataset and real test data.

### Dataset Directory (`Data/`)

- **Structure**:
  ```
  Data/
  ├── Displacement/
  │   ├── <id>_x.npy  # X-component displacement fields
  │   ├── <id>_y.npy  # Y-component displacement fields
  ├── Frames/
  │   ├── <id>_1.npy  # Moving images (first frame)
  │   ├── <id>_2.npy  # Fixed images (second frame)
  ```
- **Requirements**:
  - Files must be in `.npy` format (NumPy arrays).
  - Displacement fields (`<id>_x.npy`, `<id>_y.npy`) and frames (`<id>_1.npy`, `<id>_2.npy`) must share the same `<id>` for matching.
  - Images and displacement fields should be 128x128 pixels with a single channel (grayscale).
  - The dataset is split automatically into training (90%), validation (5%), and test (5%) sets.

### Real Test Data Directory (`real_test_data/`)

- **Structure**:
  ```
  real_test_data/
  ├── patient_4d_frame_1.npy        # Moving image
  ├── patient_4d_frame_13.npy       # Fixed image
  ├── patient_4d_frame_13_mask.npy  # Mask image
  ```
- **Requirements**:
  - Files must be in `.npy` format.
  - The moving and fixed images should be 128x128 pixels, single-channel.
  - The mask image should contain labels (e.g., 0 for background, 1 for myocardium, 2 for other regions).

### Notes

- Ensure no `.DS_Store` files are present in the directories (common on macOS).
- The framework assumes images are not normalized; uncomment the normalization code in `load_data` if your images require scaling to [0, 1].
- Verify that the mask file is correctly formatted and aligns with the images.

### Running the Framework

1. **Prepare the Script**:
   Ensure `automate_training.py` is in the project directory, and the `Models_Arch` directory is correctly placed.

2. **Update Paths** (if needed):
   Modify the following lines in `automate_training.py` to match your dataset and output directories:

   ```python
   dataset_path = "Data"
   real_test_data_path = "real_test_data"
   save_dir = current_script.parent / "Saved"
   saved_model_dir = current_script.parent / "Models"
   ```

3. **Run the Script**:
   Execute the script from the command line:

   ```bash
   python automate_training.py
   ```

4. **Training Parameters**:
   The script uses default hyperparameters:

   - `num_epochs=120`
   - `batch_size=32`
   - Optimizer: Adam with learning rate 0.0001 and gradient clipping (clipvalue=1.0)
   - Loss: Mean Squared Error (MSE)
   - Metric: Mean Absolute Error (MAE)
   - Callbacks: ModelCheckpoint (saves best model based on validation loss) and EarlyStopping (stops training if validation loss does not improve for 20 epochs)

   To modify these, update the `train_models` call:

   ```python
   trainer.train_models(num_epochs=120, batch_size=32)
   ```

5. **Model Selection**:
   The script trains a list of U-Net and Residual U-Net variants. To train a subset, modify the `models_list`:
   ```python
   models_list = [Unet(), Residual_Unet()]  # Example: Train only Unet and Residual_Unet
   ```

### Output Description

The framework generates outputs in two directories: `Saved/` and `Models/`.

### `Saved/`

Contains visualization outputs and loss data for each model:

- **Loss Plots**: `loss_plot<model_name>.png` shows training and validation loss curves.
- **Evaluation Results**: `evaluation_results<model_name>.txt` contains test loss and MAE.
- **Loss Data**: `losses<model_name>.npy` and `val_losses<model_name>.npy` store training and validation loss arrays.
- **Visualizations**:
  - **Real Test Data**:
    - `warped_image_real_test<model_name>.png`: Shows fixed, moving, warped images, original difference, and final difference with mask overlay.
    - `sample_<model_name>_analysis.png`: Detailed analysis with core images, strain heatmaps, and overlays.
  - **Training Data** (sample 992):
    - `direction_plot_train<model_name>.png`: Displacement field vectors (actual, predicted, difference).
    - `magnitude_plot_train<model_name>.png`: Displacement magnitude heatmaps (actual, predicted, difference).
    - `warped_image_train<model_name>.png`: Fixed, moving, warped images, and predicted displacement field.
    - `sample_<model_name>_train_analysis.png`: Detailed analysis for training sample.
  - **Test Data** (sample 67):
    - `direction_plot_test<model_name>.png`: Displacement field vectors.
    - `magnitude_plot_test<model_name>.png`: Displacement magnitude heatmaps.
    - `warped_image_test<model_name>.png`: Fixed, moving, warped images, and predicted displacement field.
    - `sample_<model_name>_test_analysis.png`: Detailed analysis for test sample.
    - `displacement_plot_test_simulated<model_name>.png`: X-displacement overlay on moving image.

### `Models/`

Contains trained model checkpoints:

- `<model_name>.keras`: Best model saved based on validation loss.

### Troubleshooting

- **Missing Model Files**: Ensure all model files are in `Models_Arch/`. If a model fails to import, remove it from `models_list` or implement the missing architecture.
- **Data Loading Errors**:
  - Verify that `Data/` and `real_test_data/` directories exist and follow the required structure.
  - Check for corrupted `.npy` files or mismatched IDs between frames and displacements.
- **GPU Issues**: If no GPU is detected, ensure CUDA and cuDNN are installed, or run on CPU (slower).
- **Visualization Issues**: Ensure `matplotlib` is installed. If plots fail to save, check write permissions in `Saved/`.
- **Out-of-Memory Errors**: Reduce `batch_size` or use a smaller subset of models.
- **Custom Loss/Metric Errors**: Ensure `mask_loss.py` contains valid `MaskLoss` and `MAELoss` implementations.

### Notes

- The framework assumes images are 128x128 pixels. Adjust the input shape in `train_models` if your data differs:
  ```python
  fixed_input = Input(shape=(height, width, 1), name="fixed_image")
  moving_input = Input(shape=(height, width, 1), name="moving_image")
  ```
- The commented-out normalization in `load_data` (`image = image / 255.0`) may be needed depending on your data’s intensity range.
- The `limit_strain_range` method returns redundant `None` values, which may be a placeholder for future functionality.
- The framework uses `cv2.INTER_LANCZOS4` for image warping, which is suitable for high-quality interpolation but may be computationally intensive.
- For large datasets, consider preprocessing data to reduce memory usage or using a data generator for training.

---

# Unsupervised Models Framework Instructions

The `VoxelMorph_WithMask_and_WithoutMask&SimulatedData_version_Localization` framework implements an unsupervised learning approach for image registration, specifically tailored for calculating regional cardiac function (displacement and strain) using the VoxelMorph library. It supports both masked and unmasked configurations, various kernel sizes, and hyperparameter tuning for the loss function's regularization parameter (lambda). The framework is designed to work with simulated and real cardiac imaging datasets, such as ACDC and Sunnybrook, and includes comprehensive visualization capabilities, including video generation for displacement and strain analysis.

### Table of Contents

1. [Overview](#overview-unsupervised)
2. [Prerequisites](#prerequisites-unsupervised)
3. [Installation](#installation-unsupervised)
4. [Data Preparation](#data-preparation-unsupervised)
5. [Running the Framework](#running-the-framework-unsupervised)
6. [Output Description](#output-description-unsupervised)
7. [Troubleshooting](#troubleshooting-unsupervised)
8. [Notes](#notes-unsupervised)

### Overview

The unsupervised framework leverages VoxelMorph, a deep learning-based image registration tool, to perform deformable registration on cardiac images without requiring ground-truth displacement fields during training. Key features include:

- **Unsupervised Registration**: Uses VoxelMorph's unsupervised learning approach with similarity-based loss functions (e.g., MSE) and regularization terms for smooth deformation fields.
- **Mask Support**: Supports configurations with and without masks for mean squared error (MSE) and smoothness loss, enabling targeted registration on regions of interest (e.g., myocardium).
- **Kernel Configurations**: Tests multiple U-Net kernel sizes (`default`, `first5`, `first7_second5`) to evaluate their impact on registration performance.
- **Hyperparameter Tuning**: Tests a range of lambda values (0.1 to 1.0) for the regularization term to balance image similarity and deformation smoothness.
- **Visualization and Video Generation**: Generates detailed visualizations, including warped images, displacement fields, and strain maps, with optional video output for dynamic analysis.

The framework is optimized for both local and Google Colab environments and supports GPU acceleration for efficient training and inference.

### Prerequisites

- **Python Version**: Python 3.10 or higher (based on the notebook's kernel version).
- **Hardware**: A GPU is highly recommended for faster training (CUDA-compatible for TensorFlow).
- **Dependencies**:
  - `tensorflow==2.19.0`
  - `voxelmorph==0.2`
  - `neurite==0.2`
  - `opencv-python==4.11.0`
  - `scikit-image==0.25.2`
  - `matplotlib==3.10.1`
  - `tqdm==4.67.1`
  - `scipy==1.15.2`
- **Operating System**: Compatible with Windows, macOS, or Linux.
- **Disk Space**: Ensure sufficient space for datasets, model checkpoints, and output visualizations (several GBs depending on dataset size).
- **Optional for Colab**: Google Drive for storing datasets and models if running on Google Colab.

### Installation

1. **Clone the Repository**:

   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Set Up a Virtual Environment** (recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   Install the required packages:

   ```bash
   pip install tensorflow==2.19.0 git+https://github.com/adalca/neurite.git@v0.2 git+https://github.com/voxelmorph/voxelmorph.git@v0.2 opencv-python==4.11.0 scikit-image==0.25.2 matplotlib==3.10.1 tqdm==4.67.1 scipy==1.15.2
   ```

4. **Custom VoxelMorph Installation** (if using custom VoxelMorph):
   If `USE_CUSTOM_VXM=True` in the notebook, ensure the custom VoxelMorph code is placed in `./data/voxelmorph`. Update the `CUSTOM_VXM_PATH` variable in the notebook if necessary:

   ```python
   CUSTOM_VXM_PATH = './data/voxelmorph'
   ```

5. **Verify TensorFlow GPU Support** (optional):
   Run the following Python code to check for GPU availability:

   ```python
   import tensorflow as tf
   print("GPU Devices:", tf.config.list_physical_devices('GPU'))
   ```

6. **Set Neurite Backend**:
   Ensure the `NEURITE_BACKEND` environment variable is set to `tensorflow` before running the notebook:
   ```python
   os.environ['NEURITE_BACKEND'] = 'tensorflow'
   ```

### Data Preparation

The framework expects a specific directory structure for both simulated and real datasets.

#### Directory Structure (`./data/`)

- **Simulated Data**:
  ```
  data/
  ├── Simulated_train/
  ├── Simulated_val/
  ├── Simulated_test/
  ├── Simulated_masks/
  ├── Simulated_displacements/
  ```
- **Real Data**:

  ```
  data/
  ├── train/
  ├── val/
  ├── test/
  ├── ACDC-Masks-1/
  ├── model_testing/
  ```

- **Requirements**:
  - Files must be in `.npy` format (NumPy arrays).
  - Images and displacement fields should be 128x128 pixels with a single channel (grayscale).
  - File names must follow the naming convention (e.g., `<id>_1.npy` for moving images, `<id>_2.npy` for fixed images, `<id>_mask.npy` for masks).
  - Ensure matching `<id>` across frames, masks, and displacement fields.
  - Download the dataset from the provided links:
    - Data: [data.zip](https://drive.google.com/file/d/1QJDP_EI2VUi5MdoDIQsH-LcafTM9WyNo/view?usp=sharing)
    - Preprocessed training data: [train_data.h5, val_data.h5, test_data.h5](https://drive.google.com/file/d/1qctG6eUOC3RplJY-2zlMGGTYP_sTV8fB/view?usp=sharing)

### Running the Framework

1. **Prepare the Notebook**:
   Ensure the `VoxelMorph_WithMask_and_WithoutMask&SimulatedData_version_Localization.ipynb` file is in the project directory, and the `data/` directory is correctly set up.

2. **Update Paths** (if needed):
   Modify the `LOCAL_DATA_DIR` variable in the notebook to match your dataset location:

   ```python
   LOCAL_DATA_DIR = "./data"
   ```

   If running on Google Colab, update the Colab-specific paths:

   ```python
   BASE_DATA_PATH = '/content/drive/My Drive/GP_Data_Folder/GP_Data-Sets'
   MODELS_BASE_PATH = '/content/drive/My Drive/GP_Data_Folder/Models'
   ```

3. **Configure Environment**:
   Set `RUNNING_ON_COLAB` to `True` for Google Colab or `False` for local execution:

   ```python
   RUNNING_ON_COLAB = False
   ```

4. **Run the Notebook**:
   Open the notebook in Jupyter or Google Colab and execute all cells. Key steps include:

   - **Environment Setup**: Installs dependencies and verifies package versions.
   - **Data Path Configuration**: Sets paths for simulated and real datasets, checking for their existence.
   - **Model Configuration**: Defines VoxelMorph models with different mask and kernel configurations, and lambda values.
   - **Visualization**: Generates visualizations and optional videos for selected patient data (e.g., patient "007").

5. **Model Configurations**:
   The notebook defines multiple model configurations in `MODEL_CONFIG`:

   - `no_mask`: Baseline VoxelMorph without masks.
   - (Commented out) `mse_mask`, `smoothness_mask`, `both_masks`: Configurations with MSE and/or smoothness masks.
     To enable additional configurations, uncomment the relevant sections in `MODEL_CONFIG`.

6. **Kernel Configurations**:
   The framework tests different U-Net kernel sizes:

   - `default`: All layers use 3x3 kernels.
   - `first5`: First encoder layer uses 5x5 kernels, others 3x3.
   - `first7_second5`: First encoder layer uses 7x7 kernels, second uses 5x5, others 3x3.
     Modify `KERNEL_CONFIGS` to add or remove configurations.

7. **Hyperparameter Tuning**:
   The framework tests lambda values (`LAMBDAS`) from 0.1 to 1.0 for regularization. Adjust the `LAMBDAS` list to test different values:
   ```python
   LAMBDAS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
   ```

### Output Description

The framework generates outputs in the `Models/` directory, organized by model configuration, kernel size, and lambda value (e.g., `Models/voxelmorph_no_mask_kernel_default_lambda_0.100/`).

#### Directory Structure

```
Models/
├── voxelmorph_no_mask_kernel_<kernel_key>_lambda_<lambda>/
│   ├── weights/  # Model checkpoints
│   ├── results/  # Visualization outputs
│   ├── logs/     # Training logs
```

#### Outputs

- **Model Checkpoints**: Saved in `weights/` as `.h5` or `.keras` files.
- **Visualizations**: Saved in `results/`:
  - Warped images, displacement fields, and strain maps for simulated test data.
  - Comparison of fixed, moving, and warped images with mask overlays.
  - Vector and magnitude plots for displacement fields.
- **Videos**: If `create_video=True`, videos (e.g., `.mp4` or `.gif`) showing dynamic registration and strain evolution for specified patients (e.g., patient "007").
- **Logs**: Training and evaluation metrics saved in `logs/`.

### Troubleshooting

- **Missing Dependencies**: Ensure all required packages are installed with the specified versions. Use `pip install` to install missing packages manually.
- **Data Path Errors**:
  - Verify that all paths in `paths_to_check` exist and contain the required `.npy` files.
  - Check for consistent naming across frames, masks, and displacement fields.
- **GPU Issues**: If no GPU is detected, ensure CUDA and cuDNN are installed, or run on CPU (slower). Check the environment summary output for GPU availability.
- **Custom VoxelMorph Errors**: If using `USE_CUSTOM_VXM=True`, ensure the custom VoxelMorph code is correctly placed in `CUSTOM_VXM_PATH`.
- **Visualization/Video Issues**: Ensure `matplotlib`, `opencv-python`, and `imageio` (for video generation) are installed. Check write permissions in the output directories.
- **Memory Errors**: Reduce the number of models or lambda values, or use a smaller batch size if memory issues occur during training.

### Notes

- The framework assumes images are 128x128 pixels. Adjust the VoxelMorph model input shape if your data differs.
- The notebook supports both local and Google Colab environments, with conditional paths for flexibility.
- The commented-out mask configurations (`mse_mask`, `smoothness_mask`, `both_masks`) can be enabled for additional experiments.
- Video generation requires significant disk space and computational resources; set `create_video=False` for faster execution.
- The dataset links provided must be downloaded and extracted to the `data/` directory:
  - [data.zip](https://drive.google.com/file/d/1QJDP_EI2VUi5MdoDIQsH-LcafTM9WyNo/view?usp=sharing)
  - [Preprocessed data](https://drive.google.com/file/d/1qctG6eUOC3RplJY-2zlMGGTYP_sTV8fB/view?usp=sharing)

---
# Simulator Instructions
Great — here’s the updated README with a new section titled Configuration Setup that explains how to edit the JSON files to fit your dataset path and simulation preferences.

⸻

Simulator Instructions

The Cine Image Generation Module transforms static 2D cardiac MRI frames and myocardium masks into dynamic cine sequences that simulate realistic cardiac motion. It uses wave-based simulation (via the Phillips spectrum), polar coordinate transformations, and biomechanically inspired strain modeling to generate deformed sequences useful for motion analysis, strain computation, and model validation.

This module supports batch generation using configuration files and outputs numerical data (NumPy arrays) and optional MP4 animations. It is optimized for scientific workflows and can be integrated into larger medical imaging pipelines.

⸻

Table of Contents


•	[Overview](###Overview)

•	[Prerequisites](###Prerequisites)

•	[Installation](###Installation)

•	[Data Preparation](###Data-Preparation)

•	[Configuration Setup](###Configuration-Setup)

•	[Running the Module](###Running-the-Module)

•	[Output Description](###Output-Description)

•	[Troubleshooting](###Troubleshooting)

•	[Notes](###Notes)


⸻

### Overview

Key Features of the Cine Generator:
	•	Dynamic Cine Simulation: Converts a single 2D MRI frame into a sequence of deformed frames to simulate cardiac motion over time.
	•	Biomechanical Realism: Uses wave-based displacement fields and strain tuning to mimic physiological deformation.
	•	Flexible Configuration: Control simulation behavior via JSON config files.
	•	Polar Transformations: Displacement fields are aligned with myocardial anatomy through radial and angular mapping.
	•	Rich Visualization: Supports generation of MP4 animations, strain maps, and displacement field visualizations.
	•	Scientific Use-Case Ready: Designed for research involving cardiac strain analysis, motion simulation, and data augmentation.

⸻

### Prerequisites

•	Python: 3.9 or higher recommended
•	Recommended Hardware: CPU sufficient, GPU not required but can speed up animation rendering
•	Python Dependencies:
•	numpy
•	scipy
•	matplotlib
•	opencv-python
•	tqdm
•	imageio
•	json
•	pathlib
•	scikit-image (for optional mask processing)

You can install the required packages using:

```bash
pip install -r requirements.txt
```


⸻

### Installation
1.	Clone the Repository:

```bash
git clone <repository-url>
cd <repository-directory>
```

2.	(Optional) Create a Virtual Environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3.	Install Dependencies:

```bash
pip install numpy scipy matplotlib opencv-python tqdm imageio scikit-image
```

Note: Ensure that FFmpeg is installed on your system for MP4 animation support (used by Matplotlib’s FuncAnimation).

⸻

### Data Preparation
1.	Input Files:
	•	Static 2D MRI frame (NumPy .npy file)
	•	Corresponding binary myocardium mask (NumPy .npy file)
	•	Configurations:
	•	config_parameters.json: Controls simulation physics and strain behavior
	•	config_generator.json: Controls batch generation (number of patients, slice/frame selection, output paths)
2.	File Naming Convention:
	•	MRI frames and masks should follow the format:

patientXXX_frameYY_slice_Z_ACDC.npy


	•	Must include both image and mask arrays.



⸻

### Configuration Setup

Before running the module, you must configure two JSON files to match your dataset and desired simulation parameters:

📄 config_generator.json

This file controls generation logic such as patient ID range, number of cines to generate, and output preferences.

{
  "generator": {
    "folder": "/Users/XXX/GP-2025-Strain/Data/ACDC/train_numpy",  // <== Set this to your dataset path
    "no_of_cines": 5,
    "patients_start": 1,
    "patients_end": 100,
  }
}

Make sure the "folder" path points to the directory where your patient .npy files are located.

⸻

📄 config_parameters.json

This file defines simulation parameters like wave speed, target strain, and biomechanical constraints.

{
    "num_frames": [15, 25],          // Random range for number of frames in cine
    "wind_speed": [4.0, 6.0],        // Range of wave wind speed
    "wind_dir": [0, 360],            // Wind direction range
  "strain_validation": {
    "StrainEpPeak": 0.2,             // Target max strain in myocardium  
    "inner_radius": 50,      // Inner ring radius for strain focus
    "outer_radius": 110,     // Outer ring radius for strain focus
    }
  
}

You can modify these ranges to generate different dynamics across cines.

⸻

### Running the Module

The main script is generator.py. To run the full cine generation pipeline:

```bash 
python generator.py
```

This executes the following steps:
	1.	Loads configurations from JSON files.
	2.	Selects a random patient/frame/slice.
	3.	Loads and preprocesses the MRI image and myocardium mask.
	4.	Runs wave-based simulation to generate displacement fields.
	5.	Applies iterative strain adjustment for biomechanical realism.
	6.	Transforms displacements into polar coordinates for anatomical accuracy.
	7.	Warps images and generates output sequences.
	8.	Saves all data as .npy files and optionally as .mp4 animations.

⸻

### Output Description

Each run generates the following outputs per simulation:
	•	Deformed_Frames(.npy): Cine sequence of deformed MRI images.
	•	Displacement_Fields(.npy): 3D array of X/Y displacements across frames.
	•	Information dictionary(.npy): Principal strain maps (Ep1, Ep2, Ep3) per frame and the randomized and final simulation settings used for traceability..
	•	Mask_Animations(.mp4): (optional) Video showing the warped myocardium across time.
	•	Wave_Animation(.mp4): (optional) Displacement field evolution over time.

⸻

### Troubleshooting

Issue	Solution
Missing or malformed NumPy files:	Ensure both image and mask are correctly formatted .npy files.
No animation output:	Check if FFmpeg is installed and accessible.
Strain does not converge:	Tune peak strain, radii, or increase the max iterations in config.
Slow processing:	Reduce frame count or grid size.
Artifacts in deformations:	Use smoother fading masks or increase blur in helper.py.


⸻

### Notes
	•	Extensibility: You can adapt the module to 3D in the future by expanding the deformation model and applying it slice-by-slice or volume-wise.
	•	Use in Research: Outputs are directly usable for ML training, strain validation, or as augmentation in segmentation pipelines.
	•	Randomization: For batch processing and dataset variability, use the generator config to specify number of cines, patient range, and random seeds.

⸻
# Software Instructions

### Prerequisites

Ensure you have the following installed on your system:

- **Node.js** (for the React front end)
- **Python 3.x** (for the FastAPI back end)

### Getting Started

Follow the steps below to run the project locally:

1. Clone the project repository.
2. Navigate to the project directory.
3. Navigate to the backend directory.
4. Create and activate a virtual environment by running this command:

- on macOS / Linux:

```bash
python3 -m venv venv
source venv/bin/activate


```

- on Windows:

```bash
python -m venv venv
.\venv\Scripts\activate

```

5. Install the required dependencies:

```bash
pip install -r requirements.txt
```

6. Create Models Directory

```bash
mkdir models
```

7. Navigate to app folder

8. Create nnUNet folder

```bash
mkdir nnUNet
```

9. Start the FastAPI Server

```bash
uvicorn app.main:app --reload
```

10. Add models files from this link: "https://drive.google.com/drive/folders/1ThnqA72XbFMIDNcIPwRdy3DpWPfEAfR7?usp=drive_link" to the models directory
11. Add nnUNet files from this link: "https://drive.google.com/drive/folders/1n-tPsFCArx0fRO4h6B71shyIeq-271Rl?usp=sharing" to its directory

12. Navigate to the frontend directory
13. Install the required dependencies:

```bash
npm install
```

14. Start the React development server:

```bash
npm run dev
```
