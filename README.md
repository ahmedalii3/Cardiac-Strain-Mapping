# GP-2025-Strain
Investigate novel ideas of unsupervised learning methods for calculating regional cardiac function (displacement and strain)

## Supervised Models Framework Instructions

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
## Software Instructions


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
