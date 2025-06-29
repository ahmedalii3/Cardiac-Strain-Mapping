

class test_model:
    def __init__(self, model_config, kernel_key, lambda_val, model_name, batch_size=8):
        self.model_config = model_config
        self.kernel_key = kernel_key
        self.lambda_val = lambda_val
        self.model_name = model_name
        self.batch_size = batch_size
        
        # Initialize data containers
        self.x_displacement_label = {}
        self.y_displacement_label = {}
        self.x_displacement_predicted = {}
        self.y_displacement_predicted = {}
        self.E1_predicted = {}
        self.E2_predicted = {}
        self.E1_label = {}
        self.E2_label = {}
        self.E1_loss = {}
        self.E2_loss = {}
        self.E1_std = {}
        self.E2_std = {}
        self.E1_first_quartile = {}
        self.E2_first_quartile = {}
        self.E1_second_quartile = {}
        self.E2_second_quartile = {}
        
        # Configure strain range for analysis
        self.strain_loss_ranges_E1 = [round(x, 2) for x in np.arange(-0.3, 0.35, 0.05).tolist()]
        self.strain_loss_ranges_E2 = [round(x, 2) for x in np.arange(-0.3, 0.35, 0.05).tolist()]
        
        # Setup testing
        self.setup_test_data()
        self.load_model()
        self.run_evaluation()
    
    def setup_test_data(self):
        """Setup test data paths and data loader."""
        # Define test data paths based on global variables
        test_paths = {'test': [SIMULATED_DATA_PATH]}
        
        # Create simulated test data loader
        self.data_loader = SimulatedTestDataLoader(
            test_simulated_data_paths=test_paths,
            simulated_mask_root_path=SIMULATED_MASK_PATH,
            simulated_displacement_path=SIMULATED_DISP_PATH,
            batch_size=self.batch_size,
            use_mask=(self.model_config['use_mask'])
        )
        
        print(f"Test data loaded: {self.data_loader.test_pairs_count} pairs")
    
    def load_model(self):
        """Load the VoxelMorph model for evaluation."""
        self.model = load_model_for_eval(
            self.model_config,
            self.kernel_key,
            self.lambda_val,
            load_best=True
        )
        
        if self.model is None:
            raise ValueError(f"Failed to load model: {self.model_name}")
        
        print(f"Model loaded: {self.model_name}")
    
    def run_evaluation(self):
        """Run evaluation pipeline."""
        self.predict_displacements()
        self.calculate_strain()
        self.calculate_MSE_E1()
        self.calculate_MSE_E2()
        self.plot_results()
    
    def predict_displacements(self):
        """Predict displacements using the test data generator."""
        print("Predicting displacements...")
        
        generator = self.data_loader.get_generator()
        file_idx = 0
        
        for (moving, fixed), (orig_fixed, mask, target_disp) in generator:
            # Generate predictions
            pred = self.model.predict([moving, fixed])
            # If model returns [warped, flow], use flow (usually pred[1])
            if isinstance(pred, list):
                pred_disp = pred[1]
            else:
                pred_disp = pred
            self.x_displacement_predicted[file_code] = pred_disp[b, ..., 0]
            
            # Process each item in batch
            for b in range(moving.shape[0]):
                if b + file_idx >= self.data_loader.test_pairs_count:
                    # Skip padding samples
                    continue
                
                # Create unique identifier for this sample
                file_code = f"sample_{file_idx + b:04d}"
                
                # Store ground truth displacements
                self.x_displacement_label[file_code] = target_disp[b, ..., 0]
                self.y_displacement_label[file_code] = target_disp[b, ..., 1]
                
                # Store predicted displacements
                self.x_displacement_predicted[file_code] = pred_disp[b, ..., 0]
                self.y_displacement_predicted[file_code] = pred_disp[b, ..., 1]
            
            file_idx += moving.shape[0]
            print(f"Processed {file_idx} / {self.data_loader.test_pairs_count} samples", end="\r")
        
        print(f"\nPredicted displacements for {len(self.x_displacement_predicted)} samples")
    
    def enforce_full_principal_strain_order(self, Ep1All, Ep2All, Ep3All=None):
        """
        Ensure Ep1All >= Ep2All >= Ep3All at every voxel (pixel) location.
        Sorts the three principal strains per point.

        Args:
            Ep1All (np.ndarray): First principal strain field.
            Ep2All (np.ndarray): Second principal strain field.
            Ep3All (np.ndarray): Third principal strain field (incompressibility strain).

        Returns:
            Ep1_sorted (np.ndarray): Largest principal strain.
            Ep2_sorted (np.ndarray): Middle principal strain.
            Ep3_sorted (np.ndarray): Smallest principal strain.
        """
        if Ep3All is not None:
            # Stack all principal strains along a new axis
            strain_stack = np.stack([Ep1All, Ep2All, Ep3All], axis=0)  # Shape (3, H, W)
        else:
            # Stack only the first two principal strains
            strain_stack = np.stack([Ep1All, Ep2All, Ep2All], axis=0)  # Shape (3, H, W)
            
        # Sort along the new axis (axis=0) descending
        strain_sorted = np.sort(strain_stack, axis=0)[::-1, ...]  # Reverse to get descending

        Ep1_sorted = strain_sorted[0]
        Ep2_sorted = strain_sorted[1]
        Ep3_sorted = strain_sorted[2]

        return Ep1_sorted, Ep2_sorted, Ep3_sorted

    def limit_strain_range(self, FrameDisplX, FrameDisplY, deltaX=1, deltaY=1):
        """
        Compute principal strains (Ep1, Ep2) and incompressibility strain (Ep3) 
        from displacement fields.

        Args:
            FrameDisplX (np.ndarray): X displacement field (shape: H, W).
            FrameDisplY (np.ndarray): Y displacement field (shape: H, W).
            deltaX (float): Pixel spacing in the X direction (mm).
            deltaY (float): Pixel spacing in the Y direction (mm).

        Returns:
            Strain tensors and strain ranges
        """
        final_tensor = {}
        
        # Compute spatial gradients
        UXx, UXy = np.gradient(FrameDisplX, deltaX, deltaY, axis=(0, 1))
        UYx, UYy = np.gradient(FrameDisplY, deltaX, deltaY, axis=(0, 1))

        # Compute Eulerian strain tensor components
        ExxAll = (2 * UXx - (UXx**2 + UYx**2)) / 2
        ExyAll = (UXy + UYx - (UXx * UXy + UYx * UYy)) / 2
        EyyAll = (2 * UYy - (UXy**2 + UYy**2)) / 2

        # Compute principal strains
        Ep1All = (ExxAll + EyyAll) / 2 + np.sqrt(((ExxAll - EyyAll) / 2) ** 2 + ExyAll ** 2)
        Ep2All = (ExxAll + EyyAll) / 2 - np.sqrt(((ExxAll - EyyAll) / 2) ** 2 + ExyAll ** 2)

        # Enforce principal strain ordering
        Ep1All, Ep2All, _ = self.enforce_full_principal_strain_order(Ep1All, Ep2All)

        # Compute incompressibility strain using the determinant rule
        Ep3All = 1 / ((1 + np.maximum(Ep1All, Ep2All)) * (1 + np.minimum(Ep1All, Ep2All))) - 1

        # Store strain results
        final_tensor['E1'] = Ep1All
        final_tensor['E2'] = Ep2All
        final_tensor['E3'] = Ep3All

        return None, None, final_tensor, final_tensor, np.max(Ep1All), np.max(Ep2All), np.min(Ep1All), np.min(Ep2All)

    def calculate_strain(self):
        """Calculate strains from displacements for both predicted and ground truth."""
        print("Calculating strains...")
        
        for file_code in self.x_displacement_predicted.keys():
            # Get displacement fields
            x_pred = self.x_displacement_predicted[file_code]
            y_pred = self.y_displacement_predicted[file_code]
            x_label = self.x_displacement_label[file_code]
            y_label = self.y_displacement_label[file_code]
            
            # Get mask from the data loader for this sample (if available)
            mask = None
            # Try to find this sample in the generator to get its mask
            for (moving, fixed), (orig_fixed, masks, target_disp) in self.data_loader.get_generator():
                for b in range(moving.shape[0]):
                    if file_code == f"sample_{b:04d}":  # If we found the matching sample
                        mask = masks[b, ..., 0]  # First channel is fixed mask
                        break
                if mask is not None:
                    break
            
            # Use default mask if not found
            if mask is None:
                mask = np.ones_like(x_pred)
            
            # Calculate strains from predicted displacements
            _, _, _, final_strain_tensor_pred, _, _, _, _ = self.limit_strain_range(x_pred, y_pred)
            E1_pred = final_strain_tensor_pred['E1'] * mask
            E2_pred = final_strain_tensor_pred['E2'] * mask
            
            # Calculate strains from ground truth displacements
            _, _, _, final_strain_tensor_label, _, _, _, _ = self.limit_strain_range(x_label, y_label)
            E1_label = final_strain_tensor_label['E1'] * mask
            E2_label = final_strain_tensor_label['E2'] * mask
            
            # Store results
            self.E1_predicted[file_code] = E1_pred
            self.E2_predicted[file_code] = E2_pred
            self.E1_label[file_code] = E1_label
            self.E2_label[file_code] = E2_label
        
        print(f"Calculated strains for {len(self.E1_predicted)} samples")

    def calculate_MSE_E1(self):
        """Calculate MSE for E1 strain ranges."""
        print("Calculating E1 errors...")
        
        all_E1_labelled = []
        all_E1_predicted = []
        all_masks = []

        # Collect all data
        for file_code in self.x_displacement_predicted.keys():
            all_E1_labelled.append(self.E1_label[file_code])
            all_E1_predicted.append(self.E1_predicted[file_code])
            
            # Try to get mask from the data loader
            mask = None
            for (moving, fixed), (orig_fixed, masks, _) in self.data_loader.get_generator():
                for b in range(moving.shape[0]):
                    if file_code == f"sample_{b:04d}":
                        mask = masks[b, ..., 0]
                        break
                if mask is not None:
                    break
            
            if mask is None:
                mask = np.ones_like(self.E1_label[file_code])
            
            all_masks.append(mask)

        all_E1_labelled = np.array(all_E1_labelled)
        all_E1_predicted = np.array(all_E1_predicted)
        all_masks = np.array(all_masks)
        length = all_E1_labelled.shape[0]
        
        print(f"Analyzing E1 strain in {length} samples across {len(self.strain_loss_ranges_E1)} strain ranges")

        for rang in self.strain_loss_ranges_E1:
            total_error = 0.0
            total_pixels = 0
            all_errors = []

            for i in range(length):
                label = all_E1_labelled[i]
                pred = all_E1_predicted[i]
                mask = all_masks[i]

                # Apply strain range mask (only in regions where strain is in this range)
                label_mask = (label > rang - 0.025) & (label < rang + 0.025) & (mask > 0)

                # Extract the values where mask applies
                label_values = label[label_mask]
                pred_values = pred[label_mask]
                
                active_pixel_count = len(label_values)

                if active_pixel_count > 0:
                    # Calculate absolute errors
                    error = np.abs(label_values - pred_values)
                    all_errors.extend(error.tolist())
                    total_pixels += active_pixel_count

            # Compute statistics if we have data
            if all_errors:
                self.E1_loss[rang] = np.median(all_errors)
                self.E1_std[rang] = np.std(all_errors)
                self.E1_first_quartile[rang] = np.percentile(all_errors, 25)
                self.E1_second_quartile[rang] = np.percentile(all_errors, 75)
            else:
                self.E1_loss[rang] = 0.0
                self.E1_std[rang] = 0.0
                self.E1_first_quartile[rang] = 0.0
                self.E1_second_quartile[rang] = 0.0
                
            print(f"E1 Range {rang:.2f}: Median Error = {self.E1_loss[rang]:.6f}, Active Pixels = {total_pixels}")
            
    def calculate_MSE_E2(self):
        """Calculate MSE for E2 strain ranges."""
        print("Calculating E2 errors...")
        
        all_E2_labelled = []
        all_E2_predicted = []
        all_masks = []

        # Collect all data
        for file_code in self.x_displacement_predicted.keys():
            all_E2_labelled.append(self.E2_label[file_code])
            all_E2_predicted.append(self.E2_predicted[file_code])
            
            # Try to get mask from the data loader
            mask = None
            for (moving, fixed), (orig_fixed, masks, _) in self.data_loader.get_generator():
                for b in range(moving.shape[0]):
                    if file_code == f"sample_{b:04d}":
                        mask = masks[b, ..., 0]
                        break
                if mask is not None:
                    break
            
            if mask is None:
                mask = np.ones_like(self.E2_label[file_code])
            
            all_masks.append(mask)

        all_E2_labelled = np.array(all_E2_labelled)
        all_E2_predicted = np.array(all_E2_predicted)
        all_masks = np.array(all_masks)
        length = all_E2_labelled.shape[0]
        
        print(f"Analyzing E2 strain in {length} samples across {len(self.strain_loss_ranges_E2)} strain ranges")

        for rang in self.strain_loss_ranges_E2:
            total_error = 0.0
            total_pixels = 0
            all_errors = []

            for i in range(length):
                label = all_E2_labelled[i]
                pred = all_E2_predicted[i]
                mask = all_masks[i]

                # Apply strain range mask
                label_mask = (label > rang - 0.025) & (label < rang + 0.025) & (mask > 0)

                # Extract the values where mask applies
                label_values = label[label_mask]
                pred_values = pred[label_mask]
                
                active_pixel_count = len(label_values)

                if active_pixel_count > 0:
                    # Calculate absolute errors
                    error = np.abs(label_values - pred_values)
                    all_errors.extend(error.tolist())
                    total_pixels += active_pixel_count

            # Compute statistics if we have data
            if all_errors:
                self.E2_loss[rang] = np.median(all_errors)
                self.E2_std[rang] = np.std(all_errors)
                self.E2_first_quartile[rang] = np.percentile(all_errors, 25)
                self.E2_second_quartile[rang] = np.percentile(all_errors, 75)
            else:
                self.E2_loss[rang] = 0.0
                self.E2_std[rang] = 0.0
                self.E2_first_quartile[rang] = 0.0
                self.E2_second_quartile[rang] = 0.0
                
            print(f"E2 Range {rang:.2f}: Median Error = {self.E2_loss[rang]:.6f}, Active Pixels = {total_pixels}")

    def plot_results(self):
        """Plot and save strain error results."""
        # Save the computed data
        output_dir = os.path.join(self.model_config[f'kernel_{self.kernel_key}_lambda_{self.lambda_val:.3f}']['folder'], 'results_simulated_test')
        os.makedirs(output_dir, exist_ok=True)
        
        np.save(os.path.join(output_dir, f"E1_loss_{self.model_name}.npy"), self.E1_loss)
        np.save(os.path.join(output_dir, f"E2_loss_{self.model_name}.npy"), self.E2_loss)
        np.save(os.path.join(output_dir, f"E1_first_quartile_{self.model_name}.npy"), self.E1_first_quartile)
        np.save(os.path.join(output_dir, f"E2_first_quartile_{self.model_name}.npy"), self.E2_first_quartile)
        np.save(os.path.join(output_dir, f"E1_third_quartile_{self.model_name}.npy"), self.E1_second_quartile)
        np.save(os.path.join(output_dir, f"E2_third_quartile_{self.model_name}.npy"), self.E2_second_quartile)
        
        print(f"Results saved to: {output_dir}")

        # Load the saved data for plotting
        E1_loss = self.E1_loss
        E2_loss = self.E2_loss
        E1_first_quartile = self.E1_first_quartile
        E2_first_quartile = self.E2_first_quartile
        E1_second_quartile = self.E1_second_quartile
        E2_second_quartile = self.E2_second_quartile
        
        # All unique sorted labels
        all_labels = sorted(list(set([float(label) for label in E1_loss.keys()] + [float(label) for label in E2_loss.keys()])))

        # Separate E1 and E2 labels
        E1_labels = sorted([label for label in all_labels if label in E1_loss])
        E2_labels = sorted([label for label in all_labels if label in E2_loss])

        # Prepare data for plotting
        E1_loss_list = [E1_loss.get(label, 0) for label in E1_labels]
        E1_q1_list = [E1_first_quartile.get(label, 0) for label in E1_labels]
        E1_q3_list = [E1_second_quartile.get(label, 0) for label in E1_labels]

        E2_loss_list = [E2_loss.get(label, 0) for label in E2_labels]
        E2_q1_list = [E2_first_quartile.get(label, 0) for label in E2_labels]
        E2_q3_list = [E2_second_quartile.get(label, 0) for label in E2_labels]

        # Plot setup
        plt.figure(figsize=(12, 6))
        bar_width = 0.05

        # Plot E2 bars
        bars_E2 = plt.bar(E2_labels, E2_loss_list, width=bar_width, color='lightcoral', edgecolor='black', label='E2')
        for i, bar in enumerate(bars_E2):
            center = bar.get_x() + bar.get_width() / 2
            plt.plot([center, center], [E2_q1_list[i], E2_q3_list[i]], color='darkred', lw=2)
            plt.scatter(center, E2_q1_list[i], color='red', zorder=3)
            plt.scatter(center, E2_q3_list[i], color='green', zorder=3)

        # Plot E1 bars
        bars_E1 = plt.bar(E1_labels, E1_loss_list, width=bar_width, color='skyblue', edgecolor='black', label='E1')
        for i, bar in enumerate(bars_E1):
            center = bar.get_x() + bar.get_width() / 2
            plt.plot([center, center], [E1_q1_list[i], E1_q3_list[i]], color='blue', lw=2)
            plt.scatter(center, E1_q1_list[i], color='red', zorder=3)
            plt.scatter(center, E1_q3_list[i], color='green', zorder=3)

        # Plot formatting
        plt.xticks(all_labels, [str(label) for label in all_labels], rotation=45)
        plt.ylabel("Absolute Error")
        model_title = f"{self.model_name} (lambda={self.lambda_val:.2f}, kernel={self.kernel_key})"
        plt.title(f"Strain Error Distribution: {model_title}")
        plt.ylim(0, 0.15)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        
        # Save the figure
        plt.savefig(os.path.join(output_dir, f"strain_error_{self.model_name}.png"))
        plt.close()
        
        print(f"Plot saved to: {os.path.join(output_dir, f'strain_error_{self.model_name}.png')}")



                    

#@title Configuration Settings { run: "auto" }
RUNNING_ON_COLAB = False #@param {type:"boolean"}
LOCAL_DATA_DIR = "./data" #@param {type:"string"}
USE_CUSTOM_VXM = True #@param {type:"boolean"}
CUSTOM_VXM_PATH = './data/voxelmorph' #@param {type:"string"}  # Full Colab path
#### Installing neccassary libraries
#### Install necessary libraries

from data.voxelmorph import voxelmorph as vxm



# Final verification
try:
    os.environ['NEURITE_BACKEND'] = 'tensorflow'  # Add this BEFORE importing neurite/voxelmorph
    import tensorflow, neurite, cv2, skimage, matplotlib, tqdm, scipy
    print("\nVerified package versions:")
    print(f"- TensorFlow {tensorflow.__version__}")
    print(f"- VoxelMorph {vxm.__version__}")
    print(f"- Neurite {neurite.__version__}")
    print(f"- OpenCV {cv2.__version__}")
    print(f"- scikit-image {skimage.__version__}")
    print(f"- Matplotlib {matplotlib.__version__}")
    print(f"- tqdm {tqdm.__version__}")
    print(f"- SciPy {scipy.__version__}")
except ImportError as e:
    print(f"\n❌ Missing package: {str(e)}")
    print("Try installing manually with: pip install", e.name)
#### Verifing Installation and GPU
import os
import sys
import neurite
import tensorflow as tf




from data.voxelmorph import voxelmorph as vxm
from data.voxelmorph.voxelmorph.tf.networks import VxmDense


# Add version verification
try:
    print(f"\nVoxelMorph version: {vxm.__version__}")
    print("Loaded from:", os.path.dirname(vxm.__file__))
except AttributeError:
    print("\n⚠️ Using custom VoxelMorph without version tag")

# Rest of your existing imports and checks...
# Verification
print("\n--- Environment Summary ---")
print(f"Running on: {'Colab' if RUNNING_ON_COLAB else 'Local'}")
print(f"Python path: {sys.path}")
print(f"\nPython: {sys.version}")

print(f"TensorFlow: {tf.__version__}")
print("GPU Devices:", tf.config.list_physical_devices('GPU'))

print(f"Neurite backend: {neurite.backend}")
print(f"VoxelMorph path: {os.path.dirname(vxm.__file__)}")
print(f"VoxelMorph version: {vxm.__version__ if hasattr(vxm, '__version__') else 'custom'}")

#### Configure data paths based on environment

BASE_DATA_PATH = LOCAL_DATA_DIR
MODELS_BASE_PATH = os.path.join(LOCAL_DATA_DIR, 'Models')

# Local-specific path structure
ACDC_BASE = ''
SUNNYBROOK_BASE = ''
train_data = os.path.join(LOCAL_DATA_DIR, 'train')
val_data = os.path.join(LOCAL_DATA_DIR, 'val')
test_data = os.path.join(LOCAL_DATA_DIR, 'test')
mask_data = os.path.join(LOCAL_DATA_DIR, 'ACDC-Masks-1')
MODEL_TESTING_PATH = os.path.join(LOCAL_DATA_DIR, 'model_testing')

train_simulated_data = os.path.join(LOCAL_DATA_DIR, 'Simulated_train')
val_simulated_data = os.path.join(LOCAL_DATA_DIR, 'Simulated_val')
test_simulated_data = os.path.join(LOCAL_DATA_DIR, 'Simulated_test')
mask_simulated_data = os.path.join(LOCAL_DATA_DIR, 'Simulated_masks')
displacement_simulated_data = os.path.join(LOCAL_DATA_DIR, 'Simulated_displacements')

# Simulated data paths (already updated in your script)
SIMULATED_DATA_PATH = test_simulated_data  # ./data/Simulated_test
SIMULATED_MASK_PATH = mask_simulated_data  # ./data/Simulated_masks
SIMULATED_DISP_PATH = displacement_simulated_data  # ./data/Simulated_displacements
def check_paths(paths):
    """Verify existence of required paths with enhanced feedback"""
    missing_paths = []
    existing_paths = []

    print("\nChecking data paths:")
    for name, path in paths.items():
        exists = os.path.exists(path)
        status = "✓" if exists else "✗"
        print(f"  {status} {name}: {path}")

        if exists:
            existing_paths.append(path)
        else:
            missing_paths.append(path)

    return existing_paths, missing_paths

# Check all critical paths
paths_to_check = {
    'Simulated Training': train_simulated_data,
    'Simulated Validation': val_simulated_data,
    'Simulated Testing': test_simulated_data,
    'Simulated Masks': mask_simulated_data,
    'Simulated Displacements': displacement_simulated_data,
    'train data': train_data,
    'val data': val_data,
    'test data': test_data,
    'mask data': mask_data, 
}

existing, missing = check_paths(paths_to_check)

if missing:
    print("\n⚠️ Missing paths detected!")
    if RUNNING_ON_COLAB:
        print("Ensure Google Drive is mounted correctly and data is in the expected locations.")
    else:
        print(f"Please ensure your local data directory ({LOCAL_DATA_DIR}) contains:")
        print("- Simulated Training/Validation/Testing folders")
        print("- Simulated Masks folder")
        print("- Simulated Displacements folder")
        print("- ACDC-Masks-1 folder")
        print("- model_testing")
        print("- train/val/test folders")
        raise FileNotFoundError("Missing required data paths")  # Uncomment to enforce strict checking
#### Configure Models paths based on environment
#### Model configuration
MODEL_CONFIG = {
    # 1. No Mask (Baseline)
    'no_mask': {
        'name': 'voxelmorph_no_mask',
        'use_mask': False,
        'use_mse_mask': False,
        'use_smoothness_mask': False
    },
    # 2. New MSE Mask only
    # 'mse_mask': {
    #     'name': 'voxelmorph_mse_mask',
    #     'use_mask': True,
    #     'use_mse_mask': True,
    #     'use_smoothness_mask': False
    # },
    # # 3. New Smoothness Mask only
    # 'smoothness_mask': {
    #     'name': 'voxelmorph_smoothness_mask',
    #     'use_mask': True,
    #     'use_mse_mask': False,
    #     'use_smoothness_mask': True
    # },
    # # 4. Both Masks
    # 'both_masks': {
    #     'name': 'voxelmorph_both_masks',
    #     'use_mask': True,
    #     'use_mse_mask': True,
    #     'use_smoothness_mask': True
    # }
}

# Lambda values to test (0.1 to 1.0)
LAMBDAS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
# LAMBDAS = [0.016, 0.033, 0.066, 0.1, 0.3, 0.5]

# kernel configrations
KERNEL_CONFIGS = {
    'default': {
        'encoder': [[3], [3], [3], [3]],
        'decoder': [[3], [3], [3], [3]],
        'final' : [3, 3, 3]
    },
    'first5': {
        'encoder': [[5], [3], [3], [3]], # first layer 5 rest is 3
        'decoder': [[3], [3], [3], [3]],
        'final' : [3, 3, 3]
    },
    'first7_second5': {
        'encoder': [[7], [5], [3], [3]],
        'decoder': [[3], [3], [3], [3]],
        'final' : [3, 3, 3]
    }
}

KERNEL_KEYS = list(KERNEL_CONFIGS.keys())  # ['default', 'first5', ...]
# Generate variable names and folder paths for all models
MODEL_VAR_MAP = {}
for model_key in MODEL_CONFIG:
    for kernel_key in KERNEL_KEYS:
        for lambda_val in LAMBDAS:
            # Unique variable name including kernel config
            # Use :.3f to match the actual directory naming (e.g., lambda_0.100)
            var_name = f"vm_model_{model_key}_kernel_{kernel_key}_lambda_{lambda_val:.3f}".replace('.', '_')
            MODEL_VAR_MAP[f"{model_key}_kernel_{kernel_key}_lambda_{lambda_val:.3f}"] = var_name

            MODEL_CONFIG[model_key][f'kernel_{kernel_key}_lambda_{lambda_val:.3f}'] = {
                'folder': os.path.join(MODELS_BASE_PATH,
                                       f"{MODEL_CONFIG[model_key]['name']}_kernel_{kernel_key}_lambda_{lambda_val:.3f}")
            }

# Create directories for each model configuration
for model_key in MODEL_CONFIG:
    for lambda_val in LAMBDAS:
        for kernel_key in KERNEL_KEYS:
            model_folder = MODEL_CONFIG[model_key][f'kernel_{kernel_key}_lambda_{lambda_val:.3f}']['folder']
            os.makedirs(os.path.join(model_folder, 'weights'), exist_ok=True)
            os.makedirs(os.path.join(model_folder, 'results'), exist_ok=True)
            os.makedirs(os.path.join(model_folder, 'logs'), exist_ok=True)
# Also assuming load_model_for_eval and SimulatedTestDataLoader are defined
def load_model_for_eval(config, kernel_key, lambda_val, load_best=True, epoch=None):
    """
    Robust model loading with architecture verification to load either the best model
    based on loss or a specific epoch's weights.

    Parameters:
    - config: Model configuration dictionary.
    - kernel_key: Kernel configuration key.
    - lambda_val: Lambda value for smoothness loss.
    - load_best: Boolean, if True loads the best model based on loss (considering all model files), if False loads the specified epoch (default=True).
    - epoch: Integer, the epoch number to load (required if load_best=False, ignored otherwise).

    Returns:
    - Loaded model or None if loading fails.
    """
    # Get model directory
    model_dir = os.path.join(
        config[f'kernel_{kernel_key}_lambda_{lambda_val:.3f}']['folder'],
        'weights'
    )

    # Check directory exists
    if not os.path.exists(model_dir):
        print(f"⚠️ Directory not found: {model_dir}")
        return None

    # Define custom objects
    custom_objects = {
        'Grad': Grad,
        'MSE': MSE,
        'Adam': tf.keras.optimizers.Adam,
        'vxm': vxm.losses  # If using original voxelmorph losses
    }

    if load_best:
        # Find all relevant model files:
        # 1. best_model_val_loss_* files (.weights.h5 or .keras)
        # 2. epoch*_loss*.weights.h5 files
        # 3. Other .keras files (e.g., final_model.keras)
        model_files = [f for f in os.listdir(model_dir) if (f.startswith('best_model_val_loss_') and (f.endswith('.weights.h5') or f.endswith('.keras'))) or
                       (f.startswith('epoch') and f.endswith('.weights.h5') and re.match(r'epoch\d+_loss\d+\.\d+\.weights\.h5', f)) or
                       (f.endswith('.keras') and not f.startswith('best_model_val_loss_'))]

        if not model_files:
            print(f"⛔ No model files found in {model_dir}")
            return None

        # Parse loss from filenames and find the best model
        best_model = None
        lowest_loss = float('inf')
        file_extension = None
        loss_type = None  # To track the type of loss (val_loss, train_loss, or unknown)

        for model_file in model_files:
            loss = float('inf')  # Default loss for files without a loss value in the name
            loss_type_candidate = 'unknown'

            if model_file.startswith('best_model_val_loss_'):
                # Extract validation loss from best_model_val_loss_* files
                match = re.search(r'best_model_val_loss_(\d+\.\d+)', model_file)
                if match:
                    loss = float(match.group(1))
                    loss_type_candidate = 'val_loss'
            elif model_file.startswith('epoch'):
                # Extract training loss from epoch*_loss*.weights.h5 files
                match = re.search(r'epoch\d+_loss(\d+\.\d+)\.weights\.h5', model_file)
                if match:
                    loss = float(match.group(1))
                    loss_type_candidate = 'train_loss'
            else:
                # For other .keras files (e.g., final_model.keras), we can't determine loss from the filename
                # Assign a high loss to deprioritize unless it's the only option
                loss_type_candidate = 'unknown'
                print(f"⚠️ No loss value found in filename {model_file}. Deprioritizing this file.")

            if loss < lowest_loss:
                lowest_loss = loss
                best_model = model_file
                loss_type = loss_type_candidate
                # Determine file extension based on the selected file
                file_extension = '.keras' if model_file.endswith('.keras') else '.weights.h5'

        if best_model is None:
            print(f"⛔ Could not determine best model in {model_dir}")
            return None

        model_path = os.path.join(model_dir, best_model)
        if loss_type == 'unknown':
            print(f"Loading best model: {best_model} (no loss value available in filename)")
        else:
            print(f"Loading best model: {best_model} with {loss_type}={lowest_loss}")

    else:
        # Load specific epoch weights (format: epoch{epoch:02d}_loss{loss:.5f}.weights.h5)
        if epoch is None:
            print("⛔ Epoch number must be specified when load_best=False")
            return None

        # Look for files matching the specified epoch
        epoch_pattern = f'epoch{epoch:02d}_loss[0-9]+\.[0-9]+\.weights\.h5'
        model_files = [f for f in os.listdir(model_dir) if re.match(epoch_pattern, f)]
        if not model_files:
            print(f"⛔ No weight files found for epoch {epoch} in {model_dir}")
            return None

        # There should be only one file matching the epoch
        if len(model_files) > 1:
            print(f"⚠️ Multiple files found for epoch {epoch}: {model_files}. Using the first one.")

        best_model = model_files[0]
        model_path = os.path.join(model_dir, best_model)
        file_extension = '.weights.h5'
        print(f"Loading epoch-specific model: {best_model}")

    try:
        # Recreate model architecture first
        model = create_voxelmorph_model(
            use_mse_mask=config['use_mse_mask'],
            use_smoothness_mask=config['use_smoothness_mask'],
            kernel_config=kernel_key,
            lambda_val=lambda_val
        )

        # Load weights into architecture
        model.load_weights(model_path)
        print(f"✅ Successfully loaded {best_model}")
        return model

    except Exception as e:
        print(f"❌ Loading weights failed: {str(e)}")
        print("Trying fallback load method...")
        try:
            # Fallback only makes sense for .keras files (full model)
            if file_extension == '.keras':
                return tf.keras.models.load_model(
                    model_path,
                    custom_objects=custom_objects
                )
            else:
                raise
        except Exception as e2:
            print(f"⛔ Critical load failure: {str(e2)}")
            return None

import os
import logging
import numpy as np
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SimulatedTestDataLoader:
    """
    A loader for simulated test data, including frames, masks, and displacements, mimicking the DataGenerator's simulated data logic.
    Prevents mixing of different CINEs by isolating pairs within the same (z, t) combination.
    Processes data in deterministic order (no shuffling) for consistent test evaluation.
    Always loads masks for metrics, but only concatenates them to fixed if use_mask=True.
    """
    def __init__(
            self,
            test_simulated_data_paths=None,
            simulated_mask_root_path=None,
            simulated_displacement_path=None,
            batch_size=8,
            use_mask=False
    ):
        """
        Initialize the SimulatedTestDataLoader.

        Args:
            test_simulated_data_paths (dict): Dictionary with 'test' key containing paths to simulated test data.
            simulated_mask_root_path (str): Path to simulated mask root directory.
            simulated_displacement_path (str): Path to simulated displacement root directory.
            batch_size (int): Number of samples per batch.
            use_mask (bool): If True, concatenate masks to fixed tensor; masks are always loaded for metrics.
        """
        self.test_simulated_data_paths = test_simulated_data_paths
        self.simulated_mask_root = simulated_mask_root_path
        self.simulated_displacement_path = simulated_displacement_path
        self.batch_size = batch_size
        self.use_mask = use_mask
        self.test_patients = []

        # Validation checks
        if self.test_simulated_data_paths is None:
            raise ValueError("Test simulated data paths must be provided.")
        if not self.simulated_mask_root:
            raise ValueError("Simulated mask root path must be provided for metric computation.")
        if not self.simulated_displacement_path:
            raise ValueError("Simulated displacement path must be provided.")
        if not os.path.exists(self.simulated_mask_root):
            raise ValueError(f"Simulated mask root path does not exist: {self.simulated_mask_root}")
        if not os.path.exists(self.simulated_displacement_path):
            raise ValueError(f"Simulated displacement path does not exist: {self.simulated_displacement_path}")

        # Organize patients
        self._organize_patients()

        # Precompute valid pairs with CINE isolation
        self.valid_pairs = {
            'test': self._precompute_pairs(self.test_patients)
        }

        # Count valid pairs
        self.test_pairs_count = sum(len(pairs) for pairs in self.valid_pairs['test'].values())
        logging.info(f"Total simulated test pairs: {self.test_pairs_count}")

    def _organize_patients(self):
        """Organize patient directories in deterministic order."""
        for path in self.test_simulated_data_paths.get('test', []):
            if os.path.isdir(path):
                patients = [
                    (os.path.join(path, f), 'simulated')
                    for f in os.listdir(path)
                    if f.startswith("patient") and os.path.isdir(os.path.join(path, f))
                ]
                # Sort by path for deterministic order
                patients.sort(key=lambda x: x[0])
                self.test_patients.extend(patients)
        logging.info(f"Organized {len(self.test_patients)} test patients")

    def _precompute_pairs(self, patient_list):
        """Precompute valid pairs for simulated test data, ensuring CINE isolation."""
        pair_dict = {}
        for patient_path, source in patient_list:
            pairs = self._extract_simulated_pairs(patient_path)
            logging.info(f"Pairs for {patient_path}: {len(pairs)}")
            if pairs:
                # Sort by (z, t, next_frame) for deterministic order
                pairs.sort(key=lambda x: (x[1], x[4], x[3]))
                pair_dict[patient_path] = pairs
        return pair_dict

    def _extract_simulated_pairs(self, patient_folder):
        """
        Extract valid pairs from simulated data files, grouping by (z, t) to prevent CINE mixing.
        """
        files = os.listdir(patient_folder)
        cine_groups = defaultdict(list)
        
        for fname in files:
            if not fname.endswith('.npy'):
                continue
            try:
                base_part, frame_part = fname.rsplit('#', 1)
                frame = frame_part.split('.')[0]
                parts = base_part.split('_')
                t_str, z_str = parts[-2].lstrip('t'), parts[-1].lstrip('z')
                t, z = int(t_str), int(z_str)
                cine_groups[(z, t)].append(frame)
            except (ValueError, IndexError) as e:
                logging.warning(f"Failed to parse {fname} in {patient_folder}: {str(e)}")
                continue
        
        valid_pairs = []
        for (z, t), frames in cine_groups.items():
            cine_pairs = self._generate_pairs_for_cine(z, t, frames)
            valid_pairs.extend(cine_pairs)
        
        return valid_pairs

    def _generate_pairs_for_cine(self, z, t, frames):
        """
        Generate pairs within a single CINE (z, t combination).
        """
        valid_pairs = []
        sorted_frames = sorted(frames)
        
        first_frame = None
        for frame in sorted_frames:
            if frame.endswith('_1'):
                first_frame = frame
                break
        
        if not first_frame:
            logging.warning(f"No base frame (ending with '_1') found for z={z}, t={t}")
            return valid_pairs

        for frame in sorted_frames:
            if frame == first_frame or frame.endswith('_1'):
                continue
            try:
                frame_num = int(frame)
                frame_diff = frame_num
                valid_pairs.append((first_frame, z, frame_diff, frame_num, t))
            except ValueError as e:
                logging.warning(f"Error computing frame_diff for {frame} (z={z}, t={t}): {str(e)}")
                continue
                
        return valid_pairs

    def _validate_pair_files(self, patient_path, pair, source):
        """
        Validate existence and shapes of files for a pair.

        Args:
            patient_path (str): Path to patient directory.
            pair (tuple): Pair information (current, z, frame_diff, next_frame, t).
            source (str): Data source ('simulated').

        Returns:
            bool: True if all files are valid, False otherwise.
        """
        try:
            patient_id = os.path.basename(patient_path)
            current, z, frame_diff, next_frame, t = pair
            z_str, t_str = f"{z:02d}", f"{t:02d}"
            base_name = patient_id.split('_z')[0] if '_z' in patient_id else patient_id

            # File paths
            file1 = os.path.join(patient_path, f"{base_name}_t{t_str}_z{z_str}#{next_frame}_1.npy")  # Moving frame
            file2 = os.path.join(patient_path, f"{base_name}_t{t_str}_z{z_str}#{next_frame}.npy")  # Fixed frame
            data_files = [file1, file2]

            mask_dir = os.path.join(self.simulated_mask_root, patient_id)
            fixed_mask_file = os.path.join(mask_dir, f"{base_name}_t{t_str}_z{z_str}#{next_frame}_1.npy")
            moving_mask_file = os.path.join(mask_dir, f"{base_name}_t{t_str}_z{z_str}#{next_frame}.npy")
            mask_files = [fixed_mask_file, moving_mask_file]

            disp_dir = os.path.join(self.simulated_displacement_path, patient_id)
            disp_x_file = os.path.join(disp_dir, f"{base_name}_t{t_str}_z{z_str}#{next_frame}_x.npy")
            disp_y_file = os.path.join(disp_dir, f"{base_name}_t{t_str}_z{z_str}#{next_frame}_y.npy")
            disp_files = [disp_x_file, disp_y_file]

            # Log file paths
            logging.debug(f"Validating pair for {patient_id} (source={source}):")
            logging.debug(f"  Moving: {file1}")
            logging.debug(f"  Fixed: {file2}")
            logging.debug(f"  Masks: {fixed_mask_file}, {moving_mask_file}")
            logging.debug(f"  Displacements: {disp_x_file}, {disp_y_file}")

            # Check file existence
            all_files = data_files + mask_files + disp_files
            missing_files = [f for f in all_files if not os.path.exists(f)]
            if missing_files:
                logging.warning(f"Missing files for pair in {patient_path}: {missing_files}")
                return False

            # Validate shapes
            for f in data_files + mask_files + disp_files:
                data = np.load(f, mmap_mode='r')
                if data.shape[:2] != (128, 128):
                    logging.warning(f"Invalid shape for {f}: expected (128, 128, ...), got {data.shape}")
                    return False

            return True

        except Exception as e:
            logging.warning(f"Error validating pair in {patient_path}: {str(e)}")
            return False

    def get_generator(self):
        """
        Generate batches of simulated test data including displacements and masks, with consistent batch sizes and deterministic order.

        Yields:
            tuple: ((moving, fixed), (fixed, mask, target_disp)) where:
                - moving: (batch_size, 128, 128, 1)
                - fixed: (batch_size, 128, 128, 3) if use_mask=True, else (batch_size, 128, 128, 1)
                - mask: (batch_size, 128, 128, 2) (fixed_mask, moving_mask)
                - target_disp: (batch_size, 128, 128, 2) (x, y displacements)
        """
        all_pairs = []
        for patient_path, source in self.test_patients:
            pairs = self.valid_pairs['test'].get(patient_path, [])
            all_pairs.extend([(patient_path, p, source) for p in pairs])

        if not all_pairs:
            logging.warning("No pairs available for test")
            return

        batch_moving, batch_fixed, batch_mask, batch_target_disp = [], [], [], []
        for pair_info in all_pairs:
            patient_path, pair, source = pair_info
            try:
                current, z, frame_diff, next_frame, t = pair
                patient_id = os.path.basename(patient_path)
                z_str, t_str = f"{z:02d}", f"{t:02d}"
                base_name = patient_id.split('_z')[0] if '_z' in patient_id else patient_id

                # File paths
                file1 = os.path.join(patient_path, f"{base_name}_t{t_str}_z{z_str}#{next_frame}_1.npy")  # Moving frame
                file2 = os.path.join(patient_path, f"{base_name}_t{t_str}_z{z_str}#{next_frame}.npy")  # Fixed frame
                if not (os.path.exists(file1) and os.path.exists(file2)):
                    logging.warning(f"Missing frame files: {file1} or {file2}")
                    continue

                # Load frames
                moving = np.load(file1, mmap_mode='r')[..., np.newaxis].astype(np.float32)
                fixed_base = np.load(file2, mmap_mode='r')[..., np.newaxis].astype(np.float32)
                if moving.shape[:2] != (128, 128) or fixed_base.shape[:2] != (128, 128):
                    logging.warning(f"Invalid frame shape: moving={moving.shape}, fixed={fixed_base.shape}")
                    continue

                # Load masks
                mask_folder = os.path.join(self.simulated_mask_root, patient_id)
                fixed_mask_file = os.path.join(mask_folder, f"{base_name}_t{t_str}_z{z_str}#{next_frame}_1.npy")
                moving_mask_file = os.path.join(mask_folder, f"{base_name}_t{t_str}_z{z_str}#{next_frame}.npy")
                if not (os.path.exists(fixed_mask_file) and os.path.exists(moving_mask_file)):
                    logging.warning(f"Missing mask files: {fixed_mask_file} or {moving_mask_file}")
                    continue
                fixed_mask = np.load(fixed_mask_file, mmap_mode='r')[..., np.newaxis].astype(np.float32)
                moving_mask = np.load(moving_mask_file, mmap_mode='r')[..., np.newaxis].astype(np.float32)

                # Prepare fixed tensor
                if self.use_mask:
                    fixed = np.concatenate([fixed_base, fixed_mask, moving_mask], axis=-1)
                else:
                    fixed = fixed_base

                # Prepare mask tensor
                mask = np.concatenate([fixed_mask, moving_mask], axis=-1)

                # Load displacement
                disp_dir = os.path.join(self.simulated_displacement_path, patient_id)
                disp_x_file = os.path.join(disp_dir, f"{base_name}_t{t_str}_z{z_str}#{next_frame}_x.npy")
                disp_y_file = os.path.join(disp_dir, f"{base_name}_t{t_str}_z{z_str}#{next_frame}_y.npy")
                if not (os.path.exists(disp_x_file) and os.path.exists(disp_y_file)):
                    logging.warning(f"Missing displacement files: {disp_x_file} or {disp_y_file}")
                    continue
                disp_x = np.load(disp_x_file, mmap_mode='r')
                disp_y = np.load(disp_y_file, mmap_mode='r')
                target_disp = np.stack([disp_x, disp_y], axis=-1)

                # Append to batches
                batch_moving.append(moving)
                batch_fixed.append(fixed)
                batch_mask.append(mask)
                batch_target_disp.append(target_disp)

                # Yield batch when full or at end
                if len(batch_moving) == self.batch_size or pair_info == all_pairs[-1]:
                    if batch_moving and batch_fixed:
                        fixed_channels = batch_fixed[0].shape[-1]
                        if not all(f.shape[-1] == fixed_channels for f in batch_fixed):
                            logging.warning(f"Inconsistent fixed channels: {[(i, f.shape) for i, f in enumerate(batch_fixed)]}")
                            batch_moving, batch_fixed, batch_mask, batch_target_disp = [], [], [], []
                            continue
                        mask_channels = batch_mask[0].shape[-1]
                        if not all(m.shape[-1] == mask_channels for m in batch_mask):
                            logging.warning(f"Inconsistent mask channels: {[(i, m.shape) for i, m in enumerate(batch_mask)]}")
                            batch_moving, batch_fixed, batch_mask, batch_target_disp = [], [], [], []
                            continue
                        if not (len(batch_moving) == len(batch_fixed) == len(batch_mask) == len(batch_target_disp)):
                            logging.warning(f"Batch misalignment: moving={len(batch_moving)}, fixed={len(batch_fixed)}, mask={len(batch_mask)}, disp={len(batch_target_disp)}")
                            batch_moving, batch_fixed, batch_mask, batch_target_disp = [], [], [], []
                            continue
                        yield (np.stack(batch_moving), np.stack(batch_fixed)), (np.stack(batch_fixed), np.stack(batch_mask), np.stack(batch_target_disp))
                    batch_moving, batch_fixed, batch_mask, batch_target_disp = [], [], [], []

            except Exception as e:
                logging.warning(f"Error processing pair {pair} in {patient_path}: {str(e)}")

        # Handle final batch
        if len(batch_moving) > 0:
            if not (len(batch_moving) == len(batch_fixed) == len(batch_mask) == len(batch_target_disp)):
                logging.warning(f"Final batch misalignment: moving={len(batch_moving)}, fixed={len(batch_fixed)}, mask={len(batch_mask)}, disp={len(batch_target_disp)}")
            else:
                original_batch_size = len(batch_moving)
                if original_batch_size < self.batch_size:
                    last_moving = batch_moving[-1].copy()
                    last_fixed = batch_fixed[-1].copy()
                    last_mask = batch_mask[-1].copy()
                    last_target_disp = batch_target_disp[-1].copy()
                    while len(batch_moving) < self.batch_size:
                        batch_moving.append(last_moving)
                        batch_fixed.append(last_fixed)
                        batch_mask.append(last_mask)
                        batch_target_disp.append(last_target_disp)
                    logging.info(f"Duplicated last pair to fill batch: {original_batch_size} to {self.batch_size}")
                fixed_channels = batch_fixed[0].shape[-1]
                if not all(f.shape[-1] == fixed_channels for f in batch_fixed):
                    logging.warning(f"Inconsistent final fixed channels: {[(i, f.shape) for i, f in enumerate(batch_fixed)]}")
                    return
                mask_channels = batch_mask[0].shape[-1]
                if not all(m.shape[-1] == mask_channels for m in batch_mask):
                    logging.warning(f"Inconsistent final mask channels: {[(i, m.shape) for i, m in enumerate(batch_mask)]}")
                    return
                yield (np.stack(batch_moving), np.stack(batch_fixed)), (np.stack(batch_fixed), np.stack(batch_mask), np.stack(batch_target_disp))

    def get_data_by_patient_and_skip(self, patient_numbers):
        """
        Retrieve data for specific patients, organized by CINE (z,t) and frame skip.
        Each (z,t) combination represents a unique CINE sequence.
        """
        if not patient_numbers:
            raise ValueError("Patient numbers list cannot be empty.")
        
        patient_numbers = [str(num).zfill(3) for num in patient_numbers]
        requested_patients = []
        for patient_path, source in self.test_patients:
            patient_id = os.path.basename(patient_path)
            patient_num = patient_id.replace("patient", "")
            if patient_num in patient_numbers:
                requested_patients.append((patient_path, source))

        if not requested_patients:
            logging.warning(f"No patients found matching numbers: {patient_numbers}")
            return {}

        patient_data = {}
        for patient_path, source in requested_patients:
            patient_id = os.path.basename(patient_path)
            pairs = self.valid_pairs['test'].get(patient_path, [])
            if not pairs:
                logging.warning(f"No valid pairs for {patient_id}")
                continue

            # Group pairs by CINE (z,t) - each CINE is independent
            cine_groups = defaultdict(list)
            for pair in pairs:
                current, z, frame_diff, next_frame, t = pair
                cine_key = (z, t)  # Unique CINE identifier
                cine_groups[cine_key].append(pair)

            skip_data = {}
            for cine_key, cine_pairs in cine_groups.items():
                z, t = cine_key
                # Sort pairs within this CINE by frame number
                cine_pairs.sort(key=lambda x: x[3])  # Sort by next_frame
                
                logging.info(f"Processing CINE z={z}, t={t} with {len(cine_pairs)} frames for {patient_id}")
                
                for pair in cine_pairs:
                    try:
                        current, z, frame_diff, next_frame, t = pair
                        z_str, t_str = f"{z:02d}", f"{t:02d}"
                        base_name = patient_id.split('_z')[0] if '_z' in patient_id else patient_id

                        # Load frame data
                        file1 = os.path.join(patient_path, f"{base_name}_t{t_str}_z{z_str}#{next_frame}_1.npy")
                        file2 = os.path.join(patient_path, f"{base_name}_t{t_str}_z{z_str}#{next_frame}.npy")
                        if not (os.path.exists(file1) and os.path.exists(file2)):
                            logging.warning(f"Missing frame files for CINE z={z}, t={t}: {file1} or {file2}")
                            continue

                        moving = np.load(file1, mmap_mode='r')[..., np.newaxis].astype(np.float32)
                        fixed_base = np.load(file2, mmap_mode='r')[..., np.newaxis].astype(np.float32)

                        # Load masks
                        mask_folder = os.path.join(self.simulated_mask_root, patient_id)
                        fixed_mask_file = os.path.join(mask_folder, f"{base_name}_t{t_str}_z{z_str}#{next_frame}_1.npy")
                        moving_mask_file = os.path.join(mask_folder, f"{base_name}_t{t_str}_z{z_str}#{next_frame}.npy")
                        if not (os.path.exists(fixed_mask_file) and os.path.exists(moving_mask_file)):
                            logging.warning(f"Missing mask files for CINE z={z}, t={t}")
                            continue
                        
                        fixed_mask = np.load(fixed_mask_file, mmap_mode='r')[..., np.newaxis].astype(np.float32)
                        moving_mask = np.load(moving_mask_file, mmap_mode='r')[..., np.newaxis].astype(np.float32)

                        # Prepare fixed tensor
                        if self.use_mask:
                            fixed = np.concatenate([fixed_base, fixed_mask, moving_mask], axis=-1)
                        else:
                            fixed = fixed_base

                        mask = np.concatenate([fixed_mask, moving_mask], axis=-1)

                        # Load displacement
                        disp_dir = os.path.join(self.simulated_displacement_path, patient_id)
                        disp_x_file = os.path.join(disp_dir, f"{base_name}_t{t_str}_z{z_str}#{next_frame}_x.npy")
                        disp_y_file = os.path.join(disp_dir, f"{base_name}_t{t_str}_z{z_str}#{next_frame}_y.npy")
                        if not (os.path.exists(disp_x_file) and os.path.exists(disp_y_file)):
                            logging.warning(f"Missing displacement files for CINE z={z}, t={t}")
                            continue
                        
                        disp_x = np.load(disp_x_file, mmap_mode='r')
                        disp_y = np.load(disp_y_file, mmap_mode='r')
                        target_disp = np.stack([disp_x, disp_y], axis=-1)

                        # Add batch dimension
                        moving = np.expand_dims(moving, axis=0)
                        fixed = np.expand_dims(fixed, axis=0)
                        mask = np.expand_dims(mask, axis=0)
                        target_disp = np.expand_dims(target_disp, axis=0)

                        # Create unique key that includes CINE information
                        unique_key = f"{frame_diff}_z{z_str}_t{t_str}_frame{next_frame}"
                        skip_data[unique_key] = ((moving, fixed), (fixed, mask, target_disp), next_frame)

                    except Exception as e:
                        logging.warning(f"Error processing CINE z={z}, t={t}, frame {next_frame} for {patient_id}: {str(e)}")
                        continue

            if skip_data:
                patient_data[patient_id] = skip_data
                logging.info(f"Loaded {len(skip_data)} frames across multiple CINEs for {patient_id}")
            else:
                logging.warning(f"No valid data extracted for {patient_id}")

        return patient_data

#### Model Creation
##### MSE Loss
class MSE:
    """
    Sigma-weighted mean squared error for image reconstruction.
    """

    def __init__(self, image_sigma=1.0):
        self.image_sigma = image_sigma

    def mse(self, y_true, y_pred):
        return K.square(y_true - y_pred)

    def loss(self, y_true, y_pred, reduce='mean'):
        # compute mse
        mse = self.mse(y_true, y_pred)

        mask = y_true[..., 1]  # Second channel for fixed mask
        # apply mask
        mse = mse * tf.expand_dims(mask, axis=-1)

        # reduce
        if reduce == 'mean':
            mse = K.mean(mse)
        elif reduce == 'max':
            mse = K.max(mse)
        elif reduce is not None:
            raise ValueError(f'Unknown MSE reduction type: {reduce}')
        # loss
        return 1.0 / (self.image_sigma ** 2) * mse

##### Smootheness Loss
class Grad:
    """
    N-D gradient loss.
    loss_mult can be used to scale the loss value - this is recommended if
    the gradient is computed on a downsampled vector field (where loss_mult
    is equal to the downsample factor).
    """

    def __init__(self, penalty='l1', loss_mult=None, vox_weight=None):
        self.penalty = penalty
        self.loss_mult = loss_mult
        self.vox_weight = vox_weight

    def _diffs(self, y):
        vol_shape = y.get_shape().as_list()[1:-1]
        ndims = len(vol_shape)

        df = [None] * ndims
        for i in range(ndims):
            d = i + 1
            # permute dimensions to put the ith dimension first
            r = [d, *range(d), *range(d + 1, ndims + 2)]
            yp = K.permute_dimensions(y, r)
            dfi = yp[1:, ...] - yp[:-1, ...]

            if self.vox_weight is not None:
                w = K.permute_dimensions(self.vox_weight, r)
                # TODO: Need to add square root, since for non-0/1 weights this is bad.
                dfi = w[1:, ...] * dfi

            # permute back
            # note: this might not be necessary for this loss specifically,
            # since the results are just summed over anyway.
            r = [*range(1, d + 1), 0, *range(d + 1, ndims + 2)]
            df[i] = K.permute_dimensions(dfi, r)

        return df

    def loss(self, y_true, y_pred):
        """
        returns Tensor of size [bs]
        """
        mask = y_true[..., 1]  # [batch, H, W]
        mask = tf.expand_dims(mask, -1)  # [batch, H, W, 1]

        # Resize the mask to match the spatial dimensions of y_pred
        target_size = tf.shape(y_pred)[1:3]  # assuming y_pred shape: [batch, new_H, new_W, channels]
        # Use bilinear interpolation for continuous values
        mask = tf.image.resize(mask, size=target_size, method="bilinear")

        self.vox_weight = mask

        # Reset y_true[..., 1] to zero to restore it as zero_phi
        x_channel = y_true[..., 0:1]  # [batch, H, W, 1]
        zero_channel = tf.zeros_like(x_channel)  # [batch, H, W, 1]
        y_true = tf.concat([x_channel, zero_channel], axis=-1)  # [batch, H, W, 2]

        if self.penalty == 'l1':
            dif = [tf.abs(f) for f in self._diffs(y_pred)]
        else:
            assert self.penalty == 'l2', 'penalty can only be l1 or l2. Got: %s' % self.penalty
            dif = [f * f for f in self._diffs(y_pred)]

        df = [tf.reduce_mean(K.batch_flatten(f), axis=-1) for f in dif]
        grad = tf.add_n(df) / len(df)

        if self.loss_mult is not None:
            grad *= self.loss_mult

        return grad

    def mean_loss(self, y_true, y_pred):
        """
        returns Tensor of size ()
        """

        return K.mean(self.loss(y_true, y_pred))
##### Model params
def create_voxelmorph_model(use_mse_mask=False, use_smoothness_mask=False, kernel_config='default', lambda_val=0.1):
    input_shape = (128, 128)
    src_feats = 1  # Moving image has 1 channel
    trg_feats = 3 if (use_mse_mask or use_smoothness_mask) else 1  # Fixed image + mask channels

    # Input layers
    source_input = tf.keras.Input(shape=(*input_shape, src_feats), name='source_input')
    target_input = tf.keras.Input(shape=(*input_shape, trg_feats), name='target_input')

    # Build VxmDense model
    nb_features = [
        [16, 32, 32, 32],  # encoder
        [32, 32, 32, 32, 32, 16, 16]  # decoder
    ]

    # Get kernel configuration
    kernels = KERNEL_CONFIGS[kernel_config] if USE_CUSTOM_VXM else None

    # Create base VxmDense model
    vm_model = vxm.networks.VxmDense(
        inshape=input_shape,
        nb_unet_features=nb_features,
        unet_kernel_sizes=kernels,
        src_feats=src_feats,
        trg_feats=trg_feats,
        input_model=tf.keras.Model(inputs=[source_input, target_input], outputs=[source_input, target_input]),
        int_steps=5,
        reg_field = 'warp'
    )

    # Configure losses
    losses = []
    loss_weights = []

    # Loss functions
    losses = []
    loss_weights = []

    # 1. MSE Loss (with optional mask)
    if use_mse_mask:
        # Custom MSE loss with BG-to-myocardium ratio mask
        losses.append(MSE().loss)
    else:
        losses.append(vxm.losses.MSE().loss)

    loss_weights.append(1)  # Weight for similarity loss

    # 2. Smoothness Loss (with optional mask)
    if use_smoothness_mask:
        # Custom smoothness loss
        losses.append(Grad('l2').loss)
    else:
        losses.append(vxm.losses.Grad('l2').loss)

    loss_weights.append(lambda_val)  # Weight for smoothness loss

    # Compile model
    vm_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=losses,
        loss_weights=loss_weights
    )
    return vm_model



def evaluate_simulated_models_improved(models_config, lambdas, kernel_keys, 
                                     test_simulated_data, mask_simulated_data, 
                                     displacement_simulated_data):
    """
    Improved evaluation pipeline with better resource management and error handling.
    """
    print("\nStarting Improved Simulated Test Evaluations")
    print("=" * 50)
    import time
    
    # Initialize evaluation statistics
    eval_stats = {
        'start_time': time.time(),
        'total_models': len(models_config) * len(kernel_keys) * len(lambdas),
        'successful_evaluations': 0,
        'failed_evaluations': 0
    }
    
    # Iterate through all model configurations
    for model_key in models_config:
        config = models_config[model_key]
        
        for kernel_key in kernel_keys:
                # Call test_model with the correct parameters
                try:
                    model_name = f"{config['name']}_kernel_{kernel_key}_lambda_{lambda_val:.3f}"
                    test_model(
                        model_config=config,
                        kernel_key=kernel_key,
                        lambda_val=lambda_val,
                        model_name=model_name
                    )
                    eval_stats['successful_evaluations'] += 1
                except Exception as e:
                    print(f"Error evaluating {model_name}: {str(e)}")
                    eval_stats['failed_evaluations'] += 1
                
                # Call test_model with the correct parameters
                test_model(
                    model_config=config,
                    kernel_key=kernel_key,
                    lambda_val=lambda_val,
                    model_name=model_name
                )
    
    # Final summary
    elapsed_time = time.time() - eval_stats['start_time']
    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total models: {eval_stats['total_models']}")
    print(f"Successful: {eval_stats['successful_evaluations']}")
    print(f"Failed: {eval_stats['failed_evaluations']}")
    print(f"Success rate: {eval_stats['successful_evaluations']/eval_stats['total_models']*100:.1f}%")
    print(f"Total time: {elapsed_time/60:.1f} minutes")
    print(f"Average time per model: {elapsed_time/eval_stats['total_models']:.1f} seconds")


if __name__ == "__main__":
    # Test all models using the class
    evaluate_simulated_models_improved(
        models_config=MODEL_CONFIG,
        lambdas=LAMBDAS,
        kernel_keys=KERNEL_KEYS,
        test_simulated_data={'test': [SIMULATED_DATA_PATH]},
        mask_simulated_data=SIMULATED_MASK_PATH,
        displacement_simulated_data=SIMULATED_DISP_PATH
    )