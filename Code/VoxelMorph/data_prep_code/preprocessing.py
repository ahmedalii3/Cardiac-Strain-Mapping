import os
import sys
import re
import gc
import glob
import time
import math
import random
import logging
import warnings
import traceback
import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Generator

import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import Loss
from tensorflow.keras.layers import Input
from tensorflow.keras.callbacks import (
  ModelCheckpoint,
  EarlyStopping,
  CSVLogger,
  ReduceLROnPlateau,
  TensorBoard,
)
import tensorflow.keras.backend as K

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import cv2
import h5py
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from scipy.ndimage import distance_transform_edt, gaussian_filter

import neurite as ne
from docx import Document
from docx.oxml.ns import qn

LOCAL_DATA_DIR = "./data"

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
    print(f"Please ensure your local data directory ({LOCAL_DATA_DIR}) contains:")
    print("- Simulated Training/Validation/Testing folders")
    print("- Simulated Masks folder")
    print("- Simulated Displacements folder")
    print("- ACDC-Masks-1 folder")
    print("- model_testing")
    print("- train/val/test folders")
    raise FileNotFoundError("Missing required data paths")  # Uncomment to enforce strict checking
###### New BG to myocarduim mask weighting
def create_weighted_mask_bg(mask, dilation_extent=5, sigma=2, myocardium_weight_factor=1):
    """
    Generate a weighted mask with:
    - Myocardium (label 1) = weight 2.0
    - Smoothly decaying weights outward from the myocardium (controlled by `dilation_extent`).
    - Multiplied by the ratio of background pixels to myocardium pixels for increased myocardium influence.

    Args:
        mask (np.ndarray): Input mask with labels {0, 1, 2}.
        dilation_extent (int): Number of dilation iterations (higher = wider decay).
        sigma (float): Smoothness of the decay (Gaussian blur).
        myocardium_weight_factor (float): Additional factor to control the myocardium weight (default = 1).

    Returns:
        weighted_mask (np.ndarray): Weighted mask with values ≥1.0.
    """
    # Ensure mask is 2D by squeezing singleton dimensions
    mask = mask.squeeze()

    # Extract myocardium (label 1) and background (label 0)
    myocardium = (mask == 1).astype(np.float32)
    background = (mask == 0).astype(np.float32)

    # Compute the ratio of background pixels to myocardium pixels
    num_background_pixels = np.sum(background)
    num_myocardium_pixels = np.sum(myocardium)
    ratio = num_background_pixels / (num_myocardium_pixels + 1e-6)  # Add epsilon to avoid division by zero

    # Initialize dilated mask and process mask
    dilated_mask = myocardium.copy()
    process_mask = myocardium.copy()

    # Kernel for dilation (7x7 ellipse)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    # Gradually reduce the added weight per iteration
    initial_value = 0.9
    step_size = initial_value / dilation_extent  # Controls decay per iteration

    for i in range(dilation_extent):
        old_process_mask = process_mask.copy()
        process_mask = cv2.dilate(process_mask, kernel)

        # Identify newly added pixels (boundary of the dilated region)
        added_region = (process_mask - old_process_mask).astype(np.float32)

        # Ensure added_region has the same number of dimensions as dilated_mask
        added_region = added_region[..., np.newaxis] if added_region.ndim < dilated_mask.ndim else added_region

        # Compute weight for this iteration (decays linearly with iterations)
        current_weight = initial_value - i * step_size

        # Update dilated mask with decaying weights
        dilated_mask += added_region * current_weight

    # Smooth the dilation
    dilated_mask[myocardium.astype(bool)] = 1.0
    smoothed_mask = gaussian_filter(dilated_mask, sigma=sigma)
    # smoothed_mask = (1 - np.exp(1.2 * dilated_mask)) / (1 - np.exp(1.2))

    # Multiply by the ratio of background to myocardium pixels
    smoothed_mask *= ratio

    # Apply the myocardium weight factor
    smoothed_mask *= myocardium_weight_factor

    # Add 1 to the mask
    smoothed_mask += 1.0

    mask_sum = tf.reduce_sum(smoothed_mask)  # Scalar

    # Compute the number of pixels in the mask
    num_pixels = tf.reduce_sum(tf.ones_like(smoothed_mask))  # Scalar

    # Compute the normalization factor
    normalization_factor = num_pixels / (mask_sum + 1e-6)

    smoothed_mask = smoothed_mask * normalization_factor

    return smoothed_mask[..., np.newaxis]
###### New Inverted mask for smoothing
def create_weighted_mask_inverted(mask, dilation_extent=5, sigma=2):
    """
    Generate a weighted mask with INVERTED weights (background prioritized):
    - Background (label 0) = weight 2.0
    - Smoothly decaying weights inward from the background (controlled by `dilation_extent`).
    - Same calculations as original, but with (1 - mask) applied before +1.0.

    Args:
        mask (np.ndarray): Input mask with labels {0, 1, 2}.
        dilation_extent (int): Number of dilation iterations (higher = wider decay).
        sigma (float): Smoothness of the decay (Gaussian blur).

    Returns:
        weighted_mask (np.ndarray): Weighted mask with values ≥1.0.
    """
    # Ensure mask is 2D by squeezing singleton dimensions
    mask = mask.squeeze()

    # Extract myocardium (label 1) - we'll still dilate from myocardium as in original
    myocardium = (mask == 1).astype(np.float32)

    # Initialize dilated mask and process mask (same as original)
    dilated_mask = myocardium.copy()
    process_mask = myocardium.copy()

    # Kernel for dilation (7x7 ellipse)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    # Gradually reduce the added weight per iteration (same as original)
    initial_value = 0.9
    step_size = initial_value / dilation_extent

    for i in range(dilation_extent):
        old_process_mask = process_mask.copy()
        process_mask = cv2.dilate(process_mask, kernel)

        # Identify newly added pixels (boundary of the dilated region)
        added_region = (process_mask - old_process_mask).astype(np.float32)
        added_region = added_region[..., np.newaxis] if added_region.ndim < dilated_mask.ndim else added_region

        # Compute weight for this iteration (decays linearly with iterations)
        current_weight = initial_value - i * step_size

        # Update dilated mask with decaying weights
        dilated_mask += added_region * current_weight

    # Smooth the dilation
    dilated_mask[myocardium.astype(bool)] = 1.0
    smoothed_mask = gaussian_filter(dilated_mask, sigma=sigma)
    # smoothed_mask = (1 - np.exp(1.2 * dilated_mask)) / (1 - np.exp(1.2))

    ##### KEY MODIFICATION: Invert weights here (before +1.0) #####
    smoothed_mask = 1.0 - smoothed_mask

    # Add 1 to the mask (now background will be ~2.0, myocardium ~1.0)
    smoothed_mask += 1.0

    # Normalization (same as original, but using smoothed_mask)
    mask_sum = tf.reduce_sum(smoothed_mask)  # Scalar
    num_pixels = tf.reduce_sum(tf.ones_like(smoothed_mask))  # Scalar
    normalization_factor = num_pixels / (mask_sum + 1e-6)

    return smoothed_mask[..., np.newaxis] * normalization_factor
# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@dataclass
class DataSample:
    """Simple container for a data sample"""
    moving: np.ndarray
    fixed: np.ndarray
    masks: Optional[Tuple[np.ndarray, np.ndarray]] = None
    patient_id: str = ""
    source: str = "real"

class DataPreprocessor:
    """
    Data preprocessor that collects and saves training data to HDF5 files.
    Replicates the logic from SimpleHybridDataLoader.
    """
    
    def __init__(
        self,
        real_data_paths: Dict[str, List[str]],
        simulated_data_paths: Optional[Dict[str, List[str]]] = None,
        mask_root_path: Optional[str] = None,
        simulated_mask_root_path: Optional[str] = None,
        output_dir: str = "Training_data",
        max_frame_skip: int = 7,
        min_frame_skip: int = 3,
        shuffle: bool = True,
        seed: int = 42,
        use_mask: bool = True,
        chunk_size: int = 1000  # Process data in chunks to manage memory
    ):
        self.real_data_paths = real_data_paths
        self.simulated_data_paths = simulated_data_paths or {'train': [], 'val': [], 'test': []}
        self.mask_root = mask_root_path
        self.simulated_mask_root = simulated_mask_root_path
        self.output_dir = output_dir
        self.max_frame_skip = max_frame_skip
        self.min_frame_skip = min_frame_skip
        self.shuffle = shuffle
        self.use_mask = use_mask
        self.chunk_size = chunk_size
        
        # Set random seed for reproducibility
        np.random.seed(seed)
        random.seed(seed)
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        if self.use_mask and (not self.mask_root or not self.simulated_mask_root):
            raise ValueError("Mask paths required when use_mask=True")
        
        logging.info(f"DataPreprocessor initialized. Output directory: {self.output_dir}")
        logging.info(f"Using masks: {self.use_mask}")

    def process_and_save_all_splits(self):
        """Process and save all data splits"""
        splits = ['train', 'val', 'test']
        
        for split in splits:
            logging.info(f"Processing {split} split...")
            try:
                self.process_and_save_split(split)
                logging.info(f"Successfully processed and saved {split} split")
            except Exception as e:
                logging.error(f"Failed to process {split} split: {e}")
                raise

    def process_and_save_split(self, split: str):
        """Process and save a single data split"""
        logging.info(f"Collecting samples for {split} split...")
        
        # Collect all samples first
        raw_samples = []
        
        # Load real data
        for i, data_path in enumerate(tqdm(self.real_data_paths.get(split, []), desc=f"Collecting real samples for {split}")):
            self._collect_real_samples(data_path, raw_samples)
        
        logging.info(f"Collected {len(raw_samples)} real raw samples for {split} split")
        
        # Load simulated data
        for i, data_path in enumerate(tqdm(self.simulated_data_paths.get(split, []), desc=f"Collecting simulated samples for {split}")):
            self._collect_simulated_samples(data_path, raw_samples)
        
        logging.info(f"Collected {len(raw_samples)} total raw samples for {split} split")

        if not raw_samples:
            logging.warning(f"No samples found for {split} split")
            # Create empty arrays
            self._save_empty_split(split)
            return
        
        if self.shuffle:
            random.shuffle(raw_samples)
        
        # Process and save in chunks to manage memory
        self._process_and_save_in_chunks(raw_samples, split)

    def _process_and_save_in_chunks(self, raw_samples: List[DataSample], split: str):
        """Process samples in chunks and save to HDF5"""
        total_samples = len(raw_samples)
        valid_samples_count = 0
        
        # Initialize lists to collect all processed data
        all_moving = []
        all_fixed = []
        all_zero_phi = []
        
        logging.info(f"Processing {total_samples} samples for {split} split in chunks of {self.chunk_size}")
        
        # Process in chunks
        for chunk_start in range(0, total_samples, self.chunk_size):
            chunk_end = min(chunk_start + self.chunk_size, total_samples)
            chunk_samples = raw_samples[chunk_start:chunk_end]
            
            logging.info(f"Processing chunk {chunk_start//self.chunk_size + 1}/{(total_samples-1)//self.chunk_size + 1} "
                        f"(samples {chunk_start+1}-{chunk_end})")
            
            # Process chunk
            chunk_moving = []
            chunk_fixed = []
            chunk_zero_phi = []
            
            for i, sample in enumerate(tqdm(chunk_samples, desc=f"Processing {split} chunk")):
                try:
                    processed_moving, processed_fixed, processed_zero_phi = self._preprocess_sample(sample)
                    
                    if processed_moving is not None:
                        chunk_moving.append(processed_moving)
                        chunk_fixed.append(processed_fixed)
                        chunk_zero_phi.append(processed_zero_phi)
                        valid_samples_count += 1
                
                except Exception as e:
                    logging.warning(f"Failed to preprocess sample {sample.patient_id}: {e}")
                    continue
            
            # Add chunk data to overall lists
            if chunk_moving:
                all_moving.extend(chunk_moving)
                all_fixed.extend(chunk_fixed)
                all_zero_phi.extend(chunk_zero_phi)
            
            # Clear chunk data to free memory
            del chunk_moving, chunk_fixed, chunk_zero_phi
            
            logging.info(f"Chunk processed. Valid samples so far: {valid_samples_count}")
        
        if not all_moving:
            logging.error(f"No valid samples after preprocessing for {split} split")
            self._save_empty_split(split)
            return
        
        # Convert to numpy arrays
        logging.info(f"Converting {len(all_moving)} samples to numpy arrays...")
        final_moving = np.stack(all_moving, axis=0)
        final_fixed = np.stack(all_fixed, axis=0)
        final_zero_phi = np.stack(all_zero_phi, axis=0)
        
        # Clear intermediate lists to free memory
        del all_moving, all_fixed, all_zero_phi
        
        # Save to HDF5
        self._save_to_hdf5(final_moving, final_fixed, final_zero_phi, split)
        
        logging.info(f"Successfully processed and saved {valid_samples_count} valid samples for {split}")
        logging.info(f"{split} shapes - Moving: {final_moving.shape}, "
                    f"Fixed: {final_fixed.shape}, "
                    f"Zero_phi: {final_zero_phi.shape}")

    def _save_to_hdf5(self, moving: np.ndarray, fixed: np.ndarray, zero_phi: np.ndarray, split: str):
        """Save processed data to HDF5 file"""
        output_path = os.path.join(self.output_dir, f"{split}_data.h5")
        
        logging.info(f"Saving {split} data to {output_path}")
        
        with h5py.File(output_path, 'w') as f:
            # Save arrays with compression
            f.create_dataset('moving', data=moving, compression='gzip', compression_opts=9)
            f.create_dataset('fixed', data=fixed, compression='gzip', compression_opts=9)
            f.create_dataset('zero_phi', data=zero_phi, compression='gzip', compression_opts=9)
            
            # Save metadata
            f.attrs['num_samples'] = moving.shape[0]
            f.attrs['use_mask'] = self.use_mask
            f.attrs['moving_shape'] = str(moving.shape)
            f.attrs['fixed_shape'] = str(fixed.shape)
            f.attrs['zero_phi_shape'] = str(zero_phi.shape)
        
        logging.info(f"Successfully saved {split} data to {output_path}")

    def _save_empty_split(self, split: str):
        """Save empty arrays for splits with no data"""
        output_path = os.path.join(self.output_dir, f"{split}_data.h5")
        
        fixed_channels = 3 if self.use_mask else 1
        
        empty_moving = np.empty((0, 128, 128, 1), dtype=np.float32)
        empty_fixed = np.empty((0, 128, 128, fixed_channels), dtype=np.float32)
        empty_zero_phi = np.empty((0, 128, 128, 2), dtype=np.float32)
        
        with h5py.File(output_path, 'w') as f:
            f.create_dataset('moving', data=empty_moving)
            f.create_dataset('fixed', data=empty_fixed)
            f.create_dataset('zero_phi', data=empty_zero_phi)
            
            f.attrs['num_samples'] = 0
            f.attrs['use_mask'] = self.use_mask
            f.attrs['moving_shape'] = str(empty_moving.shape)
            f.attrs['fixed_shape'] = str(empty_fixed.shape)
            f.attrs['zero_phi_shape'] = str(empty_zero_phi.shape)
        
        logging.warning(f"Saved empty {split} data to {output_path}")

    def _preprocess_sample(self, sample: DataSample) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """Preprocess a single sample into final format"""
        try:
            # Validate sample first
            if not self._validate_sample_shapes(sample) or not self._validate_sample_data(sample):
                return None, None, None
            
            # Process moving image (always same)
            processed_moving = sample.moving.copy()  # (128, 128, 1)
            
            # Process fixed image (with or without mask)
            if self.use_mask and sample.masks:
                mask_moving, mask_fixed = sample.masks
                mask_fixed_weighted = create_weighted_mask_bg(mask_fixed)
                mask_moving_weighted = create_weighted_mask_inverted(mask_moving)
                
                processed_fixed = np.concatenate([
                    sample.fixed,           # (128, 128, 1)
                    mask_fixed_weighted,    # (128, 128, 1)
                    mask_moving_weighted    # (128, 128, 1)
                ], axis=-1)  # (128, 128, 3)
            else:
                processed_fixed = sample.fixed.copy()  # (128, 128, 1)
            
            # Create zero_phi
            processed_zero_phi = np.zeros((128, 128, 2), dtype=np.float32)
            
            # Set second channel of zero_phi if using masks
            if self.use_mask and sample.masks:
                processed_zero_phi[..., 1] = processed_fixed[..., 2]
                # First channel remains zero
            
            return processed_moving, processed_fixed, processed_zero_phi
            
        except Exception as e:
            logging.warning(f"Error preprocessing sample {sample.patient_id}: {e}")
            return None, None, None

    def _collect_real_samples(self, data_path: str, sample_list: List[DataSample]):
        """Collect real data samples"""
        if not os.path.isdir(data_path):
            logging.warning(f"Directory not found: {data_path}")
            return
        
        patient_folders = [f for f in os.listdir(data_path) 
                          if f.startswith("patient") and os.path.isdir(os.path.join(data_path, f))]
        
        logging.info(f"Collecting real samples from {len(patient_folders)} patient folders in {data_path}")
        
        for patient_folder in patient_folders:
            patient_path = os.path.join(data_path, patient_folder)
            pairs = self._extract_real_pairs(patient_path)
            
            for pair in pairs:
                sample = self._load_real_sample(patient_path, pair, patient_folder)
                if sample:
                    sample_list.append(sample)

    def _collect_simulated_samples(self, data_path: str, sample_list: List[DataSample]):
        """Collect simulated data samples"""
        if not os.path.isdir(data_path):
            logging.warning(f"Directory not found: {data_path}")
            return
        
        patient_folders = [f for f in os.listdir(data_path) 
                          if f.startswith("patient") and os.path.isdir(os.path.join(data_path, f))]
        
        logging.info(f"Collecting simulated samples from {len(patient_folders)} patient folders in {data_path}")
        
        for patient_folder in patient_folders:
            patient_path = os.path.join(data_path, patient_folder)
            pairs = self._extract_simulated_pairs(patient_path)
            
            for pair in pairs:
                sample = self._load_simulated_sample(patient_path, pair, patient_folder)
                if sample:
                    sample_list.append(sample)

    def _extract_real_pairs(self, patient_folder: str) -> List[Tuple]:
        """Extract valid frame pairs from real data"""
        slice_times = {}
        
        for fname in os.listdir(patient_folder):
            if not fname.endswith('.npy'):
                continue
            try:
                parts = fname.split('_')
                t_part = parts[1].lstrip('t')
                z_part = parts[2].split('.')[0].lstrip('z')
                t, z = int(t_part), int(z_part)
                if z <= 1 or z >= 6:
                    continue
                slice_times.setdefault(z, []).append(t)
            except (ValueError, IndexError):
                continue
        
        valid_pairs = []
        for z, times in slice_times.items():
            sorted_times = sorted(times)
            for i in range(len(sorted_times)):
                current_t = sorted_times[i]
                for j in range(i + 1, len(sorted_times)):
                    next_t = sorted_times[j]
                    frame_diff = next_t - current_t
                    if self.min_frame_skip <= frame_diff <= self.max_frame_skip:
                        valid_pairs.append((current_t, z, frame_diff, next_t, current_t))
        
        return valid_pairs

    def _extract_simulated_pairs(self, patient_folder: str) -> List[Tuple]:
        """Extract valid frame pairs from simulated data"""
        files = os.listdir(patient_folder)
        slice_times = {}
        actual_frames = set()
        
        for fname in files:
            if not fname.endswith('.npy'):
                continue
            try:
                base_part, frame_part = fname.rsplit('#', 1)
                frame = frame_part.split('.')[0]
                actual_frames.add(frame)
                parts = base_part.split('_')
                t_str, z_str = parts[-2].lstrip('t'), parts[-1].lstrip('z')
                t, z = int(t_str), int(z_str)
                slice_times.setdefault((z, t), []).append(frame)
            except (ValueError, IndexError):
                continue
        
        valid_pairs = []
        for (z, t), frames in slice_times.items():
            sorted_frames = sorted(frames, key=lambda x: int(x))
            first_frame = next((frame for frame in sorted_frames if frame[-2:] == '_1'), None)
            
            for frame in sorted_frames:
                if frame[-2:] != '_1' and first_frame in actual_frames and frame in actual_frames:
                    frame_diff = int(frame) - 1
                    valid_pairs.append((first_frame, z, frame_diff, frame, t))
        
        return valid_pairs

    def _load_real_sample(self, patient_path: str, pair: Tuple, patient_id: str) -> Optional[DataSample]:
        """Load a real data sample, returning None if invalid"""
        current, z, frame_diff, next_frame, t = pair
        
        file1 = os.path.join(patient_path, f"{patient_id}_t{current:02d}_z{z:02d}.npy")
        file2 = os.path.join(patient_path, f"{patient_id}_t{next_frame:02d}_z{z:02d}.npy")
        
        if not (os.path.exists(file1) and os.path.exists(file2)):
            return None
        
        try:
            moving = np.load(file1).astype(np.float32)
            fixed = np.load(file2).astype(np.float32)
            
            if len(moving.shape) == 2:
                moving = moving[..., np.newaxis]
            if len(fixed.shape) == 2:
                fixed = fixed[..., np.newaxis]
            
            if moving.shape != (128, 128, 1) or fixed.shape != (128, 128, 1):
                return None
            
            masks = None
            if self.use_mask:
                mask_folder = os.path.join(self.mask_root, patient_id)
                mask_file1 = f"{patient_id}_t{current:02d}_z{z:02d}_mask.npy"
                mask_file2 = f"{patient_id}_t{next_frame:02d}_z{z:02d}_mask.npy"
                
                mask_path1 = os.path.join(mask_folder, mask_file1)
                mask_path2 = os.path.join(mask_folder, mask_file2)
                
                if not (os.path.exists(mask_path1) and os.path.exists(mask_path2)):
                    return None
                
                mask_moving = np.load(mask_path1).astype(np.float32)
                mask_fixed = np.load(mask_path2).astype(np.float32)
                
                if len(mask_moving.shape) == 2:
                    mask_moving = mask_moving[..., np.newaxis]
                if len(mask_fixed.shape) == 2:
                    mask_fixed = mask_fixed[..., np.newaxis]
                
                if mask_moving.shape != (128, 128, 1) or mask_fixed.shape != (128, 128, 1):
                    return None
                
                masks = (mask_moving, mask_fixed)
            
            sample = DataSample(
                moving=moving,
                fixed=fixed,
                masks=masks,
                patient_id=patient_id,
                source="real"
            )
            
            if not self._validate_sample_shapes(sample) or not self._validate_sample_data(sample):
                return None
            
            return sample
            
        except Exception as e:
            logging.warning(f"Error loading real sample {file1}: {e}")
            return None

    def _load_simulated_sample(self, patient_path: str, pair: Tuple, patient_id: str) -> Optional[DataSample]:
        """Load a simulated data sample, returning None if invalid"""
        current, z, frame_diff, next_frame, t = pair
        
        z_str, t_str = f"{z:02d}", f"{t:02d}"
        base_name = patient_id.split('_z')[0] if '_z' in patient_id else patient_id
        
        file1 = os.path.join(patient_path, f"{base_name}_t{t_str}_z{z_str}#{next_frame}_1.npy")
        file2 = os.path.join(patient_path, f"{base_name}_t{t_str}_z{z_str}#{next_frame}.npy")
        
        if not (os.path.exists(file1) and os.path.exists(file2)):
            return None
        
        try:
            moving = np.load(file1).astype(np.float32)
            fixed = np.load(file2).astype(np.float32)
            
            if len(moving.shape) == 2:
                moving = moving[..., np.newaxis]
            if len(fixed.shape) == 2:
                fixed = fixed[..., np.newaxis]
            
            if moving.shape != (128, 128, 1) or fixed.shape != (128, 128, 1):
                return None
            
            masks = None
            if self.use_mask:
                mask_folder = os.path.join(self.simulated_mask_root, patient_id)
                mask_file1 = f"{base_name}_t{t_str}_z{z_str}#{next_frame}_1.npy"
                mask_file2 = f"{base_name}_t{t_str}_z{z_str}#{next_frame}.npy"
                
                mask_path1 = os.path.join(mask_folder, mask_file1)
                mask_path2 = os.path.join(mask_folder, mask_file2)
                
                if not (os.path.exists(mask_path1) and os.path.exists(mask_path2)):
                    return None
                
                mask_moving = np.load(mask_path1).astype(np.float32)
                mask_fixed = np.load(mask_path2).astype(np.float32)
                
                if len(mask_moving.shape) == 2:
                    mask_moving = mask_moving[..., np.newaxis]
                if len(mask_fixed.shape) == 2:
                    mask_fixed = mask_fixed[..., np.newaxis]
                
                if mask_moving.shape != (128, 128, 1) or mask_fixed.shape != (128, 128, 1):
                    return None
                
                masks = (mask_moving, mask_fixed)
            
            sample = DataSample(
                moving=moving,
                fixed=fixed,
                masks=masks,
                patient_id=patient_id,
                source="simulated"
            )
            
            if not self._validate_sample_shapes(sample) or not self._validate_sample_data(sample):
                return None
            
            return sample
            
        except Exception as e:
            logging.warning(f"Error loading simulated sample {file1}: {e}")
            return None

    def _validate_sample_shapes(self, sample: DataSample) -> bool:
        """Validate that a sample has correct shapes"""
        if sample.moving.shape != (128, 128, 1):
            return False
        
        if sample.fixed.shape != (128, 128, 1):
            return False
        
        if self.use_mask and sample.masks:
            mask_moving, mask_fixed = sample.masks
            if mask_moving.shape != (128, 128, 1) or mask_fixed.shape != (128, 128, 1):
                return False
        
        return True

    def _validate_sample_data(self, sample: DataSample) -> bool:
        """Validate sample data quality"""
        # Check for NaN or infinite values
        if np.isnan(sample.moving).any() or np.isinf(sample.moving).any():
            return False
        
        if np.isnan(sample.fixed).any() or np.isinf(sample.fixed).any():
            return False
        
        # Check data range (assuming normalized 0-1 or similar)
        if sample.moving.min() < -10 or sample.moving.max() > 10:
            return False
        
        if sample.fixed.min() < -10 or sample.fixed.max() > 10:
            return False
        
        return True


def load_preprocessed_data(output_dir: str = "Training_data", split: str = "train") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load preprocessed data from HDF5 files
    
    Args:
        output_dir: Directory containing the HDF5 files
        split: Data split to load ('train', 'val', 'test')
    
    Returns:
        Tuple of (moving, fixed, zero_phi) arrays
    """
    file_path = os.path.join(output_dir, f"{split}_data.h5")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Preprocessed data file not found: {file_path}")
    
    with h5py.File(file_path, 'r') as f:
        moving = f['moving'][:]
        fixed = f['fixed'][:]
        zero_phi = f['zero_phi'][:]
        
        logging.info(f"Loaded {split} data:")
        logging.info(f"  Moving: {moving.shape}")
        logging.info(f"  Fixed: {fixed.shape}")
        logging.info(f"  Zero_phi: {zero_phi.shape}")
        logging.info(f"  Use mask: {f.attrs.get('use_mask', 'unknown')}")
    
    return moving, fixed, zero_phi

def main():
    """Main function to run data preprocessing"""
    parser = argparse.ArgumentParser(description='Preprocess and save training data')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--output_dir', type=str, default='Training_data', help='Output directory for processed data')
    parser.add_argument('--chunk_size', type=int, default=1000, help='Chunk size for processing')
    parser.add_argument('--use_mask', action='store_true', help='Use masks in preprocessing')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    # Example configuration - replace with your actual paths
    real_data_paths={
            'train': [train_data],
            'val': [val_data],
            'test': [test_data]
        }

    simulated_data_paths={
            'train': [train_simulated_data],
            'val': [val_simulated_data],
            'test': [test_simulated_data]
        }

    mask_root_path=mask_data
    simulated_mask_root_path=mask_simulated_data

    # Create preprocessor
    preprocessor = DataPreprocessor(
        real_data_paths=real_data_paths,
        simulated_data_paths=simulated_data_paths,
        mask_root_path=mask_root_path,
        simulated_mask_root_path=simulated_mask_root_path,
        output_dir=args.output_dir,
        use_mask=True,
        chunk_size=args.chunk_size,
        seed=args.seed
    )

    # Process and save all splits
    preprocessor.process_and_save_all_splits()

    logging.info("Data preprocessing completed successfully!")

    # Example of loading the data
    logging.info("Testing data loading...")
    try:
        train_moving, train_fixed, train_zero_phi = load_preprocessed_data(args.output_dir, 'train')
        logging.info("Data loading test successful!")
    except Exception as e:
        logging.error(f"Data loading test failed: {e}")

if __name__ == "__main__":
    main()
