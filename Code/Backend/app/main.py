import io
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'  # Must be set BEFORE torch import
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Disables GPU detection
from fastapi.params import Body
import torch
torch.set_default_device('cpu')
os.environ["nnUNet_use_MPS"] = "false"
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend for matplotlib
import sys
import subprocess
import importlib.util
from pathlib import Path
import numpy as np
import nibabel as nib
import pydicom
from fastapi import FastAPI, Form, UploadFile, File, HTTPException
from typing import List
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from io import BytesIO
import re
import tensorflow as tf
from fastapi.responses import StreamingResponse
import matplotlib.pyplot as plt
from pydicom.uid import ExplicitVRLittleEndian
from PIL import Image
import json
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from PIL import Image
from .processing import get_metadata, process_uploaded_files, cleanup_temp_files, get_center_of_mass, create_localized_series, crop_to_center_of_mass, transform_coordinates
from .bullseye import generate_bullseye_plots
from .strain import compute_strains, create_strain_dicom_grayscale, create_localized_dicom, dicom_to_base64
from .supervised import MaskLoss, MAELoss, Unet, Conv_block, UpConv_block, Max_pool
from .segmentation_class import segment

app = FastAPI()

# CORS Setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:5173", "http://localhost:5174"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "temp/uploaded_files"
PROCESSED_NP_DIR = "temp/processed_numpy_files"
LOCALIZED_DIR = "temp/localized_files"
STRAIN_DIR = "temp/strain_temp"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PROCESSED_NP_DIR, exist_ok=True)
os.makedirs(LOCALIZED_DIR, exist_ok=True)
os.makedirs(STRAIN_DIR, exist_ok=True)

# Load U-Net Model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "Unet.keras")
try:
    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={'MaskLoss': MaskLoss, 'MAELoss': MAELoss, 'Unet': Unet,
                       'Conv_block': Conv_block, 'UpConv_block': UpConv_block, 'Max_pool': Max_pool}
    )
except Exception as e:
    raise Exception(f"Failed to load U-Net model: {str(e)}")

# Load Segment Model
segmentor = segment()
segmentor.set_global_variables()
segmentor.install_nnUnet()


@app.post("/metadata")
async def get_series_metadata(files: List[UploadFile] = File(...)):
    """Get metadata from DICOM series."""
    try:
        metadata_list = []
        for file in files:
            content = await file.read()
            dicom_data = pydicom.dcmread(BytesIO(content))
            metadata = get_metadata(dicom_data)
            
            # For localized series, add additional metadata
            if getattr(dicom_data, 'SeriesDescription', '').startswith('Localized'):
                metadata.update({
                    'IsLocalized': True,
                    'OriginalSeriesUID': getattr(dicom_data, 'OriginalSeriesUID', ''),
                    'LocalizationTimestamp': getattr(dicom_data, 'ContentDate', '') + getattr(dicom_data, 'ContentTime', '')
                })
            
            metadata_list.append(metadata)
        
        return {"metadata": metadata_list}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing metadata: {str(e)}")

@app.post("/localize")
async def localize_series(files: List[UploadFile] = File(...)):
    """
    Process DICOM series to localize and crop around the heart region.
    """
    processed_files = []
    new_series_uid = pydicom.uid.generate_uid()
    series_dir = os.path.join(LOCALIZED_DIR, str(new_series_uid))
    localized_files = []

    try:
        # Step 1: Process and save uploaded files
        original_shapes, processed_files = await process_uploaded_files(files)
        
        # Step 2: Get center of mass using the localization model
        preprocessed_center = get_center_of_mass(processed_files)
        
        # Step 3: Create output directory
        os.makedirs(series_dir, exist_ok=True)
        
        # Step 4: Process each file to create localized versions
        localized_files = create_localized_series(
            processed_files,
            original_shapes,
            preprocessed_center,
            new_series_uid,
            series_dir
        )
        
        return {
            "status": "success",
            "series_uid": new_series_uid,
            "series_description": "Localized Series",
            "files": localized_files
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error during localization: {str(e)}")
    finally:
        # Clean up temporary files
        cleanup_temp_files(processed_files)

@app.post("/strain/dicom")
async def compute_strain_dicom(files: List[UploadFile] = File(...)):
    try:
        print(f"Received {len(files)} files: {[f.filename for f in files]}")
        if len(files) < 2:
            raise HTTPException(status_code=400, detail="At least 2 DICOM frames required for strain analysis")

        # Step 1: Localization preprocessing
        print("Starting localization preprocessing")
        new_series_uid = pydicom.uid.generate_uid()
        series_dir = os.path.join(STRAIN_DIR, str(new_series_uid))
        os.makedirs(series_dir, exist_ok=True)
        processed_files = []

        # Step 1.1: Process uploaded files for localization
        print("Step 1.1: Processing uploaded files for localization")
        original_shapes, processed_files = await process_uploaded_files(files)
        print(f"Processed {len(processed_files)} files, original shapes: {original_shapes}")
        
        # Step 1.2: Get center of mass
        print("Step 1.2: Computing center of mass")
        preprocessed_center = get_center_of_mass(processed_files)
        print(f"Center of mass: {preprocessed_center}")
        
        # Step 1.3: Create localized NumPy arrays
        print("Step 1.3: Creating localized NumPy arrays")
        series_arrays = []
        dicom_data_list = []
        for idx, (file_info, original_shape) in enumerate(zip(processed_files, original_shapes)):
            # image = sitk.ReadImage(file_info['np_path'])
            print(f"Reading image from {file_info['np_path']}")
            image = np.load(file_info['np_path'])
            # array = sitk.GetArrayFromImage(image)
            cropped = crop_to_center_of_mass(image, preprocessed_center)
            series_arrays.append(cropped[0])  # Shape: (128, 128)
            dicom_data_list.append(file_info['dicom_data'])
            print(f"Processed frame {idx+1}, shape: {cropped[0].shape}")

        # Step 2: Sort by InstanceNumber
        print("Step 2: Sorting by InstanceNumber")
        sorted_indices = sorted(
            range(len(dicom_data_list)),
            key=lambda i: int(dicom_data_list[i].InstanceNumber)
        )
        series_arrays = [series_arrays[i] for i in sorted_indices]
        dicom_data_list = [dicom_data_list[i] for i in sorted_indices]
        print(f"Sorted {len(series_arrays)} arrays, first array shape: {series_arrays[0].shape}")

        # Step 3: Compute displacement fields
        print("Step 3: Computing displacement fields")
        frame1 = series_arrays[0] 
        frame1_norm = (frame1 - np.min(frame1)) / (np.max(frame1) - np.min(frame1) + 1e-8)
        frame1_scaled = (frame1_norm * 255).astype(np.uint8)
        displacement_fields = []
        for i, frame_n in enumerate(series_arrays[1:], 1):
            print(f"Computing displacement for frame {i+1}")
            frame_n_norm = (frame_n - np.min(frame_n)) / (np.max(frame_n) - np.min(frame_n) + 1e-8)
            frame_n_scaled = (frame_n_norm * 255).astype(np.uint8)
            print(f"Frame1 scaled range: min={np.min(frame1_scaled)}, max={np.max(frame1_scaled)}")
            print(f"Frame {i+1} scaled range: min={np.min(frame_n_scaled)}, max={np.max(frame_n_scaled)}")
            frame1_input = tf.expand_dims(frame1_scaled, axis=0)
            frame_n_input = tf.expand_dims(frame_n_scaled, axis=0)
            pred = model.predict([frame1_input, frame_n_input], verbose=0)
            displacement_fields.append(pred[0])
            print(f"Displacement field {i} shape: {pred[0].shape}")

        # Step 4: Compute strain maps
        print("Step 4: Computing strain maps")
        FrameDisplX = np.stack([df[:, :, 0] for df in displacement_fields], axis=-1)
        FrameDisplY = np.stack([df[:, :, 1] for df in displacement_fields], axis=-1)
        print(f"FrameDisplX shape: {FrameDisplX.shape}, FrameDisplY shape: {FrameDisplY.shape}")
        deltaX = 1.0
        deltaY = 1.0
        print(f"Pixel spacing: deltaX={deltaX}, deltaY={deltaY}")
        Ep1All, Ep2All, Ep3All = compute_strains(FrameDisplX, FrameDisplY, deltaX, deltaY)
        print(f"Strain maps shapes: Ep1All={Ep1All.shape}, Ep2All={Ep2All.shape}, Ep3All={Ep3All.shape}")
        print(f"E1 strain value range: min={np.min(Ep1All):.4f}, max={np.max(Ep1All):.4f}")
        print(f"E2 strain value range: min={np.min(Ep2All):.4f}, max={np.max(Ep2All):.4f}")
        print(f"E3 strain value range: min={np.min(Ep3All):.4f}, max={np.max(Ep3All):.4f}")

        # Step 5: Create DICOM files for the localized original series
        print("Step 5: Creating DICOM files for localized original series")
        localized_dicoms = []
        for i, (array, original_dicom) in enumerate(zip(series_arrays, dicom_data_list), 1):
            new_dicom = create_localized_dicom(
                original_dicom=original_dicom,
                image_array=array,
                series_description="Localized Original Series",
                series_uid=new_series_uid,
                instance_number=i
            )
            content = dicom_to_base64(new_dicom)
            localized_dicoms.append({"filename": f"localized_{i}.dcm", "content": content})
            print(f"Prepared localized series DICOM: localized_{i}.dcm")

        # Step 6: Create strain DICOMs for E1, E2, E3 in grayscale
        print("Step 6: Creating strain DICOMs in grayscale")
        strain1_series_uid = pydicom.uid.generate_uid()
        strain2_series_uid = pydicom.uid.generate_uid()
        strain3_series_uid = pydicom.uid.generate_uid()
        strain1_dicoms = []
        strain2_dicoms = []
        strain3_dicoms = []
        for i in range(Ep1All.shape[-1]):
            # E1
            new_dicom, strain_min, strain_max, hist_bins, hist_counts, percentile_5, percentile_95 = create_strain_dicom_grayscale(
                original_dicom=dicom_data_list[i + 1],
                strain_array=Ep1All[:, :, i],
                series_description="Strain-E1-Grayscale",
                series_uid=strain1_series_uid,
                instance_number=i + 1
            )
            content = dicom_to_base64(new_dicom)
            strain1_dicoms.append({
                "filename": f"strain1_{i+1}.dcm",
                "content": content,
                "strain_min": float(strain_min),
                "strain_max": float(strain_max),
                "histogram_bins": hist_bins,
                "histogram_counts": hist_counts,
                "percentile_5": float(percentile_5),
                "percentile_95": float(percentile_95)
            })
            print(f"Created strain1 DICOM: strain1_{i+1}.dcm")
            
            # E2
            new_dicom, strain_min, strain_max, hist_bins, hist_counts, percentile_5, percentile_95 = create_strain_dicom_grayscale(
                original_dicom=dicom_data_list[i + 1],
                strain_array=Ep2All[:, :, i],
                series_description="Strain-E2-Grayscale",
                series_uid=strain2_series_uid,
                instance_number=i + 1
            )
            content = dicom_to_base64(new_dicom)
            strain2_dicoms.append({
                "filename": f"strain2_{i+1}.dcm",
                "content": content,
                "strain_min": float(strain_min),
                "strain_max": float(strain_max),
                "histogram_bins": hist_bins,
                "histogram_counts": hist_counts,
                "percentile_5": float(percentile_5),
                "percentile_95": float(percentile_95)
            })
            print(f"Created strain2 DICOM: strain2_{i+1}.dcm")
            
            # E3
            new_dicom, strain_min, strain_max, hist_bins, hist_counts, percentile_5, percentile_95 = create_strain_dicom_grayscale(
                original_dicom=dicom_data_list[i + 1],
                strain_array=Ep3All[:, :, i],
                series_description="Strain-E3-Grayscale",
                series_uid=strain3_series_uid,
                instance_number=i + 1
            )
            content = dicom_to_base64(new_dicom)
            strain3_dicoms.append({
                "filename": f"strain3_{i+1}.dcm",
                "content": content,
                "strain_min": float(strain_min),
                "strain_max": float(strain_max),
                "histogram_bins": hist_bins,
                "histogram_counts": hist_counts,
                "percentile_5": float(percentile_5),
                "percentile_95": float(percentile_95)
            })
            print(f"Created strain3 DICOM: strain3_{i+1}.dcm")

        # Step 7: Segment localized series with temporary directory for validation
        print("Step 7: Segmenting localized series with debugging")
        segmentation_masks = []

        # Save localized arrays for debugging
        for i, array in enumerate(series_arrays):
            frame_idx = i + 1
            print(f"Processing frame {frame_idx} for segmentation")
            
            affine = np.eye(4)
            #print min and max of the array
            print(f"Frame {frame_idx} array min: {np.min(array)}, max: {np.max(array)}")
            nifti_image = nib.Nifti1Image(array, affine)  
            
            # Save to both nnUNet directory and debug directory
            nifti_path = os.path.join("/Users/muhannad159/Documents/GP-DICOM-VIEWER/backend_/app/nnUNet/nnUNet_raw/Dataset007_ShortAX/imagesTs", f"case_{frame_idx}_0000.nii.gz")
            
            
            nib.save(nifti_image, nifti_path)
           
            
            # Log NIfTI info
            nifti_info = {
                "frame": frame_idx,
                "shape": array.shape,
                "affine": affine.tolist(),
                "pixel_spacing": [deltaX, deltaY],
                "original_data_range": [float(np.min(array)), float(np.max(array))]
            }
          

        # Run segmentation
       
        segmentor.predict_masks()

        # Step 8: Remap and load segmented masks with debugging
        print("Step 8: Remapping segmented masks with debugging")
        mask_summary = []
        
        for i in range(1, len(series_arrays) + 1):
            mask_path = os.path.join("/Users/muhannad159/Documents/GP-DICOM-VIEWER/backend_/app/nnUNet/nnUNet_raw/Dataset007_ShortAX/pred_nnUnet", f"case_{i}.nii.gz")
            
            if os.path.exists(mask_path):
                # Load mask
                nifti_image = nib.load(mask_path)
                mask = nifti_image.get_fdata().astype(np.uint8)
                
                # Create visualization
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                
                # Original image
                axes[0].imshow(series_arrays[i-1], cmap='gray')
                axes[0].set_title(f'Original Frame {i}')
                axes[0].axis('off')
                
                # Mask
                axes[1].imshow(mask, cmap='jet')
                axes[1].set_title(f'Segmentation Mask {i}')
                axes[1].axis('off')
                
                # Overlay
                axes[2].imshow(series_arrays[i-1], cmap='gray')
                masked = np.ma.masked_where(mask == 0, mask)
                axes[2].imshow(masked, cmap='jet', alpha=0.5)
                axes[2].set_title(f'Overlay Frame {i}')
                axes[2].axis('off')
                
                plt.tight_layout()
                # plt.savefig(os.path.join(debug_dir, "mask_visualizations", f"visualization_{i}.png"))
                plt.close()
                
                # Log mask info
                unique_values, counts = np.unique(mask, return_counts=True)
                mask_info = {
                    "frame": i,
                    "shape": mask.shape,
                    "unique_values": unique_values.tolist(),
                    "value_counts": counts.tolist(),
                    "num_segmented_pixels": int(np.sum(mask > 0)),
                    "segmentation_percentage": float(np.sum(mask > 0) / mask.size * 100)
                }
                mask_summary.append(mask_info)
                
                print(f"Loaded mask for frame {i}, shape: {mask.shape}, unique values: {np.unique(mask)}")
                print(f"Segmentation percentage: {mask_info['segmentation_percentage']:.2f}%")
                
                # Process mask for response
                mask_binary = (mask == 1).astype(np.uint8)  # MYO mask
                mask_list = mask_binary.tolist()
                segmentation_masks.append({
                    "filename": f"mask_{i}",
                    "values": mask_list
                })
            else:
                print(f"Mask not found for frame {i}, using zero mask")
                zero_mask = np.zeros((128, 128), dtype=np.uint8)
                segmentation_masks.append({
                    "filename": f"mask_{i}",
                    "values": zero_mask.tolist()
                })
                
                mask_info = {
                    "frame": i,
                    "error": "Mask file not found",
                    "shape": [128, 128],
                    "num_segmented_pixels": 0,
                    "segmentation_percentage": 0.0
                }
                mask_summary.append(mask_info)
        
        # Save mask summary
        
        # Create overall summary plot
        if mask_summary:
            frames = [m['frame'] for m in mask_summary if 'segmentation_percentage' in m]
            percentages = [m['segmentation_percentage'] for m in mask_summary if 'segmentation_percentage' in m]
            
            plt.figure(figsize=(10, 6))
            plt.plot(frames, percentages, 'b-o')
            plt.xlabel('Frame Number')
            plt.ylabel('Segmentation Percentage (%)')
            plt.title('Segmentation Coverage Across Frames')
            plt.grid(True)
            # plt.savefig(os.path.join(debug_dir, "segmentation_summary.png"))
            plt.close()
        
        # print(f"Debug files saved to: {debug_dir}")
        # Step 9: Generate bullseye plots
        print("Step 9: Generating bullseye plots")
        strain_arrays = {
            "Ep1All": Ep1All,
            "Ep2All": Ep2All,
            "Ep3All": Ep3All
        }
        num_strain_frames = len(series_arrays) - 1
        bullseye_plots = generate_bullseye_plots(
            series_arrays=series_arrays,
            strain_arrays=strain_arrays,
            masks=segmentation_masks,
            num_strain_frames=num_strain_frames,
            ring=True
        )
        # Extract segment means for frontend tooltips
        segment_means = {
            "bullseye1": [],
            "bullseye2": [],
            "bullseye3": []
        }
        for strain_idx in range(num_strain_frames):
            frame_idx = strain_idx + 1  # Frames 2–7
            mask = np.array(segmentation_masks[frame_idx]["values"], dtype=np.uint8)
            cx, cy = 64, 64
            for strain_key, bullseye_key in [("Ep1All", "bullseye1"), ("Ep2All", "bullseye2"), ("Ep3All", "bullseye3")]:
                strain_map = strain_arrays[strain_key][:, :, strain_idx] * 100
                segment_values = np.zeros(6)
                segment_counts = np.zeros(6)
                for row in range(128):
                    for col in range(128):
                        if mask[row, col]:
                            x, y = col - cx, row - cy
                            angle = (np.arctan2(y, x) * 180 / np.pi + 360) % 360
                            segmen = int(angle // 60)
                            segment_values[segmen] += strain_map[row, col]
                            segment_counts[segmen] += 1
                segment_values = np.divide(
                    segment_values,
                    segment_counts,
                    out=np.zeros_like(segment_values),
                    where=segment_counts != 0
                ).tolist()
                segment_means[bullseye_key].append({
                    "frame": frame_idx + 1,  
                    "segment_means": segment_values
                })

        # Step 10: Clean up (keep debug files)
        print("Step 10: Cleaning up temporary files (keeping debug files)")
        cleanup_temp_files(processed_files)
        print(f"Cleaned up {len(processed_files)} processed files")

        # Step 11: Prepare response
        print("Step 11: Preparing response")
        response = {
            "localized": localized_dicoms,
            "strain1": strain1_dicoms,
            "strain2": strain2_dicoms,
            "strain3": strain3_dicoms,
            "masks": segmentation_masks,
            "bullseye1": bullseye_plots['bullseye1'],
            "bullseye2": bullseye_plots['bullseye2'],
            "bullseye3": bullseye_plots['bullseye3'],
            "bullseye1_ring": bullseye_plots['bullseye1_ring'],
            "bullseye2_ring": bullseye_plots['bullseye2_ring'],
            "bullseye3_ring": bullseye_plots['bullseye3_ring'],
            "segment_means": segment_means
        }
        print("Response prepared, returning")
        return response

        # Error handling
    except Exception as e:
        print(f"Error in strain DICOM computation: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error computing strain DICOMs: {str(e)}")

@app.post("/report")
async def generate_report(
    bullseye1_png: UploadFile = File(...),
    bullseye2_png: UploadFile = File(...),
    bullseye3_png: UploadFile = File(...),
    bullseye1_ring_png: UploadFile = File(...),
    bullseye2_ring_png: UploadFile = File(...),
    bullseye3_ring_png: UploadFile = File(...),
    segment_means: str = Form(...)
):
    """
    Generate a PDF report from strain analysis results with dynamic frame handling.
    Accepts segment means in the format:
    [
        E1_frame1_segments, E1_frame2_segments, ..., E1_frameN_segments,
        E2_frame1_segments, E2_frame2_segments, ..., E2_frameN_segments,
        E3_frame1_segments, E3_frame2_segments, ..., E3_frameN_segments
    ]
    Where each segment array contains 6 values.
    """
    
    # Parse segment_means from JSON string
    try:
        segment_means_data = json.loads(segment_means)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON format for segment_means")
    
    # Validate we have data for all three views (E1, E2, E3)
    if len(segment_means_data) % 3 != 0:
        raise HTTPException(status_code=400, detail="Segment means must contain data for all three views (E1, E2, E3)")
    
    # Calculate number of frames per view
    num_frames = len(segment_means_data) // 3
    
    # Validate all segments have exactly 6 values
    for segment in segment_means_data:
        if len(segment) != 6:
            raise HTTPException(status_code=400, detail="Each segment array must contain exactly 6 values")
    
    # Extract current frame number from filenames
    def get_frame_number(filename: str) -> int:
        match = re.search(r"frame(\d+)", filename)
        if not match:
            raise HTTPException(status_code=400, detail=f"Invalid filename format: {filename}")
        return int(match.group(1))

    # Get frame number from files
    files = [bullseye1_png, bullseye2_png, bullseye3_png, 
             bullseye1_ring_png, bullseye2_ring_png, bullseye3_ring_png]
    current_frame_numbers = [get_frame_number(file.filename) for file in files]
    
    # Ensure all frame numbers match for the uploaded images
    if len(set(current_frame_numbers)) != 1:
        raise HTTPException(status_code=400, detail="All image files must correspond to the same frame")
    current_frame_number = current_frame_numbers[0]

    # Validate current frame is within our data range
    if current_frame_number < 1 or current_frame_number > num_frames:
        raise HTTPException(status_code=400, detail=f"Frame number {current_frame_number} is out of range (1-{num_frames})")

    # Validate and process PNG files
    try:
        images = []
        file_labels = [
            (bullseye1_png, "E1 Bullseye"),
            (bullseye2_png, "E2 Bullseye"),
            (bullseye3_png, "E3 Bullseye"),
            (bullseye1_ring_png, "E1 Ring Bullseye"),
            (bullseye2_ring_png, "E2 Ring Bullseye"),
            (bullseye3_ring_png, "E3 Ring Bullseye")
        ]
        
        for file, label in file_labels:
            if file.content_type != "image/png":
                raise HTTPException(status_code=400, detail=f"File {file.filename} must be a PNG")
            content = await file.read()
            img = Image.open(io.BytesIO(content))
            if img.format != "PNG":
                raise HTTPException(status_code=400, detail=f"File {file.filename} is not a valid PNG")
            images.append((ImageReader(io.BytesIO(content)), label))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing images: {str(e)}")

    # Create PDF
    pdf_buffer = io.BytesIO()
    c = canvas.Canvas(pdf_buffer, pagesize=A4)
    width, height = A4  # 595 x 842 points

    # Title
    c.setFont("Helvetica-Bold", 16)
    c.drawCentredString(width / 2, height - 50, 
                       f"Strain Analysis Report - Frame {current_frame_number} (of {num_frames} frames)")

    # Draw Bullseye plots (2x3 grid) preserving aspect ratio
    img_width, img_height = 180, 150
    aspect_ratio = img_width / img_height
    x_positions = [100, 250, 400]
    y_positions = [600, 400]
    for idx, (img, label) in enumerate(images):
        row = idx // 3
        col = idx % 3
        x = x_positions[col]
        y = y_positions[row]
        # Preserve aspect ratio by adjusting height based on width
        c.drawImage(img, x, y, img_width, img_height)
        c.setFont("Helvetica", 10)
        c.drawCentredString(x + img_width / 2, y - 20, label)

    # Generate segment means line plot (dynamic frame handling)
    try:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
        frame_numbers = list(range(1, num_frames + 1))  # 1-based frame numbers
        colors = ['b', 'g', 'r', 'c', 'm', 'y']
        labels = [f"Segment {i+1}" for i in range(6)]
        
        # Calculate current frame index (0-based)
        current_frame_idx = current_frame_number - 1
        
        # Adjust subplot spacing to make room for legend
        plt.subplots_adjust(right=0.85)
        
        # E1: first num_frames segments
        for i in range(6):  # For each of the 6 segments
            # Get values for this segment across all E1 frames
            segment_values = [segment_means_data[frame][i] for frame in range(num_frames)]
            line, = ax1.plot(frame_numbers, segment_values, color=colors[i], label=labels[i])
            # Highlight current frame
            ax1.plot(frame_numbers[current_frame_idx], segment_values[current_frame_idx], 
                    'o', color=line.get_color(), markersize=8)
        ax1.set_title("E1 Segment Means")
        ax1.set_ylabel("Strain Value")
        ax1.grid(True)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        # Increase y-axis range from 0-40 to 0-50
        ax1.set_ylim(-20, 60)
        
        # Set x-axis ticks with thinning for large frame numbers
        if num_frames > 10:
            step = max(2, num_frames // 10)  # Adjust step based on number of frames
            ax1.set_xticks(frame_numbers[::step])
        else:
            ax1.set_xticks(frame_numbers)
        ax1.set_xlim(frame_numbers[0] - 0.5, frame_numbers[-1] + 0.5)

        # E2: middle num_frames segments
        for i in range(6):
            segment_values = [segment_means_data[num_frames + frame][i] for frame in range(num_frames)]
            line, = ax2.plot(frame_numbers, segment_values, color=colors[i], label=labels[i])
            ax2.plot(frame_numbers[current_frame_idx], segment_values[current_frame_idx], 
                    'o', color=line.get_color(), markersize=8)
        ax2.set_title("E2 Segment Means")
        ax2.set_ylabel("Strain Value")
        ax2.grid(True)
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        # Increase y-axis range from -20-0 to -25-0
        ax2.set_ylim(-40, 10)
        if num_frames > 10:
            ax2.set_xticks(frame_numbers[::step])
        else:
            ax2.set_xticks(frame_numbers)
        ax2.set_xlim(frame_numbers[0] - 0.5, frame_numbers[-1] + 0.5)

        # E3: last num_frames segments
        for i in range(6):
            segment_values = [segment_means_data[2*num_frames + frame][i] for frame in range(num_frames)]
            line, = ax3.plot(frame_numbers, segment_values, color=colors[i], label=labels[i])
            ax3.plot(frame_numbers[current_frame_idx], segment_values[current_frame_idx], 
                    'o', color=line.get_color(), markersize=8)
        ax3.set_title("E3 Segment Means")
        ax3.set_xlabel("Frame Number")
        ax3.set_ylabel("Strain Value")
        ax3.grid(True)
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        # Increase y-axis range from -20-20 to -25-25
        ax3.set_ylim(-40, 40)
        if num_frames > 10:
            ax3.set_xticks(frame_numbers[::step])
        else:
            ax3.set_xticks(frame_numbers)
        ax3.set_xlim(frame_numbers[0] - 0.5, frame_numbers[-1] + 0.5)

        plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust right margin for legend
        plot_buffer = io.BytesIO()
        plt.savefig(plot_buffer, format="png", dpi=100, bbox_inches='tight')
        plt.close(fig)
        plot_buffer.seek(0)
        c.drawImage(ImageReader(plot_buffer), 50, 50, 500, 300)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating plot: {str(e)}")

    # Finalize PDF
    c.showPage()
    c.save()
    pdf_buffer.seek(0)

    return StreamingResponse(
        pdf_buffer,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=strain_report_frame_{current_frame_number}.pdf"}
    )
