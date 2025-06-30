import os
import numpy as np
import pydicom
from fastapi import FastAPI, UploadFile, File, HTTPException
from typing import List
from fastapi.middleware.cors import CORSMiddleware
import SimpleITK as sitk
from pathlib import Path
import base64
from io import BytesIO
from .preprocessing import preprocess_series_in_memory, normalize_series_in_memory, pad_series_in_memory
from .ventricular_short_axis_3label.docs.localize import Localize

# Directories setup
UPLOAD_DIR = "temp/uploaded_files"
PROCESSED_NP_DIR = "temp/processed_numpy_files"
LOCALIZED_DIR = "temp/localized_files"
STRAIN_DIR = "temp/strain_temp"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PROCESSED_NP_DIR, exist_ok=True)
os.makedirs(LOCALIZED_DIR, exist_ok=True)

def get_metadata(dicom_data):
    """Extract metadata from DICOM file."""
    return {
        "StudyInstanceUID": str(dicom_data.StudyInstanceUID),
        "SeriesInstanceUID": str(dicom_data.SeriesInstanceUID),
        "InstanceNumber": str(dicom_data.InstanceNumber),
        "FrameNumber": str(getattr(dicom_data, 'FrameNumber', '')),
        "PatientName": str(dicom_data.PatientName),
        "PatientID": str(dicom_data.PatientID),
        "Modality": str(dicom_data.Modality),
        "StudyDescription": str(getattr(dicom_data, 'StudyDescription', '')),
        "SeriesDescription": str(getattr(dicom_data, 'SeriesDescription', '')),
        "SliceThickness": str(getattr(dicom_data, 'SliceThickness', '')),
        "SliceLocation": str(getattr(dicom_data, 'SliceLocation', '')),
        "PixelSpacing": str(getattr(dicom_data, 'PixelSpacing', '')),
        "Coordinates": (dicom_data.ImagePositionPatient[0], dicom_data.ImagePositionPatient[1],
                        dicom_data.ImagePositionPatient[2]),
        "Resolution": (dicom_data.Rows, dicom_data.Columns),
        "FrameOfReferenceUID": str(getattr(dicom_data, 'FrameOfReferenceUID', '')),
        "WindowLevel": str(getattr(dicom_data, 'WindowCenter', '')),
        "WindowWidth": str(getattr(dicom_data, 'WindowWidth', '')),
        "ContentDate": str(getattr(dicom_data, 'ContentDate', '')),
        "SeriesNumber": str(getattr(dicom_data, 'SeriesNumber', '')),
        "AquisitionTime": str(getattr(dicom_data, 'AquisitionTime', '')),
        "TR": str(getattr(dicom_data, 'RepetitionTime', '')),
        "TE": str(getattr(dicom_data, 'EchoTime', '')),

    }


def transform_coordinates(center_mass, original_shape, preprocessed_shape):
    """
    Transform coordinates from preprocessed image space to original image space.
    
    Args:
        center_mass (tuple): (x, y) coordinates in preprocessed space
        original_shape (tuple): Shape of original image (height, width)
        preprocessed_shape (tuple): Shape of preprocessed image (height, width)
    
    Returns:
        tuple: (x, y) coordinates in original image space
    """
    # Calculate scale factors
    scale_x = original_shape[1] / preprocessed_shape[1]
    scale_y = original_shape[0] / preprocessed_shape[0]
    
    # Transform coordinates
    original_x = center_mass[0] * scale_x
    original_y = center_mass[1] * scale_y
    
    return (original_x, original_y)

def process_dicom_file(dicom_data):
    """Process a single DICOM file and return preprocessed numpy array."""
    standardized_data = preprocess_series_in_memory(dicom_data, target_resolution=[1.0, 1.0])
    normalized_data = normalize_series_in_memory(standardized_data)
    padded_data = pad_series_in_memory(normalized_data, target_resolution=[512, 512])
    
    # Convert to numpy and ensure correct shape
    padded_data_np = sitk.GetArrayFromImage(padded_data)
    if padded_data_np.shape != (1, 512, 512):
        padded_data_np = np.expand_dims(padded_data_np, axis=0)
    
    return padded_data_np

def crop_to_center_of_mass(image_array, center_mass, crop_size=(128, 128)):
    """
    Crop a region around the center of mass from the image array.
    
    Args:
        image_array (np.ndarray): Input image array
        center_mass (tuple): (x, y) coordinates of center of mass
        crop_size (tuple): Desired size of cropped image (height, width)
    
    Returns:
        np.ndarray: Cropped image array
    """
    if len(image_array.shape) == 2:
        image_array = image_array[np.newaxis, :, :]
    
    h, w = image_array.shape[-2:]
    crop_h, crop_w = crop_size
    
    # Calculate crop boundaries
    x_center, y_center = int(center_mass[0]), int(center_mass[1])
    x_start = max(0, x_center - crop_w // 2)
    x_end = min(w, x_center + crop_w // 2)
    y_start = max(0, y_center - crop_h // 2)
    y_end = min(h, y_center + crop_h // 2)
    
    # Handle edge cases where crop window exceeds image boundaries
    if x_end - x_start < crop_w:
        if x_start == 0:
            x_end = crop_w
        else:
            x_start = w - crop_w
    if y_end - y_start < crop_h:
        if y_start == 0:
            y_end = crop_h
        else:
            y_start = h - crop_h
    
    # Perform cropping
    cropped = image_array[:, y_start:y_end, x_start:x_end]
    
    # Ensure exact crop size
    if cropped.shape[-2:] != crop_size:
        temp = np.zeros((cropped.shape[0], crop_h, crop_w))
        temp[:, :cropped.shape[1], :cropped.shape[2]] = cropped
        cropped = temp
    
    return cropped

def create_localized_dicom(original_dicom, image_data, new_series_uid, instance_number):
    """Create a new DICOM file with localized image data."""
    new_dicom = pydicom.Dataset()
    new_dicom.file_meta = pydicom.Dataset()
    
    # Copy all existing attributes from original DICOM
    for elem in original_dicom:
        if elem.tag != (0x7fe0, 0x0010):  # Skip pixel data
            new_dicom.add(elem)
    
    # Update necessary attributes
    new_dicom.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
    new_dicom.file_meta.MediaStorageSOPClassUID = original_dicom.SOPClassUID
    new_dicom.file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    new_dicom.file_meta.ImplementationClassUID = pydicom.uid.PYDICOM_IMPLEMENTATION_UID
    
    # Set new series-specific attributes
    new_dicom.SeriesInstanceUID = new_series_uid
    new_dicom.SeriesDescription = "Localized Series"
    new_dicom.InstanceNumber = instance_number
    new_dicom.SOPInstanceUID = new_dicom.file_meta.MediaStorageSOPInstanceUID
    
    # Set image-specific attributes
    new_dicom.Rows = image_data.shape[0]
    new_dicom.Columns = image_data.shape[1]
    new_dicom.PixelData = image_data.astype(np.uint16).tobytes()
    
    return new_dicom

def save_dicom_file(dicom_dataset, filepath):
    """Save DICOM dataset with proper encoding."""
    dicom_dataset.is_little_endian = True
    dicom_dataset.is_implicit_VR = False
    dicom_dataset.save_as(filepath, write_like_original=False)

async def process_uploaded_files(files):
    """Process and save uploaded DICOM files."""
    processed_files = []
    original_shapes = []
    
    for file in files:
        # Save uploaded file
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)
        
        # Read DICOM and store original dimensions
        dicom_data = pydicom.dcmread(file_path, force=True)
        original_shapes.append((dicom_data.Rows, dicom_data.Columns))
        
        # Process and save numpy array
        processed_np = process_dicom_file(dicom_data)
        file_name = os.path.splitext(file.filename)[0]
        print(f"Processing file: {file.filename}")
        print(f"Processed file: {file_name}")
        np_path = os.path.join(PROCESSED_NP_DIR, f"{file_name}.npy")
        np.save(np_path, processed_np)
        
        processed_files.append({
            'dicom_path': file_path,
            'np_path': np_path,
            'dicom_data': dicom_data
        })
    
    return original_shapes, processed_files

def get_center_of_mass(processed_files):
    """Calculate center of mass using the localization model."""
    localize = Localize(PROCESSED_NP_DIR)
    return localize.get_center_mass_avg()

def create_localized_series(processed_files, original_shapes, preprocessed_center, new_series_uid, series_dir):
    """Create localized DICOM series from processed files."""
    localized_files = []
    
    for idx, (file_info, original_shape) in enumerate(zip(processed_files, original_shapes)):
        try:
            # Transform coordinates to original image space
            original_center = transform_coordinates(
                preprocessed_center,
                original_shape,
                (512, 512)
            )
            
            # Read and crop original image
            image = sitk.ReadImage(file_info['dicom_path'])
            array = sitk.GetArrayFromImage(image)
            cropped = crop_to_center_of_mass(array, original_center)
            
            # Create and save new DICOM
            new_dicom = create_localized_dicom(
                file_info['dicom_data'],
                cropped[0],
                new_series_uid,
                idx + 1
            )
            
            output_path = os.path.join(series_dir, f"IMG{idx:04d}.dcm")
            save_dicom_file(new_dicom, output_path)
            
            # Prepare response data
            with open(output_path, "rb") as f:
                file_content = f.read()
                localized_files.append({
                    "filename": f"IMG{idx:04d}.dcm",
                    "content": base64.b64encode(file_content).decode('utf-8')
                })
                
        except Exception as e:
            print(f"Error processing file {file_info['dicom_path']}: {str(e)}")
            continue
            
    return localized_files

def cleanup_temp_files(processed_files):
    """Clean up temporary files after processing."""
    for file_info in processed_files:
        for path in [file_info.get('np_path'), file_info.get('dicom_path')]:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except Exception as e:
                    print(f"Error cleaning up {path}: {str(e)}")
