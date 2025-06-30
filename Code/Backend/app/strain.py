import numpy as np
import pydicom
from pydicom.filebase import DicomBytesIO
from reportlab.lib.pagesizes import A4
from scipy.ndimage import gaussian_filter
import base64


# Strain Calculation Functions
def enforce_full_principal_strain_order(Ep1All: np.ndarray, Ep2All: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Ensure Ep1All >= Ep2All by swapping where necessary."""
    mask = Ep1All < Ep2All
    Ep1All_new = np.where(mask, Ep2All, Ep1All)
    Ep2All_new = np.where(mask, Ep1All, Ep2All)
    return Ep1All_new, Ep2All_new

def compute_strains(FrameDisplX: np.ndarray, FrameDisplY: np.ndarray, deltaX: float, deltaY: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    print(f"Computing strains with deltaX={deltaX}, deltaY={deltaY}")
    print(f"FrameDisplX shape: {FrameDisplX.shape}, FrameDisplY shape: {FrameDisplY.shape}")

    dUx_dX, dUx_dY = np.gradient(FrameDisplX, deltaX, deltaY, axis=(0, 1))
    dUy_dX, dUy_dY = np.gradient(FrameDisplY, deltaX, deltaY, axis=(0, 1))
    print(f"Gradient shapes: dUx_dX={dUx_dX.shape}, dUx_dY={dUx_dY.shape}, dUy_dX={dUy_dX.shape}, dUy_dY={dUy_dY.shape}")
    print(f"dUx_dL    dX range: min={np.min(dUx_dX):.4f}, max={np.max(dUx_dX):.4f}")
    print(f"dUx_dY range: min={np.min(dUx_dY):.4f}, max={np.max(dUx_dY):.4f}")
    print(f"dUy_dX range: min={np.min(dUy_dX):.4f}, max={np.max(dUy_dX):.4f}")
    print(f"dUy_dY range: min={np.min(dUy_dY):.4f}, max={np.max(dUy_dY):.4f}")

    Fxx = 1 + dUx_dX
    Fxy = dUx_dY
    Fyx = dUy_dX
    Fyy = 1 + dUy_dY
    print(f"Fxx range: min={np.min(Fxx):.4f}, max={np.max(Fxx):.4f}")
    print(f"Fxy range: min={np.min(Fxy):.4f}, max={np.max(Fxy):.4f}")
    print(f"Fyx range: min={np.min(Fyx):.4f}, max={np.max(Fyx):.4f}")
    print(f"Fyy range: min={np.min(Fyy):.4f}, max={np.max(Fyy):.4f}")

    FT_F_xx = Fxx**2 + Fyx**2
    FT_F_xy = Fxx * Fxy + Fyx * Fyy
    FT_F_yx = Fxy * Fxx + Fyy * Fyx
    FT_F_yy = Fxy**2 + Fyy**2
    Exx = 0.5 * (FT_F_xx - 1)
    Eyy = 0.5 * (FT_F_yy - 1)
    Exy = 0.5 * (FT_F_xy)
    print(f"Exx range: min={np.min(Exx):.4f}, max={np.max(Exx):.4f}")
    print(f"Eyy range: min={np.min(Eyy):.4f}, max={np.max(Eyy):.4f}")
    print(f"Exy range: min={np.min(Exy):.4f}, max={np.max(Exy):.4f}")

    trace_E = Exx + Eyy
    det_E = Exx * Eyy - Exy**2
    sqrt_term = np.sqrt(((Exx - Eyy) / 2)**2 + Exy**2)
    Ep1All = trace_E / 2 + sqrt_term
    Ep2All = trace_E / 2 - sqrt_term
    print(f"Ep1All (pre-clamp) range: min={np.min(Ep1All):.4f}, max={np.max(Ep1All):.4f}")
    print(f"Ep2All (pre-clamp) range: min={np.min(Ep2All):.4f}, max={np.max(Ep2All):.4f}")

    Ep1All = np.clip(Ep1All, -0.5, 0.5)
    Ep2All = np.clip(Ep2All, -0.5, 0.5)
    Ep1All, Ep2All = enforce_full_principal_strain_order(Ep1All, Ep2All)
    print(f"Ep1All (post-clamp) range: min={np.min(Ep1All):.4f}, max={np.max(Ep1All):.4f}")
    print(f"Ep2All (post-clamp) range: min={np.min(Ep2All):.4f}, max={np.max(Ep2All):.4f}")

    denom = (1 + Ep1All) * (1 + Ep2All)
    denom = np.where(denom <= 0, 1e-6, denom)
    Ep3All = 1 / denom - 1
    Ep3All = np.clip(Ep3All, -0.5, 0.5)
    print(f"Ep3All range: min={np.min(Ep3All):.4f}, max={np.max(Ep3All):.4f}")

    volume_ratio = (1 + Ep1All) * (1 + Ep2All) * (1 + Ep3All)
    print(f"Incompressibility check: volume ratio min={np.min(volume_ratio):.4f}, max={np.max(volume_ratio):.4f}")

    return Ep1All, Ep2All, Ep3All

def create_localized_dicom(original_dicom: pydicom.Dataset, image_array: np.ndarray, series_description: str, series_uid: str, instance_number: int) -> pydicom.Dataset:
    new_dicom = pydicom.Dataset()
    for elem in original_dicom:
        if elem.tag.group != 0x0002 and elem.tag != (0x7fe0, 0x0010):  # Exclude file meta and pixel data
            new_dicom.add(elem)
    
    file_meta = pydicom.dataset.FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = original_dicom.file_meta.MediaStorageSOPClassUID
    file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = pydicom.uid.PYDICOM_IMPLEMENTATION_UID
    new_dicom.file_meta = file_meta
    
    new_dicom.SeriesInstanceUID = series_uid
    new_dicom.SeriesDescription = series_description
    new_dicom.InstanceNumber = instance_number
    new_dicom.SOPInstanceUID = pydicom.uid.generate_uid()
    new_dicom.Rows, new_dicom.Columns = image_array.shape
    new_dicom.SamplesPerPixel = 1
    new_dicom.PhotometricInterpretation = "MONOCHROME2"
    new_dicom.BitsAllocated = 16
    new_dicom.BitsStored = 16
    new_dicom.HighBit = 15
    new_dicom.PixelRepresentation = 0
    
    # Scale image_array to 16-bit range (0 to 65535)
    img_min, img_max = np.min(image_array), np.max(image_array)
    if img_max > img_min:
        image_scaled = ((image_array - img_min) / (img_max - img_min) * 65535).astype(np.uint16)
    else:
        image_scaled = np.zeros_like(image_array, dtype=np.uint16)  # Fallback for uniform arrays
    
    image_scaled = np.ascontiguousarray(image_scaled)
    new_dicom.PixelData = image_scaled.tobytes()
    
    # Set window center and width for visualization
    window_center = 32768.0  # Middle of 16-bit range
    window_width = 65535.0   # Full range
    new_dicom.WindowCenter = window_center
    new_dicom.WindowWidth = window_width
    
    # Log for debugging
    print(f"WindowCenter: {window_center}, WindowWidth: {window_width}")
    print(f"Image array min: {np.min(image_scaled)}, max: {np.max(image_scaled)}")
    
    return new_dicom


def dicom_to_base64(dicom_dataset: pydicom.Dataset) -> str:
    buffer = DicomBytesIO()
    pydicom.filewriter.dcmwrite(buffer, dicom_dataset, write_like_original=False)
    buffer.seek(0)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def create_strain_dicom_grayscale(
    original_dicom: pydicom.Dataset, 
    strain_array: np.ndarray, 
    series_description: str, 
    series_uid: str, 
    instance_number: int
) -> tuple[pydicom.Dataset, float, float, list, list, float, float]:
    # Check for NaN or infinite values in strain array
    if np.any(np.isnan(strain_array)) or np.any(np.isinf(strain_array)):
        print(f"Warning: NaN or infinite values detected in strain array. Replacing with zeros.")
        strain_array = np.nan_to_num(strain_array, nan=0.0, posinf=0.5, neginf=-0.5)

    # Compute the actual range of the strain array
    strain_min, strain_max = np.min(strain_array), np.max(strain_array)
    print(f"Strain array range for {series_description}: min={strain_min:.4f}, max={strain_max:.4f}")
    
    # Compute histogram and percentiles (keep for metadata)
    num_bins = 50
    hist_counts, hist_bins = np.histogram(strain_array, bins=num_bins, range=(strain_min, strain_max))
    hist_counts = hist_counts.tolist()
    hist_bins = hist_bins.tolist()
    strain_percentile_5 = np.percentile(strain_array, 5)
    strain_percentile_95 = np.percentile(strain_array, 95)
    
    # IMPORTANT: Use fixed mapping where -0.5 maps to 0, 0 maps to 32768, and 0.5 maps to 65535
    # This ensures consistent colormap application across all strain types
    strain_clipped = np.clip(strain_array, -0.5, 0.5)
    
    # Linear mapping: pixel_value = (strain_value + 0.5) * 65535
    strain_scaled = ((strain_clipped + 0.5) * 65535).astype(np.uint16)
    
    # Apply slight Gaussian smoothing
    strain_scaled = gaussian_filter(strain_scaled.astype(np.float32), sigma=0.5)
    strain_scaled = np.clip(strain_scaled, 0, 65535).astype(np.uint16)

    # Create new DICOM dataset
    new_dicom = pydicom.Dataset()
    for elem in original_dicom:
        if (elem.tag.group != 0x0002 and 
            elem.tag != (0x7fe0, 0x0010) and 
            elem.tag != (0x0028, 0x1050) and 
            elem.tag != (0x0028, 0x1051)):
            new_dicom.add(elem)

    file_meta = pydicom.dataset.FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = original_dicom.file_meta.MediaStorageSOPClassUID
    file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = pydicom.uid.PYDICOM_IMPLEMENTATION_UID
    new_dicom.file_meta = file_meta
    new_dicom.SeriesInstanceUID = series_uid
    new_dicom.SeriesDescription = series_description
    new_dicom.InstanceNumber = instance_number
    new_dicom.SOPInstanceUID = pydicom.uid.generate_uid()
    new_dicom.Rows, new_dicom.Columns = strain_array.shape
    new_dicom.SamplesPerPixel = 1
    new_dicom.PhotometricInterpretation = "MONOCHROME2"
    new_dicom.BitsAllocated = 16
    new_dicom.BitsStored = 16
    new_dicom.HighBit = 15
    new_dicom.PixelRepresentation = 0

    # Set pixel data
    new_dicom.PixelData = strain_scaled.tobytes()
    
    # Calculate pixel values for percentiles
    pixel_5th = (strain_percentile_5 + 0.5) * 65535
    pixel_95th = (strain_percentile_95 + 0.5) * 65535
    
    # Set window based on percentiles for better initial contrast
    window_width = pixel_95th - pixel_5th
    window_center = (pixel_95th + pixel_5th) / 2
    
    new_dicom.WindowCenter = window_center
    new_dicom.WindowWidth = window_width

    # Store strain range metadata
    new_dicom.ImageComments = f"Strain Range: min={strain_min:.4f}, max={strain_max:.4f}, p5={strain_percentile_5:.4f}, p95={strain_percentile_95:.4f}"

    print(f"Pixel mapping: strain 0 -> pixel 32768, strain -0.5 -> pixel 0, strain 0.5 -> pixel 65535")
    print(f"Window settings: center={window_center:.0f}, width={window_width:.0f}")
    
    return new_dicom, strain_min, strain_max, hist_bins, hist_counts, strain_percentile_5, strain_percentile_95