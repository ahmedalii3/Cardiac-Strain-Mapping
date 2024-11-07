import os
import pydicom
import numpy as np
import pandas as pd
from skimage.draw import polygon
import matplotlib.pyplot as plt


def load_dicom_image(dicom_path):
  dicom_image = pydicom.dcmread(dicom_path)
  return dicom_image

def load_contour(contour_path):
  contour = []
  with open(contour_path, 'r') as f:
    for line in f:
      x, y = line.strip().split()
      contour.append([float(x), float(y)])
  return np.array(contour)

def create_mask(dicom_image, inner_contour, outer_contour):
  mask = np.zeros(dicom_image.pixel_array.shape, dtype=np.uint8)
  
  if inner_contour is not None:
    rr, cc = polygon(inner_contour[:, 1] - 1, inner_contour[:, 0] - 1, mask.shape)
    mask[rr, cc] = 1
  
  if outer_contour is not None:
    rr, cc = polygon(outer_contour[:, 1] - 1, outer_contour[:, 0] - 1, mask.shape)
    mask[rr, cc] = np.where(mask[rr, cc] != 1, 2, mask[rr, cc])
  
  return mask

def display_mask(mask):
  plt.imshow(mask, cmap='gray')
  plt.title('Generated Mask')
  plt.show()

def overlay_mask_on_image(dicom_image, mask):
  fig, axes = plt.subplots(1, 2, figsize=(12, 6))
  
  _, axes = plt.subplots(1, 2, figsize=(12, 6))
  axes[0].imshow(dicom_image.pixel_array, cmap='gray')
  axes[0].set_title('Original DICOM Image')
  
  # Display DICOM image with mask overlay
  axes[1].imshow(dicom_image.pixel_array, cmap='gray')
  axes[1].imshow(mask, cmap='jet', alpha=0.5)  # Overlay mask with transparency
  axes[1].set_title('Mask Overlay on DICOM Image')
  
  plt.show()

def save_mask_as_numpy(mask_data, patient_id, frame_number, output_dir):
  patient_folder_name = f"patient{patient_id:04d}"
  patient_output_path = os.path.join(output_dir, patient_folder_name)
  os.makedirs(patient_output_path, exist_ok=True)
  
  folder_name = f"frame{frame_number:03d}_gt"
  output_path = os.path.join(patient_output_path, folder_name)
  os.makedirs(output_path, exist_ok=True)
  
  file_name = f"{folder_name}.npy"
  np.save(os.path.join(output_path, file_name), mask_data)
  print(f"Saved mask numpy file to {os.path.join(output_path, file_name)}")
  
  file_name = f"{folder_name}.npy"
  np.save(os.path.join(output_path, file_name), mask_data)
  print(f"Saved mask numpy file to {os.path.join(output_path, file_name)}")

def process_patient_contours(image_dir, contour_dir, output_dir, csv_path, max_patients=2):
  csv_path = os.path.join(os.path.dirname(__file__), csv_path)
  patient_data = pd.read_csv(csv_path)
  patient_mapping = dict(zip(patient_data['PatientID'], patient_data['OriginalID']))
  
  patient_count = 0

  # Check if the directory exists
  if not os.path.exists(image_dir):
      print(f"Error: The directory {image_dir} does not exist.")
  else:
      for patient_folder in os.listdir(image_dir):
          if patient_count >= max_patients:
              break

          patient_path = os.path.join(image_dir, patient_folder)
          if not os.path.isdir(patient_path):
              continue
          
          print(f"\nProcessing patient: {patient_folder}")
          original_id = patient_mapping.get(patient_folder)
          if not original_id:
              print(f"No mapping found for patient: {patient_folder}")
              continue
          
          original_id_padded = original_id if original_id[-2].isdigit() else original_id[:-1] + '0' + original_id[-1]
          contour_patient_folder = os.path.join(contour_dir, original_id_padded, 'contours-manual', 'IRCCI-expert')
          if not os.path.isdir(contour_patient_folder):
              print(f"Contour folder not found: {contour_patient_folder}")
              continue

          dicom_files = []
          for root, _, files in os.walk(patient_path):
              for file in files:
                  if file.endswith('.dcm') and 'CINESAX' in root:
                      dicom_files.append(os.path.join(root, file))
          if not dicom_files:
              print(f"No DICOM files found in: {patient_path}")
              continue

          for dicom_file in dicom_files:
              dicom_id = dicom_file.split('-')[-1].split('.')[0]
              dicom_image = load_dicom_image(dicom_file)
              print(f"Processing slice {dicom_id} for patient {patient_folder}")
              
              contours = {'inner': None, 'outer': None}
              for contour_file in os.listdir(contour_patient_folder):
                  if dicom_id in contour_file:
                      contour_path = os.path.join(contour_patient_folder, contour_file)
                      if 'icontour' in contour_file:
                          contours['inner'] = load_contour(contour_path)
                          print(f"Loaded inner contour from {contour_file}")
                      elif 'ocontour' in contour_file:
                          contours['outer'] = load_contour(contour_path)
                          print(f"Loaded outer contour from {contour_file}")

              if contours['inner'] is None:
                  print(f"No inner contour found for slice {dicom_id}")
              if contours['outer'] is None:
                  print(f"No outer contour found for slice {dicom_id}")

              if contours['inner'] is not None and contours['outer'] is not None:
                  mask_data = create_mask(dicom_image, contours['inner'], contours['outer'])
                  save_mask_as_numpy(mask_data, int(patient_folder[-4:]), int(dicom_id), output_dir)
                  overlay_mask_on_image(dicom_image, mask_data)

          patient_count += 1


# Set the current working directory to the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Set paths
image_dir = os.path.relpath(r'..\..\Data\SunnyBrook', start=r'.')
contour_dir = os.path.relpath(r'..\..\Data\SunnyBrook\SCD_ManualContours', start=r'.')
output_dir = os.path.relpath(r'..\..\Data\SunnyBrook\Generated_Masks', start=r'.')
csv_path = os.path.relpath(r'..\..\Data\SunnyBrook\scd_patientdata.csv', start=r'.')

# Process with a limit of 2 patients for testing
process_patient_contours(image_dir, contour_dir, output_dir, csv_path, max_patients=2)
