# import os
# import re

# def count_manual_contours(folder_path):
#     inner_contours_count = 0
#     outer_contours_count = 0

#     # Regex to match inner and outer contour files
#     inner_contour_pattern = re.compile(r"IM-\d{4}-\d{4}-icontour-manual\.txt")
#     outer_contour_pattern = re.compile(r"IM-\d{4}-\d{4}-ocontour-manual\.txt")

#     # Loop through files in the specified folder
#     for filename in os.listdir(folder_path):
#         if inner_contour_pattern.match(filename):
#             inner_contours_count += 1
#         elif outer_contour_pattern.match(filename):
#             outer_contours_count += 1

#     total_contours = inner_contours_count + outer_contours_count

#     print(f"Total inner contours: {inner_contours_count}")
#     print(f"Total outer contours: {outer_contours_count}")
#     print(f"Total contours: {total_contours}")

# # Replace 'your_folder_path' with the actual path to the folder containing the contour files
# folder_path = '/Users/ahmed_ali/Documents/GitHub/GP-2025-Strain/Data/SunnyBrook/SCD_ManualContours/SC-HF-I-01/contours-manual/IRCCI-expert'
# count_manual_contours(folder_path)

import os
import re

def count_manual_contours_in_patient_folder(patient_folder_path):
    inner_contours_count = 0
    outer_contours_count = 0

    # Regex to match inner and outer contour files
    inner_contour_pattern = re.compile(r"IM-\d{4}-\d{4}-icontour-manual\.txt")
    outer_contour_pattern = re.compile(r"IM-\d{4}-\d{4}-ocontour-manual\.txt")

    # Loop through files in the specified folder
    for filename in os.listdir(patient_folder_path):
        if inner_contour_pattern.match(filename):
            inner_contours_count += 1
        elif outer_contour_pattern.match(filename):
            outer_contours_count += 1

    total_contours = inner_contours_count + outer_contours_count

    return inner_contours_count, outer_contours_count, total_contours

def count_manual_contours(base_path):
    # Path to the contours-manual directory
    contours_manual_path = os.path.join(base_path, 'contours-manual')

    # Loop through patient folders within contours-manual
    for patient_folder in os.listdir(contours_manual_path):
        patient_folder_path = os.path.join(contours_manual_path, patient_folder, 'IRCCI-expert')

        if os.path.isdir(patient_folder_path):
            inner_count, outer_count, total_count = count_manual_contours_in_patient_folder(patient_folder_path)
            print(f"Patient: {patient_folder}")
            print(f"  Total inner contours: {inner_count}")
            print(f"  Total outer contours: {outer_count}")
            print(f"  Total contours: {total_count}\n")

# Replace 'your_base_path' with the actual path to the base directory containing 'contours-manual'
base_path = '/Users/ahmed_ali/Documents/GitHub/GP-2025-Strain/Data/SunnyBrook/SCD_ManualContours'
count_manual_contours(base_path)
