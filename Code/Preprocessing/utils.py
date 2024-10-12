import os
import glob

def get_frame_gt_dict(folder_path):
    # Get all file paths containing 'frame' (case-insensitive)
    files_with_frame = glob.glob(os.path.join(folder_path, '**', '*frame*'), recursive=True)
    
    frame_gt_list = []
    unmatched_frames = {}  # Temporary storage for unmatched frames/GTs
    
    # Separate the files into frames and their corresponding GT
    for file in files_with_frame:
        base_name = os.path.basename(file)  # Get the file name
        # Extract frame number (e.g., frame14 from patient101_frame14.nii or patient101_frame14_gt.nii)
        frame_number = base_name.split('_frame')[1].split('.')[0].replace('_gt', '')
        
        # Check if it's a ground truth file or a regular frame
        if '_gt' in base_name:
            # If the frame has already been found, pair it
            if frame_number in unmatched_frames:
                frame_gt_list.append({'frame': unmatched_frames.pop(frame_number), 'gt': file})
            else:
                # Otherwise, store the GT until the frame is found
                unmatched_frames[frame_number] = {'gt': file}
        else:
            # If the GT has already been found, pair it
            if frame_number in unmatched_frames:
                gt_file = unmatched_frames.pop(frame_number).get('gt')
                frame_gt_list.append({'frame': file, 'gt': gt_file})
            else:
                # Otherwise, store the frame until the GT is found
                unmatched_frames[frame_number] = {'frame': file}
    
    return frame_gt_list