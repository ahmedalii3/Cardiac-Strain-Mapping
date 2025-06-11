import numpy as np
import matplotlib.pyplot as plt
import Strain_Calculate
from Strain_Calculate import limit_strain_range, enforce_full_principal_strain_order

def generate_bulls_eye(strain_map, mask, save_path='bulls_eye_plot.png'):
    assert strain_map.shape == mask.shape, "Strain map and mask must have the same shape"
    
    h, w = strain_map.shape
    cx, cy = w // 2, h // 2  # center of the image
    
    # Create coordinate grid
    y, x = np.indices((h, w))
    x = x - cx
    y = y - cy

    angles = (np.arctan2(y, x) * 180 / np.pi + 360) % 360  # Convert to [0, 360)
    
    segment_means = []
    segmented_map = np.zeros_like(strain_map, dtype=np.float32)

    for i in range(6):
        theta1 = i * 60
        theta2 = (i + 1) * 60

        # Create segment mask: in myocardium and within angle range
        segment_mask = (
            (angles >= theta1) & (angles < theta2) &
            (mask > 0)
        )

        # Get mean strain in this segment
        if np.any(segment_mask):
            mean_val = np.mean(strain_map[segment_mask])
        else:
            mean_val = 0.0  # or np.nan
        
        segment_means.append(mean_val)

        # Assign mean strain to this segment for visualization
        segmented_map[segment_mask] = mean_val

    # Plotting
    plt.figure(figsize=(6, 6))
    cmap = plt.cm.jet
    cmap.set_under(color='black')  # for background
    plt.imshow(segmented_map, cmap=cmap, vmin=-0.3, vmax=0.3)
    plt.title("Bull's Eye Plot (6 Segments)")
    plt.axis('off')
    plt.colorbar(label='Strain')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    return segment_means, segmented_map


displacement_x = np.load("/Users/ahmed_ali/Documents/GitHub/GP-2025-Strain/Code/Wave_SimulatorV2/generatedData/Displacements/patient096_frame01_slice_4_ACDC_#10_x.npy")
displacement_y = np.load("/Users/ahmed_ali/Documents/GitHub/GP-2025-Strain/Code/Wave_SimulatorV2/generatedData/Displacements/patient096_frame01_slice_4_ACDC_#10_y.npy")
_,_,final_tensor, _, max_initial_strain, max_strain, min_initial_strain, min_strain = limit_strain_range(displacement_x, displacement_y)
E1 = final_tensor['E1']
# strain_map = np.load("/Users/ahmed_ali/Documents/GitHub/GP-2025-Strain/Code/Archive/strain5/frame_5_strain.npy")
strain_map = E1
mask = np.load("/Users/ahmed_ali/Documents/GitHub/GP-2025-Strain/Code/Wave_SimulatorV2/generatedData/npy_masks/patient096_frame01_slice_4_ACDC_#10_2.npy")

print("Strain map shape:", strain_map.shape)
print("Mask shape:", mask.shape)
mask = mask == 1
plt.imshow(strain_map, cmap='jet', vmin=-0.3, vmax = 0.3)
plt.colorbar()
plt.show()
means, seg_img = generate_bulls_eye(strain_map, mask)
print("Mean strain values per segment:", means)