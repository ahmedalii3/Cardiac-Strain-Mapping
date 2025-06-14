import numpy as np
import matplotlib.pyplot as plt
import Strain_Calculate
from Strain_Calculate import limit_strain_range, enforce_full_principal_strain_order

def generate_bulls_eye(strain_map, mask,frame,ring = False, save_path='bulls_eye_plot.png'):
    assert strain_map.shape == mask.shape, "Strain map and mask must have the same shape"
    strain_map = strain_map * 100  # Convert to percentage
    h, w = strain_map.shape
    cx, cy = w // 2, h // 2  # center of the image
    
    # Create coordinate grid
    y, x = np.indices((h, w))
    x = x - cx
    y = y - cy


    angles = (np.arctan2(y, x) * 180 / np.pi + 360) % 360  # Convert to [0, 360)
    radius = np.sqrt(x**2 + y**2)
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
    if ring:
        # Step 2: Create thinner ring and fill with segment means
        r_inner = 45  # <- tighter inner radius
        r_outer = 55  # <- tighter outer radius
        ring_mask = (radius >= r_inner) & (radius <= r_outer)

        ring_segmented_map = np.zeros_like(strain_map, dtype=np.float32)

        for i in range(6):
            theta1 = i * 60
            theta2 = (i + 1) * 60
            ring_segment_mask = (
                (angles >= theta1) & (angles < theta2) &
                ring_mask
            )
            ring_segmented_map[ring_segment_mask] = segment_means[i]

        # Step 3: Plot ring and add labels
        plt.figure(figsize=(6, 6))

        plt.imshow(ring_segmented_map, cmap='jet', vmin=-30, vmax=30)

        # Add text annotations at center of each segment
        for i in range(6):
            angle_deg = (i + 0.5) * 60
            angle_rad = np.deg2rad(angle_deg)
            text_radius = (r_inner + r_outer) / 2
            x_text = cx + text_radius * np.cos(angle_rad)
            y_text = cy + text_radius * np.sin(angle_rad)
            val = segment_means[i]
            plt.text(x_text, y_text, f"{val:.3f}", ha='center', va='center', fontsize=10,
                    color='white', weight='bold')

        plt.title("Ring Bull's Eye (Anatomical Strain)")
        plt.axis('off')
        plt.colorbar(label='Strain')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return segment_means, ring_segmented_map
    # Plotting
    else:
        plt.figure(figsize=(6, 6))
        cmap = plt.cm.jet
        cmap.set_under(color='black')  # for background
        plt.imshow(frame, cmap='gray')  # Show the frame as background
        plt.imshow(segmented_map, cmap=cmap, vmin=-30, vmax=30, alpha=0.5)  # Overlay the segmented strain map
        # Add text annotations at center of each segment
        for i in range(6):
            theta1 = i * 60
            theta2 = (i + 1) * 60
            segment_mask = (
                (angles >= theta1) & (angles < theta2) &
                (mask > 0)
            )

            if np.any(segment_mask):
                # Get coordinates of pixels in the segment
                ys, xs = np.where(segment_mask)
                x_text = np.mean(xs)
                y_text = np.mean(ys)
                val = segment_means[i]
                plt.text(x_text, y_text, f"{val:.3f}", ha='center', va='center', fontsize=8,
                        color='white', weight='bold')

        plt.title("Bull's Eye Plot (6 Segments)")
        plt.axis('off')
        plt.colorbar(label='Strain')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        return segment_means, segmented_map


displacement_x = np.load("/Users/ahmed_ali/Documents/GitHub/GP-2025-Strain/Code/Wave_SimulatorV2/generatedData/Displacements/patient097_frame11_slice_5_ACDC_#3_x.npy")
displacement_y = np.load("/Users/ahmed_ali/Documents/GitHub/GP-2025-Strain/Code/Wave_SimulatorV2/generatedData/Displacements/patient097_frame11_slice_5_ACDC_#3_y.npy")
frame = np.load("/Users/ahmed_ali/Documents/GitHub/GP-2025-Strain/Code/Wave_SimulatorV2/generatedData/Frames/patient097_frame11_slice_5_ACDC_#3_2.npy")
_,_,final_tensor, _, max_initial_strain, max_strain, min_initial_strain, min_strain = limit_strain_range(displacement_x, displacement_y)
E1 = final_tensor['E1']
# strain_map = np.load("/Users/ahmed_ali/Documents/GitHub/GP-2025-Strain/Code/Archive/strain5/frame_5_strain.npy")
strain_map = E1
mask = np.load("/Users/ahmed_ali/Documents/GitHub/GP-2025-Strain/Code/Wave_SimulatorV2/generatedData/npy_masks/patient097_frame11_slice_5_ACDC_#3_2.npy")

print("Strain map shape:", strain_map.shape)
print("Mask shape:", mask.shape)
mask = mask == 1
plt.imshow(strain_map, cmap='jet', vmin=-0.3, vmax = 0.3)
plt.colorbar()
# plt.show()
plt.savefig('strain_map.png', dpi=300, bbox_inches='tight')
means, seg_img = generate_bulls_eye(strain_map, mask, frame, ring = False, save_path='bulls_eye_plot.png')

print("Mean strain values per segment:", means)