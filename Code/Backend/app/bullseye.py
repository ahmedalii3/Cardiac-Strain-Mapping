import os
import numpy as np
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from typing import List, Dict, Any
import matplotlib.cm as cm

def generate_bullseye_plots(
    series_arrays: List[np.ndarray],
    strain_arrays: Dict[str, np.ndarray],
    masks: List[Dict[str, Any]],
   
    num_strain_frames: int,
    ring: bool = False
) -> Dict[str, List[Dict[str, str]]]:
    """
    Generate Bull's Eye Plots for each strain type and strain frame, overlaying on localized series.
    Optionally generate ring-style plots. Save plots as PNGs in debug_dir/bullseye_plots and return base64-encoded images.
    
    Args:
        series_arrays: List of localized frame arrays (128x128).
        strain_arrays: Dictionary with 'Ep1All', 'Ep2All', 'Ep3All' strain arrays (128x128xN).
        masks: List of mask dictionaries with 'filename' and 'values' (128x128).
        num_strain_frames: Number of strain frames (e.g., 6 for frames 2–7).
        ring: If True, generate additional ring-style plots (default: False).
    
    Returns:
        Dictionary with 'bullseye1', 'bullseye2', 'bullseye3' lists containing base64 PNGs.
        If ring=True, includes 'bullseye1_ring', 'bullseye2_ring', 'bullseye3_ring' lists.
    """
    # bullseye_dir = os.path.join(debug_dir, "bullseye_plots")
    # os.makedirs(bullseye_dir, exist_ok=True)
    
    bullseye_plots = {
        "bullseye1": [],
        "bullseye2": [],
        "bullseye3": []
    }
    if ring:
        bullseye_plots.update({
            "bullseye1_ring": [],
            "bullseye2_ring": [],
            "bullseye3_ring": []
        })
    
    jet_cmap = cm.get_cmap('jet')
    
    # print(f"Generating Bull's Eye Plots in: {bullseye_dir}")
    for strain_idx in range(num_strain_frames):
        frame_idx = strain_idx + 1  # Frames 2–7
        print(f"Processing frame {frame_idx + 1} (strain index {strain_idx})")
        localized_frame = series_arrays[frame_idx]
        mask = np.array(masks[frame_idx]["values"], dtype=np.uint8)
        
        frame_min, frame_max = np.min(localized_frame), np.max(localized_frame)
        if frame_max > frame_min:
            normalized_frame = (localized_frame - frame_min) / (frame_max - frame_min)
        else:
            normalized_frame = np.zeros_like(localized_frame)
        
        cx, cy = 64, 64
        y, x = np.indices((128, 128))
        x = x - cx
        y = y - cy
        angles = (np.arctan2(y, x) * 180 / np.pi + 360) % 360
        radius = np.sqrt(x**2 + y**2)
        
        for strain_key, strain_name in [
            ("Ep1All", "bullseye1"),
            ("Ep2All", "bullseye2"),
            ("Ep3All", "bullseye3")
        ]:
            print(f"Processing strain: {strain_key}")
            strain_map = strain_arrays[strain_key][:, :, strain_idx] * 100
            
            # Compute segment means
            segment_means = np.zeros(6)
            segment_counts = np.zeros(6)
            for row in range(128):
                for col in range(128):
                    if mask[row, col]:
                        angle = angles[row, col]
                        segment = int(angle // 60)
                        segment_means[segment] += strain_map[row, col]
                        segment_counts[segment] += 1
            segment_means = np.divide(
                segment_means,
                segment_counts,
                out=np.zeros_like(segment_means),
                where=segment_counts != 0
            )
            
            # Standard Bull's Eye Plot
            segmented_map = np.zeros((128, 128))
            for row in range(128):
                for col in range(128):
                    if mask[row, col]:
                        angle = angles[row, col]
                        segment = int(angle // 60)
                        segmented_map[row, col] = segment_means[segment]
            
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.imshow(normalized_frame, cmap='gray')
            masked_strain = np.ma.masked_where(mask == 0, segmented_map)
            im = ax.imshow(masked_strain, cmap='jet', alpha=0.5, vmin=-30, vmax=30)
            
            for i in range(6):
                angle_deg = (i + 0.5) * 60
                angle_rad = np.deg2rad(angle_deg)
                radius_text = 40
                x_text = cx + radius_text * np.cos(angle_rad)
                y_text = cy + radius_text * np.sin(angle_rad)
                ax.text(
                    x_text, y_text,
                    f"{segment_means[i]:.1f}",
                    color='white',
                    ha='center', va='center',
                    fontsize=8,
                    bbox=dict(facecolor='black', alpha=0.5, pad=2)
                )
            
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Strain (%)')
            cbar.set_ticks([-30, 0, 30])
            ax.axis('off')
            
            # output_path = os.path.join(
            #     bullseye_dir,
            #     f"bullseye_{strain_name[-1]}_frame_{frame_idx + 1}.png"
            # )
            # plt.savefig(output_path, bbox_inches='tight', dpi=100)
            
            buffer = BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
            buffer.seek(0)
            img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            buffer.close()
            
            bullseye_plots[strain_name].append({
                "filename": f"bullseye_{strain_name[-1]}_frame_{frame_idx + 1}.png",
                "content": img_base64
            })
            
            plt.close(fig)
            # print(f"Generated Standard Bull's Eye Plot: {output_path}")
            
            # Ring Bull's Eye Plot
            if ring:
                ring_inner, ring_outer = 30, 55
                ring_mask = (radius >= ring_inner) & (radius <= ring_outer)
                ring_segmented_map = np.zeros((128, 128))
                
                for i in range(6):
                    theta1 = i * 60
                    theta2 = (i + 1) * 60
                    ring_segment_mask = (
                        (angles >= theta1) & (angles < theta2) &
                        ring_mask
                    )
                    ring_segmented_map[ring_segment_mask] = segment_means[i]
                
                fig, ax = plt.subplots(figsize=(4, 4))
                ax.imshow(ring_segmented_map, cmap='jet', vmin=-30, vmax=30)
                
                text_radius = (ring_inner + ring_outer) / 2
                for i in range(6):
                    angle_deg = (i + 0.5) * 60
                    angle_rad = np.deg2rad(angle_deg)
                    x_text = cx + text_radius * np.cos(angle_rad)
                    y_text = cy + text_radius * np.sin(angle_rad)
                    ax.text(
                        x_text, y_text,
                        f"{segment_means[i]:.1f}",
                        color='white',
                        ha='center', va='center',
                        fontsize=8,
                        bbox=dict(facecolor='black', alpha=0.5, pad=2)
                    )
                
                cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label('Strain (%)')
                cbar.set_ticks([-30, 0, 30])
                ax.axis('off')
                
                # ring_output_path = os.path.join(
                #     bullseye_dir,
                #     f"bullseye_{strain_name[-1]}_ring_frame_{frame_idx + 1}.png"
                # )
                # plt.savefig(ring_output_path, bbox_inches='tight', dpi=100)
                
                buffer = BytesIO()
                plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
                buffer.seek(0)
                ring_img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                buffer.close()
                
                bullseye_plots[f"{strain_name}_ring"].append({
                    "filename": f"bullseye_{strain_name[-1]}_ring_frame_{frame_idx + 1}.png",
                    "content": ring_img_base64
                })
                
                plt.close(fig)
                # print(f"Generated Ring Bull's Eye Plot: {ring_output_path}")

    return bullseye_plots