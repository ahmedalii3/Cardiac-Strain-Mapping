# Simulator Guide

### File Structure

- wave_generator.py : responsible for generating the wave
- create_displaced_mask.py : responsible for creating the displaced mask for all frames
- mask_dilation.py : responsible for dilating the displacement masks and filtering them using a gaussian filter
- wave_displacement_save.py : responsible for saving displacements for all frames
- wave_displacement_load.py : responsible for loading and applying displacements to the original image

- wave_displacement.py : responsible for applying the displacements to the original image straight away
- displ_strain_conversion.py : responsible for converting the displacements to strain maps (without modifying)
- strain_validation.py : responsible for validating the strain range(stretch or compress) and plotting the results

## Steps:

1. Run create_displaced_mask.py
   - Strain validation is controlled at line 121 -> comment or modify values
   - To visualzie the strain validation plots -> change self.plot_strain to True (line 34)
   - Closed manually when reaches frame 30 (Look in the terminal for the frame count)
2. Run mask_dilation.py
3. Run wave_displacement.py
   - if using stretching of strain -> run wave_displacement_save.py(close manually after frame 30) then wave_displacement_load.py as it can't be done in one go without lagging the visualization.
   - Controlling the strain range is at lines ~150-160 (varies by file)
   - To visualzie the strain validation plots -> change self.plot_strain to True (line 48)

#### - Visualizing the strain maps stops the displacement visualization.

#### - When using strain validation at the creation of the masks make sure to use the same validation ranges in the displacement applying later, to match the displaced masks with the displacement maps.

#### - To visualize strain maps after the 0.0 bar(background) -> uncomment lines 280-283 and control ranges in file strain_validation.py
