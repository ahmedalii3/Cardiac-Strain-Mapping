import os
from scipy.ndimage import center_of_mass
import numpy as np
import matplotlib.pyplot as plt
import math
class localize_simulated_only:
    def __init__(self,Frame_path, displacement_path):
        self.Frame_path = Frame_path
        self.displacement_path = displacement_path
        self.frames = os.listdir(self.Frame_path)
        self.displacements = os.listdir(self.displacement_path)
        self.frames.sort()
        self.displacements.sort()
        self.counter = 0
        self.visited = []
        self.localize()

    def localize(self):
        # for frame in self.frames:
        #     frame_path = os.path.join(self.Frame_path, frame)
        #     frame_id = "_".join(frame.split("_")[:-1])
        #     if frame_id == ".DS":
        #         continue
        #     centerofmass = self.get_center_of_mass(frame_id)
        #     if centerofmass == False:
        #         continue
        #     # if math.isnan(centerofmass[0]) or math.isnan(centerofmass[1]):
        #     #     plt.imshow(np.load(frame_path))
        #     #     plt.show()
        #     # Load the frame
        #     frame_data = np.load(frame_path)
           
        #     # frame_data = frame_data[...,0]
        #     print(frame_data.shape)
            
        

        #     # # # Crop the image
        #     # plt.imshow(frame_data)
        #     # plt.show()
        #     cropped_image = self.crop_image(frame_data, centerofmass)
        #     # save__path = os.path.join("/Users/ahmed_ali/Documents/GitHub/GP-2025-Strain/Data/Simulated_V2.1_small/Frames", frame)
        #     np.save(frame_path, cropped_image)
        
        for displacement in self.displacements:
            if displacement not in self.visited:
                displacement_path = os.path.join(self.displacement_path, displacement)
                displacement_id = "_".join(displacement.split("_")[:-1])
                self.visited.append(displacement_id)
                print(displacement_id)
                if displacement_id == ".DS":
                    continue
                centerofmass = self.get_center_of_mass(displacement_id)
                # Load the frame
                displacement_data_x = np.load(os.path.join(self.displacement_path, f"{displacement_id}_x.npy"))
                displacement_data_y = np.load(os.path.join(self.displacement_path, f"{displacement_id}_y.npy"))
                print(displacement_path)
                print(displacement_data_x.shape)
                # Crop the image
                cropped_image_x = self.crop_image(displacement_data_x, centerofmass)
                cropped_image_y = self.crop_image(displacement_data_y, centerofmass)

                print(cropped_image_x.shape)
                # save__path = os.path.join("/Users/ahmed_ali/Documents/GitHub/GP-2025-Strain/Code/Wave_SimulatorV2/generatedData/Cropped_Displacements", displacement)
                # os.makedirs(os.path.dirname(save__path), exist_ok=True)
                np.save(os.path.join(self.displacement_path, f"{displacement_id}_x.npy"), cropped_image_x)
                np.save(os.path.join(self.displacement_path, f"{displacement_id}_y.npy"), cropped_image_y)
                
    
    def crop_image(self, image, center_mass, size = 128):
        if image.shape[0] <= size or image.shape[1] <= size:
            return image
        cropx, cropy = size, size   # size of the cropped image
        y, x = center_mass
        print(f"Center of mass: {x}, {y}")  
        startx = int(x - cropx // 2)
        starty = int(y - cropy // 2)
        return image[starty:starty + cropy, startx:startx + cropx]
    
    def get_center_of_mass(self, frame_id):
        if os.path.exists(os.path.join(self.displacement_path, f"{frame_id}_x.npy")):

            x_disp = np.load(os.path.join(self.displacement_path, f"{frame_id}_x.npy"))
            y_disp = np.load(os.path.join(self.displacement_path, f"{frame_id}_y.npy"))

            magnitude = np.sqrt(x_disp**2 + y_disp**2)
            # Calculate the center of mass
            com = center_of_mass(magnitude)
            if math.isnan(com[0]) or math.isnan(com[1]):
                print(f"Frame {frame_id} has no displacement")
                # delete the frame and the displacement
                os.remove(os.path.join(self.displacement_path, f"{frame_id}_x.npy"))
                os.remove(os.path.join(self.displacement_path, f"{frame_id}_y.npy"))
                os.remove(os.path.join(self.Frame_path, f"{frame_id}_1.npy"))
                os.remove(os.path.join(self.Frame_path, f"{frame_id}_2.npy"))
                print(f"Frame {frame_id} has no displacement and has been deleted")
                return False
            return com
        else:
            return False
          

Frames_path = "/Users/ahmed_ali/Documents/GitHub/GP-2025-Strain/Code/Wave_SimulatorV2/generatedData/Frames"
Displacement_path = "/Users/ahmed_ali/Documents/GitHub/GP-2025-Strain/Data/Disp_temp"
localize = localize_simulated_only(Frames_path, Displacement_path)
            
        

