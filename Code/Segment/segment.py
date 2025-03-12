import os
import sys
import subprocess
import importlib.util
from pathlib import Path
import numpy as np
import nibabel as nib

class segment ():
    def __init__(self, dataset_path = None, nifti_folder_path = None, load_only = False):
        self.dataset_path = dataset_path
        self.nifti_folder_path = nifti_folder_path
        self.load_only = load_only
        
        self.current_script = Path(__file__)
        self.imagesTs_path = self.current_script.parent / "nnUNetFrame/dataset/nnUNet_raw/Dataset007_ShortAX/imagesTs"

        # self.install_nnUnet()
        # self.set_global_variables()
        # self.predict_masks()
        # self.load_dataset()
    def install_nnUnet(self, package = "nnunetv2"):
        if importlib.util.find_spec(package) is None:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        else:
            print(f"{package} is already installed!")

    def set_global_variables(self):
        
        nnUnet_raw_path = self.current_script.parent / "nnUNetFrame/dataset/nnUNet_raw"
        nnUnet_preprocessed_path = self.current_script.parent / "nnUNetFrame/dataset/nnUNet_preprocessed"
        nnUnet_results_path = self.current_script.parent / "nnUNetFrame/dataset/nnUNet_trained_models"
        os.environ["nnUNet_raw"] = str(nnUnet_raw_path)
        os.environ["nnUNet_preprocessed"] = str(nnUnet_preprocessed_path)
        os.environ["nnUNet_results"] = str(nnUnet_results_path)

    def predict_masks(self):
        imagesTs_path = self.current_script.parent / "nnUNetFrame/dataset/nnUNet_raw/Dataset007_ShortAX/imagesTs"
        pred_nnUnet_path = self.current_script.parent / "nnUNetFrame/dataset/nnUNet_raw/Dataset007_ShortAX/pred_nnUNet"
        command = [
            "nnUNetv2_predict",
            "-i", str(imagesTs_path),
            "-o", str(pred_nnUnet_path),
            "-d", "Dataset007_ShortAX",
            "-c", "2d",
            "-f", "all",
            "-chk", "checkpoint_best.pth"
        ]

        # Run the command
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            print("Error running nnUNetv2_predict:", e)

    def load_dataset(self):
        # load dataset slices
        case_to_name_dict = {}
        directories = os.listdir(self.dataset_path)
        
        for i, slice in enumerate(directories):
            # check if fixed or moving
            id = slice.split("_")[-1]
            if id == "1.npy" :
                continue
            elif id == "2.npy" or id == "slice3.npy":
                image = np.load(self.dataset_path + "/" + slice, allow_pickle=True)
                affine = np.eye(4)
                nifti_image = nib.Nifti1Image(image, affine)
                if self.load_only:
                    nib.save(nifti_image, os.path.join(self.nifti_folder_path, f"case_{i+1}_0000.nii.gz"))
                else:
                    nib.save(nifti_image, os.path.join(str(self.imagesTs_path), f"case_{i+1}_0000.nii.gz"))
                case_to_name_dict[ f"case_{i+1}.nii.gz"] = slice
        # save numpy dictionary
        np.save(self.current_script.parent / "case_to_name_dict.npy", case_to_name_dict)
        print("Dataset loaded successfully!")

    def convert_nifti_back_to_numpy(self):
        # convert nifti back to numpy
        # create mask paths in the parent directory
        mask_path = self.current_script.parent / "npy_masks"
        mask_path.mkdir(parents=True, exist_ok=True)

        mask_path = self.current_script.parent / "nnUNetFrame/dataset/nnUNet_raw/Dataset007_ShortAX/pred_nnUNet"

        case_to_name_dict = np.load(self.current_script.parent / "case_to_name_dict.npy", allow_pickle=True).item()
        for case in case_to_name_dict:
            nifti_image = nib.load(mask_path / case)
            mask = nifti_image.get_fdata()
            np.save(os.path.join(self.current_script.parent / "npy_masks", case_to_name_dict[case]), mask)
                

# try segment class
if __name__ == "__main__":
    segment = segment("/Users/ahmed_ali/Documents/GitHub/GP-2025-Strain/Code/FrameWork/real_test_data")
    # segment.load_dataset()
    # segment.install_nnUnet()
    # segment.set_global_variables()
    # segment.predict_masks()
    segment.convert_nifti_back_to_numpy()

    