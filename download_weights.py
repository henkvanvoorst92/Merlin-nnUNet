'''
Download the weights for the Merlin nnU-Net trainer from the Hugging Face repository.
python download_weights.py
'''

from huggingface_hub import hf_hub_download
import os

def download_file(
    repo_id: str,
    filename: str,
    local_dir: str,
):
    os.makedirs(local_dir, exist_ok=True)
    local_file_path = hf_hub_download(
        repo_id=repo_id, filename=filename, local_dir=local_dir
    )
    print(f"{filename} downloaded and saved to {local_file_path}")
    return local_file_path


if __name__ == "__main__":
    files = [
        'nnUNetTrainerMerlin__nnUNetPlans__3d_fullres/dataset.json',
        'nnUNetTrainerMerlin__nnUNetPlans__3d_fullres/dataset_fingerprint.json',
        'nnUNetTrainerMerlin__nnUNetPlans__3d_fullres/plans.json',
        'nnUNetTrainerMerlin__nnUNetPlans__3d_fullres/fold_0/checkpoint_best.pth',
        'nnUNetTrainerMerlin__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth',
        'nnUNetTrainerMerlin__nnUNetPlans__3d_fullres/fold_0/debug.json',
        'nnUNetTrainerMerlin__nnUNetPlans__3d_fullres/fold_0/training_log_2024_11_13_14_46_33.txt'
    ]
    
    print(f"Downloading files to {os.path.join(os.getcwd(), 'nnUNetTrainerMerlin__nnUNetPlans__3d_fullres')}")
    
    os.makedirs(os.path.join(os.getcwd(), 'nnUNetTrainerMerlin__nnUNetPlans__3d_fullres'), exist_ok=True)
    os.makedirs(os.path.join(os.getcwd(), 'nnUNetTrainerMerlin__nnUNetPlans__3d_fullres', 'fold_0'), exist_ok=True)
    
    for file in files:        
        download_file(repo_id="stanfordmimi/Merlin", 
                      filename=file, 
                      local_dir=os.getcwd())
    print("Done downloading weights")