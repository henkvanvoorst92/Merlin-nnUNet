
import os
import json
import random
import pandas as pd
import numpy as np
import SimpleITK as sitk
import torch
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
import argparse
import yaml
from typing import Dict
from tqdm import tqdm
#torch.serialization.add_safe_globals([np._core.multiarray.scalar])

def load_yaml_config(yaml_file):
    with open(yaml_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

def update_args_with_yaml(args, yaml_config):
    if args is None:
        args = argparse.Namespace()

    for key, value in yaml_config.items():
        if isinstance(value, dict):
            # Store the entire dictionary as an attribute
            setattr(args, key, value)
            # Also flatten nested dictionary keys
            for sub_key, sub_value in value.items():
                arg_key = f"{key}_{sub_key}"
                setattr(args, arg_key, sub_value)
        else:
            setattr(args, key, value)
    return args

def init_args(args=None, print_args=True):
    # Create the parser
    parser = argparse.ArgumentParser(
        description='')

    # Required positional arguments
    parser.add_argument('--yml_args', type=str, default=None,
                        help='If a path to a yml file is defined this is used to override all other args')
    parser.add_argument('--job', default=None,
                        help='list of job numbers to process')
    parser.add_argument('--experiment', default=None,
                        help='for validation proc experiment name in Train folder')
    parser.add_argument('--fold', default=None,
                        help='for separate jobs per fold')
    parser.add_argument('--models', default=None,
                        help='dict with subpath to model and list of channel ext strings to use')
    parser.add_argument('--use_monai_inferers', action="store_true",
                        help='For large models monai inferers are needed')

    args = parser.parse_args(args)

    return args


def load_segmentation_model(p_seg_model, fold,
                            tile_step_size=.5,
                            checkpoint_name = 'checkpoint_final.pth',
                            gpu_id=None):
    if p_seg_model is not None:
        if os.path.exists(p_seg_model) and os.path.isdir(p_seg_model):


            predictor = nnUNetPredictor(
                tile_step_size=tile_step_size,
                use_gaussian=True,
                use_mirroring=True,
                #perform_everything_on_gpu=True,
                device=torch.device('cuda', 0) if gpu_id is None else gpu_id,
                verbose=False,
                verbose_preprocessing=False,
                allow_tqdm=False
            )
            # initializes the network architecture, loads the checkpoint
            predictor.initialize_from_trained_model_folder(
                p_seg_model,
                use_folds=fold,
                checkpoint_name=checkpoint_name
            )
        else:
            print('No model found for using hence no inference possible')
        return predictor


def image_or_path_load(img_or_path):
    if isinstance(img_or_path, sitk.Image):
        out = sitk.Image(img_or_path)
    elif isinstance(img_or_path, str):
        if os.path.exists(img_or_path):
            out = sitk.ReadImage(img_or_path)
        else:
            out = None
            #raise ValueError('does not exist:',img_or_path)
    else:
        out = None
        #raise ValueError("img_or_path is not sitk.Image or a path",img_or_path)
    if out is not None:
        out = sitk.Cast(out, sitk.sitkInt16)
    return out

def chunk_ids(ids, n_jobs):
    """
    Split a list of IDs into almost equal-sized chunks.

    Parameters:
        ids (list): The list of IDs to chunk.
        n_jobs (int): The number of chunks to create.

    Returns:
        list: A list of chunks, where each chunk is a list of IDs.
    """
    chunk_size = len(ids) // n_jobs
    remainder = len(ids) % n_jobs

    random.shuffle(ids)

    chunks = []
    start = 0
    for i in range(n_jobs):
        # Distribute the remainder across the first few chunks
        end = start + chunk_size + (1 if i < remainder else 0)
        chunks.append(ids[start:end])
        start = end

    return chunks

def assign_job_numbers(df, num_jobs):
    num_rows = len(df)
    jobs_per_group = num_rows // num_jobs  # Number of times each job number appears
    remainder = num_rows % num_jobs  # If rows are not perfectly divisible

    job_numbers = []

    for i in range(num_jobs):
        job_numbers.extend([i] * jobs_per_group)

    # Distribute remainder evenly
    job_numbers.extend(np.random.choice(range(num_jobs), remainder, replace=False))

    np.random.shuffle(job_numbers)  # Shuffle job numbers

    df["job"] = job_numbers
    return df

def create_input_file(args, n_jobs=10, image_dirs=['NCCT-3chan', 'dwi'],
                      input_file=None, save=True, ID_splitter='_'):
    dir_inp = os.path.join(args.p_out, args.image_dir)
    IDs = list(set([f.split(ID_splitter)[0] for f in os.listdir(dir_inp) if '.nii' in f]))

    # find the local files
    dct = {}
    na_imd = []
    for imd in image_dirs:
        tmpd = {}
        p_dir = os.path.join(args.p_out, imd)
        if not os.path.exists(p_dir):
            na_imd.append(imd)
            continue
        for f in os.listdir(p_dir):
            if '.nii' in f and not ('_0001' in f or '_0002' in f):
                ID = f.split('.')[0].split(ID_splitter)[0]
                if ID in IDs:
                    file = os.path.join(p_dir, f)
                    tmpd[ID] = file if os.path.exists(file) else None
        dct[imd] = tmpd
    df = pd.DataFrame.from_dict(dct)
    for imd in na_imd:
        df[imd] = None
    df = assign_job_numbers(df, n_jobs)
    if save:
        df.to_excel(input_file)

    return df

def fetch_IDs(dir_inp):
    return list(set([f.split('_')[0] for f in os.listdir(dir_inp) if '.nii' in f]))

def nnunet_input_output_files_list(IDs, channels, dir_in, dir_out, overwrite=False, ID_splitter='_'):

    files_in = []
    files_out = []

    for ID in IDs:
        ID_files = []
        for chan in channels:
            for f in os.listdir(dir_in):
                #print(f, ID, chan, '.nii' in f, chan in f)
                if '.nii' in f:
                    if str(ID)==f.split(ID_splitter)[0] and chan in f:
                        ID_files.append(os.path.join(dir_in, f))
        if len(ID_files) == len(channels):
            f_out = os.path.join(dir_out, f'{ID}.nii.gz')
            if not os.path.exists(f_out) or overwrite:
                files_in.append(ID_files)
                files_out.append(f_out)
        else:
            print(f"Not all channels found for ID {ID}\n Found: {ID_files}, expected: {channels}")

    return files_in, files_out

def load_ids_per_split(preprocess_folder):
    """
    Load all IDs per split (train, val, test) from an nnUNet preprocess folder.

    Parameters:
        preprocess_folder (str): Path to the nnUNet preprocess folder.

    Returns:
        dict: A dictionary with keys as split names ('train', 'val', etc.) and values as lists of IDs.
    """
    dataset_json_path = os.path.join(preprocess_folder, 'splits_final.json')

    if not os.path.exists(dataset_json_path):
        raise FileNotFoundError(f"'dataset.json' not found in {preprocess_folder}")

    with open(dataset_json_path, 'r') as f:
        dataset_info = json.load(f)

    splits = {f'fold_{i}':dataset_info[i] for i in range(len(dataset_info))} #dataset_info.get('splits', [])

    return splits

def get_nnUNet_paths(nnunet_dir, dataset):
    """
    Retrieve the paths for the preprocessed and trained models directories in an nnUNet directory.

    Parameters:
        nnunet_dir (str): Path to the base nnUNet directory.

    Returns:
        dict: A dictionary containing the paths for 'preprocessed' and 'trained_models'.
    """
    raw_path = os.path.join(nnunet_dir, 'nnUNet_raw', dataset)
    preprocessed_path = os.path.join(nnunet_dir, 'nnUNet_preprocessed', dataset)
    trained_models_path = os.path.join(nnunet_dir, 'nnUNet_trained_models', dataset)

    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw directory not found in {nnunet_dir}")
    if not os.path.exists(preprocessed_path):
        raise FileNotFoundError(f"Raw directory not found in {preprocessed_path}")
    if not os.path.exists(trained_models_path):
        raise FileNotFoundError(f"Trained models directory not found in {nnunet_dir}")

    return raw_path, preprocessed_path, trained_models_path

def get_experiments(train_dir):
    """
    Retrieve all experiments from the train_dir and extract their keys.

    Parameters:
        train_dir (str): Path to the directory containing experiment folders.

    Returns:
        dict: A dictionary where keys are the extracted experiment keys and values are the full folder names.
    """
    experiments = {}
    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Train directory not found: {train_dir}")

    for folder in os.listdir(train_dir):
        parts = folder.split("__")
        if folder.startswith("MynnUNetTrainer") and "__" in folder:

            if len(parts) > 1:
                key = parts[0].split("_")[-1]  # Extract the key (e.g., 't2')
                experiments[key] = os.path.join(train_dir,folder)
        elif folder.startswith("nnUNetTrainer"):
            if parts[1] != 'nnUNetPlans':
                experiments['t0_'+parts[1]] = os.path.join(train_dir,folder)
            else:
                experiments['t0'] = os.path.join(train_dir,folder)
    return experiments

def get_path_dict(path, ID_splitter='_', filext='.graphml', incl_str=''):

    path_dct = {}
    for f in os.listdir(path):
        if filext is not None:
            if not f.endswith(filext):
                continue

        if incl_str in f:
            ID = str(f.split('.')[0].split(ID_splitter)[0])
            path_dct[ID] = os.path.join(path,f)

    return path_dct

class NiftiLoader:
    def __init__(self, root_dir: str, ID_splitter='_', incl_str='', filext='.nii.gz'):
        """
        Initialize the NiftiLoader

        Parameters:
        - root_dir (str): The root directory containing .nii.gz files.
        - id_list (List[str]): A list of identifiers to filter filenames.
        """
        self.root_dir = root_dir
        self.ID_splitter = ID_splitter
        self.incl_str = incl_str
        self.filext = filext

        self.file_paths = self._find_files()

    def _find_files(self) -> Dict[str, str]:
        """
        Search for .nii.gz files in the root directory that contain any of the specified IDs.

        Returns:
        - Dict[str, str]: A dictionary mapping IDs to their corresponding file paths.
        """
        return get_path_dict(self.root_dir, ID_splitter=self.ID_splitter, filext=self.filext, incl_str=self.incl_str)

    def __call__(self, ID: str) -> sitk.Image:
        """
        Load a specific image by ID.

        Parameters:
        - ID (str): The identifier for the image to load.

        Returns:
        - sitk.Image: The loaded SimpleITK Image object, or None if not found.
        """
        if ID in self.file_paths:
            return sitk.ReadImage(self.file_paths[ID])
        else:
            print(f"Image with ID {ID} not found.")
            return None

class NumpyLoader:
    def __init__(self, root_dir: str, ID_splitter='_', incl_str=''):
        """
        Initialize the NumpyLoader.

        Parameters:
        - root_dir (str): The root directory containing .npy files.
        - ID_splitter (str): The character used to split the file name to extract the ID.
        - incl_str (str): A string that must be included in the file name to be considered.
        """
        self.root_dir = root_dir
        self.ID_splitter = ID_splitter
        self.incl_str = incl_str

        self.file_paths = self._find_files()

    def _find_files(self) -> Dict[str, str]:
        """
        Search for .npy files in the root directory that contain the specified inclusion string.

        Returns:
        - Dict[str, str]: A dictionary mapping IDs to their corresponding file paths.
        """
        path_dict = {}
        for f in os.listdir(self.root_dir):
            if f.endswith('.npy') and self.incl_str in f:
                ID = f.split('.')[0].split(self.ID_splitter)[0]
                path_dict[ID] = os.path.join(self.root_dir, f)
        return path_dict

    def __call__(self, ID: str) -> np.ndarray:
        """
        Load a specific .npy file by ID.

        Parameters:
        - ID (str): The identifier for the file to load.

        Returns:
        - np.ndarray: The loaded NumPy array, or None if not found.
        """
        if ID in self.file_paths:
            return np.load(self.file_paths[ID])
        else:
            print(f"File with ID {ID} not found.")
            return None

def load_segmentation_model(p_seg_model, fold,
                            tile_step_size=.5,
                            checkpoint_name = 'checkpoint_final.pth',
                            gpu_id=None):

    if p_seg_model is not None:
        if os.path.exists(p_seg_model) and os.path.isdir(p_seg_model):

            predictor = nnUNetPredictor(
                tile_step_size=tile_step_size,
                use_gaussian=True,
                use_mirroring=True,
                #perform_everything_on_gpu=True,
                device=torch.device('cuda', 0) if gpu_id is None else gpu_id,
                verbose=False,
                verbose_preprocessing=False,
                allow_tqdm=False
            )
            # initializes the network architecture, loads the checkpoint
            predictor.initialize_from_trained_model_folder(
                p_seg_model,
                use_folds=[fold] if isinstance(fold, int) else fold,
                checkpoint_name=checkpoint_name
            )
        else:
            print('No model found for using hence no inference possible')
        return predictor

def image_or_path_load(img_or_path):
    if isinstance(img_or_path, sitk.Image):
        out = sitk.Image(img_or_path)
    elif isinstance(img_or_path, str):
        if os.path.exists(img_or_path):
            out = sitk.ReadImage(img_or_path)
        else:
            out = None
            #raise ValueError('does not exist:',img_or_path)
    else:
        out = None
        #raise ValueError("img_or_path is not sitk.Image or a path",img_or_path)
    if out is not None:
        out = sitk.Cast(out, sitk.sitkInt16)
    return out

def np2sitk(arr: np.ndarray, original_img: sitk.SimpleITK.Image):
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing(original_img.GetSpacing())
    img.SetOrigin(original_img.GetOrigin())
    img.SetDirection(original_img.GetDirection())
    # this does not allow cropping (such as removing thorax, neck)
    #img.CopyInformation(original_img)
    return img

def combine_excel_files(p_dir, incl_string='.xlsx'):
    return pd.concat([pd.read_excel(os.path.join(p_dir, f)) for f in tqdm(os.listdir(p_dir)) if incl_string in f], axis=0)

def rename_result_columns(data, rename_dct=None):

    if rename_dct is None:
        rename_dct = {
            'Class': {0:'Background', 1:'Artery', 2:'Vein'}
        }
    for col, mapping in rename_dct.items():
        if col in data.columns:
            data[col] = data[col].replace(mapping)

    return data

def write_multitab_excel(dataframes, file_path, index=True):
    """
    Write multiple pandas DataFrames to an Excel file with separate sheets.

    Parameters:
    - dataframes: dict, where keys are sheet names and values are DataFrames.
    - file_path: str, path to save the Excel file.
    """
    with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
        for sheet_name, df in dataframes.items():
            df.to_excel(writer, sheet_name=sheet_name, index=index)
