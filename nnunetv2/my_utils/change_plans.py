
import os
import copy
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import load_json, save_json
from sklearn.model_selection import KFold, StratifiedKFold

def update_nested_dict(original_dict, update_dict):
    """
    Creates a deep copy of the original dictionary and recursively updates it using values from the update dictionary
    only for corresponding keys that exist in both dictionaries. Returns the updated dictionary.

    :param original_dict: The original dictionary that will remain unchanged.
    :param update_dict: The dictionary with updates.
    :return: A new dictionary that is a deep copy of original_dict but updated according to update_dict.
    """
    # Create a deep copy of the original dictionary to work with
    updated_dict = copy.deepcopy(original_dict)

    # Define a recursive function to navigate through the dictionary
    def recursive_update(curr_original, curr_update):
        for key, value in curr_update.items():
            if key in curr_original:
                if isinstance(value, dict) and isinstance(curr_original[key], dict):
                    recursive_update(curr_original[key], value)
                else:
                    curr_original[key] = value

    # Call the recursive update function starting from the copied dictionary
    recursive_update(updated_dict, update_dict)
    return updated_dict
def change_plans(p_plans: str,
                 dct_change: dict,
                 f_new_plans: str = None):
    """
    p_plans : path to json file with nnUnetPlans for compiling and training model
    dct_change : dictionary mimicing p_plans dict but with different values, varying values are updated
    f_new_plans : if file path defined updated plans will be stored

    returns : updated plans using dct_change
    """

    plans = load_json(p_plans)
    # update_plansname
    plans['plans_name'] = f_new_plans.replace('.json', '')
    new_plans = update_nested_dict(plans, dct_change)

    p_new_plans = os.path.join(os.sep, *p_plans.split(os.sep)[:-1], f_new_plans)

    save_json(new_plans, p_new_plans)
    return new_plans

def generate_pixel_patchsize(spacing, mm_patch_size):
    spacing = np.array(spacing)
    mm_patch_size = np.array(mm_patch_size)
    vox_patch_size = [int(i) for i in (mm_patch_size / spacing)]
    return list(vox_patch_size)

def k_fold_split(X, k, y=None, random_seed=42):
    # Choose the splitting strategy based on the presence of y
    if y is not None:
        kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_seed)
    else:
        kf = KFold(n_splits=k, shuffle=True, random_state=random_seed)

    result = []
    # Generate indices for splitting
    for train_index, test_index in kf.split(X, y):
        dct = {}
        dct['train'] = [X[i] for i in train_index]
        dct['val'] = [X[i] for i in test_index]
        result.append(dct)
    return result