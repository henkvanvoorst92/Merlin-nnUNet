
import os
import itertools
import multiprocessing as mp
import pandas as pd
import ast
import numpy as np
import torch
from nnunetv2.my_utils.utils import init_args, update_args_with_yaml, load_yaml_config
from nnunetv2.my_utils.utils import image_or_path_load, load_segmentation_model, chunk_ids, assign_job_numbers, \
    create_input_file, fetch_IDs, nnunet_input_output_files_list

#torch.serialization.add_safe_globals([np._core.multiarray.scalar])
def segmentation_worker(inputs):

    for inpt in inputs:
        yml_file, input_files_or_dir, output_files_or_dir, model_dir, gpu_id, save_probabilities, fold, use_monai_inferers = inpt

        print('Model running:', model_dir)

        args = update_args_with_yaml(None, load_yaml_config(yml_file))

        if isinstance(gpu_id, int):
            gpu_id = torch.device('cuda', gpu_id)

        if fold is None:
            fold = ast.literal_eval((args.fold))

        nnunet_predictor = load_segmentation_model(model_dir, fold,
                                                    tile_step_size=args.tile_step_size,
                                                    checkpoint_name=args.checkpoint_name if hasattr(args, 'checkpoint_name') else 'checkpoint_best.pth',
                                                    gpu_id=gpu_id)

        nnunet_predictor.use_monai_inferers = use_monai_inferers

        #try:
        nnunet_predictor.predict_from_files(input_files_or_dir,
                                                output_files_or_dir,
                                                save_probabilities=save_probabilities,
                                                overwrite=args.overwrite,
                                                num_processes_preprocessing=1,
                                                num_processes_segmentation_export=1,
                                                folder_with_segs_from_prev_stage=None,
                                                num_parts=1,
                                                part_id=0)
        # except:
        #     print(f"Error during segmentation with model {model_dir} on files {input_files_or_dir}. Skipping.")
        #     continue


def main_segmentation_processor(job_inputs, n_procs=1):

    if n_procs>1:
        mp.set_start_method("spawn", force=True)
        procs = []
        for inputs in job_inputs:
            p = mp.Process(target=segmentation_worker, args=(inputs,))
            p.daemon = False   # <— make sure it can spawn its own children
            p.start()
            procs.append(p)

        for p in procs:
            p.join()
    else:
        # If only one job, run it directly
        segmentation_worker(job_inputs)


if __name__ == "__main__":
    args = init_args()
    args = update_args_with_yaml(args, load_yaml_config(args.yml_args))
    addname = '_'+args.addname if hasattr(args, 'addname') else ''

    """
    args should contain:
    model_dir: root dir where models are
    models: paths to separate models
    fold: fold to use for inference (0,1,2) or all or (0,1,2,3,4)
    tile_step_size: overlap (default 0.5)
    resolution: 'full_res' etc

    dir_out: output directory --> also dir in
    gpus: list of gpu ids that are available
    n_jobs: number of jobs to run in parallel (across gpus)

    """
    if hasattr(args, 'input_file'):
        input_file = os.path.join(args.p_out, args.input_file) if os.sep not in args.input_file else args.input_file
    else:
        input_file = ''

    if hasattr(args, 'image_folders'):
        image_dirs = args.image_folders
    elif hasattr(args, 'image_dir'):
        image_dirs = [args.image_dir]
    elif hasattr(args, 'image_type'):
        image_dirs = [args.image_type]
        args.image_dir = args.image_type

    if os.path.exists(input_file):
        df = pd.read_excel(input_file, index_col=0)
    else:
        df = create_input_file(args, image_dirs=image_dirs, input_file=input_file, ID_splitter=args.ID_splitter if hasattr(args, 'ID_splitter') else '_')

    #IDs = list(set([f.split('_')[0] for f in os.listdir('/media/hvv/71672b1c-e082-495c-b560-a2dfc7d5de59/data/BL_NCCT/CRISP2/processed_june25/iat_dwi_bl_seg_june25')]))
    #df[np.isin(df.index, IDs)].to_excel(input_file)

    if args.job is not None:
        #slice the part of the IDs out that represent the job
        job = ast.literal_eval(args.job)
        df = df[np.isin(df['job'], job)]
    IDs = df.index.tolist()

    #if to many models are used reduce the size of the total jobs (otherwise processing goes x len models)
    #this distributes multiple jobs within a gpu
    if args.n_jobs < len(args.models):
        n_jobs = 1
    else:
        n_jobs = args.n_jobs // len(args.models)
    ID_chunks = chunk_ids(IDs, n_jobs)

    gpu_cycle = itertools.cycle(args.gpus)
    job_inputs = []
    pp_jobs = []
    for model, channels in args.models.items():
        addname = '_' + args.addname if hasattr(args, 'addname') else ''

        model_dir = os.path.join(args.model_dir, model)
        m = os.path.basename(model_dir)
        if 'MynnUNetTrainer' in m:
            name = m.split('MynnUNetTrainer')[1].split('__')[0]#.replace('__','')
            addname = name+addname
            if addname[0] != '_':
                addname = '_'+addname

        subdir_out = '{}_{}{}'.format(args.image_dir, model.split(os.sep)[0], addname)
        print(subdir_out)
        for job in range(n_jobs):
            gpu_id = next(gpu_cycle)
            save_probabilities, fold = False, None
            ID_selection = ID_chunks[job]
            dir_in = os.path.join(args.p_out, args.image_dir)
            dir_out = os.path.join(args.p_out, subdir_out)
            if isinstance(ast.literal_eval(args.fold),int):
                dir_out = os.path.join(dir_out, 'fold_{}'.format(ast.literal_eval(args.fold)))
                save_probabilities = True
                fold = ast.literal_eval(args.fold)

            print(dir_in, dir_out)
            os.makedirs(dir_out, exist_ok=True)
            files_in, files_out = nnunet_input_output_files_list(ID_selection,
                                                                 channels,
                                                                 dir_in,
                                                                 dir_out,
                                                                 overwrite=args.overwrite,
                                                                 ID_splitter=args.ID_splitter if hasattr(args, 'ID_splitter') else '_'
                                                                 )
            print('Files in and out examples',files_in[:3], files_out[:3])

            if len(files_in) == 0:
                print(f"No files to segment for {model} in job {job}. Skipping.")
                continue
            else:
                inp = (args.yml_args, files_in, files_out, model_dir, gpu_id, save_probabilities, fold, args.use_monai_inferers)
                job_inputs.append(inp)

    if len(job_inputs) > 0:
        #print('Starting multiprocessing with {} jobs'.format(len(job_inputs)))
        main_segmentation_processor(job_inputs,  n_procs=1)

