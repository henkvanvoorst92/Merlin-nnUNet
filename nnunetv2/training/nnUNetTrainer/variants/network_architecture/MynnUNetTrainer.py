import inspect
import multiprocessing
import os
import shutil
import sys
import warnings
from copy import deepcopy
from datetime import datetime
from time import time, sleep
from typing import Union, Tuple, List

import numpy as np
import torch
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
from batchgenerators.transforms.abstract_transforms import AbstractTransform, Compose
from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform, \
    ContrastAugmentationTransform, GammaTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform
from batchgenerators.transforms.resample_transforms import SimulateLowResolutionTransform
from batchgenerators.transforms.spatial_transforms import SpatialTransform, MirrorTransform
from batchgenerators.transforms.utility_transforms import RemoveLabelTransform, RenameTransform, NumpyToTensor
from batchgenerators.utilities.file_and_folder_operations import join, load_json, isfile, save_json, maybe_mkdir_p
from torch._dynamo import OptimizedModule

from nnunetv2.configuration import ANISO_THRESHOLD, default_num_processes
from nnunetv2.evaluation.evaluate_predictions import compute_metrics_on_folder
from nnunetv2.inference.export_prediction import export_prediction_from_logits, resample_and_save
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.inference.sliding_window_prediction import compute_gaussian
from nnunetv2.paths import nnUNet_preprocessed, nnUNet_results
from nnunetv2.training.data_augmentation.compute_initial_patch_size import get_patch_size
from nnunetv2.training.data_augmentation.custom_transforms.cascade_transforms import MoveSegAsOneHotToData, \
    ApplyRandomBinaryOperatorTransform, RemoveRandomConnectedComponentFromOneHotEncodingTransform
from nnunetv2.training.data_augmentation.custom_transforms.deep_supervision_donwsampling import \
    DownsampleSegForDSTransform2
#from nnunetv2.training.data_augmentation.custom_transforms.limited_length_multithreaded_augmenter import LimitedLenWrapper
from nnunetv2.training.data_augmentation.custom_transforms.masking import MaskTransform
from nnunetv2.training.data_augmentation.custom_transforms.region_based_training import \
    ConvertSegmentationToRegionsTransform
from nnunetv2.training.data_augmentation.custom_transforms.transforms_for_dummy_2d import Convert2DTo3DTransform, \
    Convert3DTo2DTransform
from nnunetv2.training.dataloading.data_loader_2d import nnUNetDataLoader2D
from nnunetv2.training.dataloading.data_loader_3d import nnUNetDataLoader3D
from nnunetv2.training.dataloading.data_loader_3d_random_raters import nnUNetDataLoader3D_channel_sampler

from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDataset
from nnunetv2.training.dataloading.utils import get_case_identifiers, unpack_dataset
from nnunetv2.training.logging.nnunet_logger import nnUNetLogger
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss, DC_and_BCE_loss #, DC_CE_clDC_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn, MemoryEfficientSoftDiceLoss
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from nnunetv2.utilities.collate_outputs import collate_outputs
from nnunetv2.utilities.crossval_split import generate_crossval_split
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
from nnunetv2.utilities.file_path_utilities import check_workers_alive_and_busy
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.utilities.helpers import empty_cache, dummy_context
from nnunetv2.utilities.label_handling.label_handling import convert_labelmap_to_one_hot, determine_num_input_channels
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from torch import autocast, nn
from torch import distributed as dist
from torch.cuda import device_count
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

class MynnUNetTrainer(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int,
                 dataset_json: dict,
                 unpack_dataset: bool = True,
                 model_addname=None,
                 device: torch.device = torch.device('cuda')

                 ):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        #new self arguments for training
        self.patch_size = None #--> parse an args where default patch size can be defined
        self.save_checkpoint_list = [] #--> store these checkpoints for analyses
        self.weight_ctline_dice_loss = 0.0 #-->
        self.adjusted_sampling = False
        if model_addname is None:
            self.output_folder_base = join(nnUNet_results, self.plans_manager.dataset_name,
                                           self.__class__.__name__ + '__' + self.plans_manager.plans_name + "__" + configuration) \
                if nnUNet_results is not None else None
        else:
            self.output_folder_base = join(nnUNet_results, self.plans_manager.dataset_name,
                                           self.__class__.__name__ + model_addname + '__' + self.plans_manager.plans_name + "__" + configuration) \
                if nnUNet_results is not None else None

        self.output_folder = join(self.output_folder_base, f'fold_{fold}')

    ###THESE are added arguments for this specific class
    def add_args(self,args):
        #arguments were added by hvv@stanford.edu

        #new self arguments for training
        if args.save_multiple_checkpoints: #pm change this to parse a list itself
            self.save_checkpoint_list = [5,10,50,100,250,500] #--> store these checkpoints for analyses
        self.weight_ctline_dice_loss = args.w_cldc
        random_gt_sampling = args.random_gt_sampling if hasattr(args, 'random_gt_sampling') else False
        random_img_sampling = args.random_img_sampling if hasattr(args, 'random_img_sampling') else False
        random_gt_img_pair = args.random_gt_img_pair if hasattr(args, 'random_gt_img_pair') else False
        ix_seg = args.ix_seg if hasattr(args, 'ix_seg') else None
        ix_img = args.ix_img if hasattr(args, 'ix_img') else None
        possible_channels = args.possible_channels if hasattr(args, 'possible_channels') else None

        self.img_gt_sampling_strategy =  (random_gt_sampling, random_img_sampling, random_gt_img_pair,
                                          ix_seg, ix_img, possible_channels)
        self.adjusted_sampling = any(self.img_gt_sampling_strategy[:3]) or ix_seg is not None or ix_img is not None
        self.num_epochs = int(args.num_epochs)

    def get_dataloaders(self):
        if self.dataset_class is None:
            self.dataset_class = infer_dataset_class(self.preprocessed_dataset_folder)

        # we use the patch size to determine whether we need 2D or 3D dataloaders. We also use it to determine whether
        # we need to use dummy 2D augmentation (in case of 3D training) and what our initial patch size should be
        patch_size = self.configuration_manager.patch_size

        # needed for deep supervision: how much do we need to downscale the segmentation targets for the different
        # outputs?
        deep_supervision_scales = self._get_deep_supervision_scales()

        (
            rotation_for_DA,
            do_dummy_2d_data_aug,
            initial_patch_size,
            mirror_axes,
        ) = self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()

        # training pipeline
        tr_transforms = self.get_training_transforms(
            patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes, do_dummy_2d_data_aug,
            use_mask_for_norm=self.configuration_manager.use_mask_for_norm,
            is_cascaded=self.is_cascaded, foreground_labels=self.label_manager.foreground_labels,
            regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
            ignore_label=self.label_manager.ignore_label)

        # validation pipeline
        val_transforms = self.get_validation_transforms(deep_supervision_scales,
                                                        is_cascaded=self.is_cascaded,
                                                        foreground_labels=self.label_manager.foreground_labels,
                                                        regions=self.label_manager.foreground_regions if
                                                        self.label_manager.has_regions else None,
                                                        ignore_label=self.label_manager.ignore_label)

        dataset_tr, dataset_val = self.get_tr_and_val_datasets()




        print('Training keys:', dataset_tr.keys())
        print('Validation keys:', dataset_val.keys())

        if dim == 2:
            dl_tr = nnUNetDataLoader2D(dataset_tr, self.batch_size,
                                       initial_patch_size,
                                       self.configuration_manager.patch_size,
                                       self.label_manager,
                                       oversample_foreground_percent=self.oversample_foreground_percent,
                                       sampling_probabilities=None, pad_sides=None, transforms=tr_transforms,
                                       probabilistic_oversampling=self.probabilistic_oversampling)
            dl_val = nnUNetDataLoader2D(dataset_val, self.batch_size,
                                        self.configuration_manager.patch_size,
                                        self.configuration_manager.patch_size,
                                        self.label_manager,
                                        oversample_foreground_percent=self.oversample_foreground_percent,
                                        sampling_probabilities=None, pad_sides=None, transforms=val_transforms,
                                        probabilistic_oversampling=self.probabilistic_oversampling)
        elif dim==3 and self.adjusted_sampling:
            #before jun 6 there was a working version of this
            dl_tr = nnUNetDataLoader3D_channel_sampler(dataset_tr, self.batch_size,
                                                       initial_patch_size,
                                                       self.configuration_manager.patch_size,
                                                       self.label_manager,
                                                       oversample_foreground_percent=self.oversample_foreground_percent,
                                                       sampling_probabilities=None, pad_sides=None,  transforms=tr_transforms,
                                                        img_gt_sampling_strategy=self.img_gt_sampling_strategy,
                                                       probabilistic_oversampling=self.probabilistic_oversampling
                                                       )

            dl_val = nnUNetDataLoader3D_channel_sampler(dataset_val, 1,
                                                        self.configuration_manager.patch_size,
                                                        self.configuration_manager.patch_size,
                                                        self.label_manager,
                                                        oversample_foreground_percent=self.oversample_foreground_percent,
                                                        sampling_probabilities=None, pad_sides=None, transforms=val_transforms,
                                                        img_gt_sampling_strategy=self.img_gt_sampling_strategy,
                                                        multichannel_val_loader=True, #will load all cases per ID for validation
                                                        probabilistic_oversampling = self.probabilistic_oversampling
                                                        )

        elif dim==3:
            dl_tr = nnUNetDataLoader3D(dataset_tr, self.batch_size,
                                       initial_patch_size,
                                       self.configuration_manager.patch_size,
                                       self.label_manager,
                                       oversample_foreground_percent=self.oversample_foreground_percent,
                                       sampling_probabilities=None, pad_sides=None, transforms=tr_transforms,
                                       probabilistic_oversampling=self.probabilistic_oversampling)

            dl_val = nnUNetDataLoader3D(dataset_val, self.batch_size,
                                        self.configuration_manager.patch_size,
                                        self.configuration_manager.patch_size,
                                        self.label_manager,
                                        oversample_foreground_percent=self.oversample_foreground_percent,
                                        sampling_probabilities=None, pad_sides=None, transforms=val_transforms,
                                        probabilistic_oversampling=self.probabilistic_oversampling)

        # origianl datasets for this repo
        #
        # dl_tr = nnUNetDataLoader(dataset_tr, self.batch_size,
        #                          initial_patch_size,
        #                          self.configuration_manager.patch_size,
        #                          self.label_manager,
        #                          oversample_foreground_percent=self.oversample_foreground_percent,
        #                          sampling_probabilities=None, pad_sides=None, transforms=tr_transforms,
        #                          probabilistic_oversampling=self.probabilistic_oversampling)
        # dl_val = nnUNetDataLoader(dataset_val, self.batch_size,
        #                           self.configuration_manager.patch_size,
        #                           self.configuration_manager.patch_size,
        #                           self.label_manager,
        #                           oversample_foreground_percent=self.oversample_foreground_percent,
        #                           sampling_probabilities=None, pad_sides=None, transforms=val_transforms,
        #                           probabilistic_oversampling=self.probabilistic_oversampling)

        allowed_num_processes = get_allowed_n_proc_DA()
        if allowed_num_processes == 0:
            mt_gen_train = SingleThreadedAugmenter(dl_tr, None)
            mt_gen_val = SingleThreadedAugmenter(dl_val, None)
        else:
            mt_gen_train = NonDetMultiThreadedAugmenter(data_loader=dl_tr, transform=None,
                                                        num_processes=allowed_num_processes,
                                                        num_cached=max(6, allowed_num_processes // 2), seeds=None,
                                                        pin_memory=self.device.type == 'cuda', wait_time=0.002)
            mt_gen_val = NonDetMultiThreadedAugmenter(data_loader=dl_val,
                                                      transform=None, num_processes=max(1, allowed_num_processes // 2),
                                                      num_cached=max(3, allowed_num_processes // 4), seeds=None,
                                                      pin_memory=self.device.type == 'cuda',
                                                      wait_time=0.002)
        # # let's get this party started
        _ = next(mt_gen_train)
        _ = next(mt_gen_val)
        return mt_gen_train, mt_gen_val


    def get_plain_dataloaders(self, initial_patch_size: Tuple[int, ...], dim: int):
        dataset_tr, dataset_val = self.get_tr_and_val_datasets()

        print('Training keys:', dataset_tr.keys())
        print('Validation keys:', dataset_val.keys())

        if dim == 2:
            dl_tr = nnUNetDataLoader2D(dataset_tr, self.batch_size,
                                       initial_patch_size,
                                       self.configuration_manager.patch_size,
                                       self.label_manager,
                                       oversample_foreground_percent=self.oversample_foreground_percent,
                                       sampling_probabilities=None, pad_sides=None)
            dl_val = nnUNetDataLoader2D(dataset_val, self.batch_size,
                                        self.configuration_manager.patch_size,
                                        self.configuration_manager.patch_size,
                                        self.label_manager,
                                        oversample_foreground_percent=self.oversample_foreground_percent,
                                        sampling_probabilities=None, pad_sides=None)
        elif dim==3 and self.adjusted_sampling:
            #before jun 6 there was a working version of this
            dl_tr = nnUNetDataLoader3D_channel_sampler(dataset_tr, self.batch_size,
                                                       initial_patch_size,
                                                       self.configuration_manager.patch_size,
                                                       self.label_manager,
                                                       oversample_foreground_percent=self.oversample_foreground_percent,
                                                       sampling_probabilities=None, pad_sides=None,
                                                        img_gt_sampling_strategy=self.img_gt_sampling_strategy
                                                       )

            dl_val = nnUNetDataLoader3D_channel_sampler(dataset_val, 1,
                                                        self.configuration_manager.patch_size,
                                                        self.configuration_manager.patch_size,
                                                        self.label_manager,
                                                        oversample_foreground_percent=self.oversample_foreground_percent,
                                                        sampling_probabilities=None, pad_sides=None,
                                                        img_gt_sampling_strategy=self.img_gt_sampling_strategy,
                                                        multichannel_val_loader=True #will load all cases per ID for validation
                                                        )

        elif dim==3:
            dl_tr = nnUNetDataLoader3D(dataset_tr, self.batch_size,
                                       initial_patch_size,
                                       self.configuration_manager.patch_size,
                                       self.label_manager,
                                       oversample_foreground_percent=self.oversample_foreground_percent,
                                       sampling_probabilities=None, pad_sides=None)

            dl_val = nnUNetDataLoader3D(dataset_val, self.batch_size,
                                        self.configuration_manager.patch_size,
                                        self.configuration_manager.patch_size,
                                        self.label_manager,
                                        oversample_foreground_percent=self.oversample_foreground_percent,
                                        sampling_probabilities=None, pad_sides=None)


        return dl_tr, dl_val
