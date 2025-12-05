from typing import Union, Tuple, List
from torch import nn
import torch
import os, sys

#add Mednext import
mednext_path = os.path.join(os.path.dirname(os.getcwd()), 'MedNeXt')
if not os.path.exists(mednext_path):
    mednext_path = os.getenv('MEDNEXT_PATH')

sys.path.append(mednext_path)
from nnunet_mednext import create_mednext_v1

from huggingface_hub import hf_hub_download
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
#from nnunetv2.training.nnUNetTrainer.variants.network_architecture.MynnUNetTrainer import MynnUNetTrainer
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.models import clip_model_3d
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.models import unet_decoder
from batchgenerators.utilities.file_and_folder_operations import join
from nnunetv2.paths import nnUNet_preprocessed, nnUNet_results
from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader

from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
from nnunetv2.training.data_augmentation.custom_transforms.limited_length_multithreaded_augmenter import LimitedLenWrapper
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class

from nnunetv2.training.dataloading.data_loader_3d_random_raters import nnUNetDataLoader3D_channel_sampler

#not working right now
class MynnUNetTrainerMedNeXt(nnUNetTrainer):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        unpack_dataset: bool = True,
        device: torch.device = torch.device("cuda"),
        model_addname: str = None,
    ):
        super().__init__(plans, configuration, fold, dataset_json,  device)

        self.my_init_kwargs['unpack_dataset'] = unpack_dataset
        #new self arguments for training
        self.patch_size = None #--> parse an args where default patch size can be defined
        self.save_checkpoint_list = [] #--> store these checkpoints for analyses
        self.weight_ctline_dice_loss = 0.0 #-->
        self.adjusted_sampling = False
        self.freeze_encoder = False

        if model_addname is None:
            self.output_folder_base = join(nnUNet_results, self.plans_manager.dataset_name,
                                           self.__class__.__name__ + '__' + self.plans_manager.plans_name + "__" + configuration) \
                if nnUNet_results is not None else None
        else:
            self.output_folder_base = join(nnUNet_results, self.plans_manager.dataset_name,
                                           self.__class__.__name__ + model_addname + '__' + self.plans_manager.plans_name + "__" + configuration) \
                if nnUNet_results is not None else None

        self.output_folder = join(self.output_folder_base, f'fold_{fold}')
    
    @staticmethod
    def build_network_architecture(
                                   architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:

        model = create_mednext_v1(
                      num_input_channels = num_input_channels,
                      num_classes = num_output_channels,
                      model_id = arch_init_kwargs['model_id'], # S, B, M and L are valid model ids
                      kernel_size = arch_init_kwargs['kernel_size'],  # 3x3x3 and 5x5x5 were tested in publication
                      deep_supervision = True     # was used in publication
                    )
        return model

    def set_deep_supervision_enabled(self, enabled: bool):
        """
        This function is specific for the default architecture in nnU-Net. If you change the architecture, there are
        chances you need to change this as well!
        """
        pass

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

        #add learning rate
        if hasattr(args, 'init_lr'):
            if args.init_lr is not None:
                self.initial_lr = args.init_lr

        #for pretrained option to freeze encoder weights (Merlin)
        self.freeze_encoder = args.freeze_encoder if hasattr(args, 'freeze_encoder') else False

        #only for each n_grad_accum the optimizer is optimized (virtually increasing batch size without VRAM overload)
        self.n_grad_accum = args.n_grad_accum if hasattr(args, 'n_grad_accum') else 1

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

        #print('Training keys:', dataset_tr.keys())
        #print('Validation keys:', dataset_val.keys())

        dim = len(self.configuration_manager.patch_size)

        if not self.adjusted_sampling:
            #original for this dataset
            dl_tr = nnUNetDataLoader(dataset_tr, self.batch_size,
                                     initial_patch_size,
                                     self.configuration_manager.patch_size,
                                     self.label_manager,
                                     oversample_foreground_percent=self.oversample_foreground_percent,
                                     sampling_probabilities=None, pad_sides=None, transforms=tr_transforms,
                                     probabilistic_oversampling=self.probabilistic_oversampling)
            dl_val = nnUNetDataLoader(dataset_val, self.batch_size,
                                      self.configuration_manager.patch_size,
                                      self.configuration_manager.patch_size,
                                      self.label_manager,
                                      oversample_foreground_percent=self.oversample_foreground_percent,
                                      sampling_probabilities=None, pad_sides=None, transforms=val_transforms,
                                      probabilistic_oversampling=self.probabilistic_oversampling)

        else:
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

        #original stuff in myunettrainer
        # if allowed_num_processes == 0:
        #     mt_gen_train = SingleThreadedAugmenter(dl_tr, tr_transforms)
        #     mt_gen_val = SingleThreadedAugmenter(dl_val, val_transforms)
        # else:
        #     mt_gen_train = LimitedLenWrapper(self.num_iterations_per_epoch, data_loader=dl_tr, transform=tr_transforms,
        #                                      num_processes=allowed_num_processes, num_cached=6, seeds=None,
        #                                      pin_memory=self.device.type == 'cuda', wait_time=0.02)
        #     mt_gen_val = LimitedLenWrapper(self.num_val_iterations_per_epoch, data_loader=dl_val,
        #                                    transform=val_transforms, num_processes=max(1, allowed_num_processes // 2),
        #                                    num_cached=3, seeds=None, pin_memory=self.device.type == 'cuda',
        #                                    wait_time=0.02)


        # # # let's get this party started
        _ = next(mt_gen_train)
        _ = next(mt_gen_val)

        return mt_gen_train, mt_gen_val
