from typing import Union, Tuple, List
from torch import nn
import torch
import os, sys
from monai.networks.nets import DynUNet

from huggingface_hub import hf_hub_download
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from torch.optim.lr_scheduler import LinearLR
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
class MynnUNetTrainerVesselFM(nnUNetTrainer):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        unpack_dataset: bool = True,
        model_addname: str = None,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, model_addname, device)

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

        self.weight_decay = 0.1
        self.oversample_foreground_percent = 0.33
        self.probabilistic_oversampling = False
        self.num_iterations_per_epoch = 250
        self.num_val_iterations_per_epoch = 50
        self.num_epochs = 1000
    
    @staticmethod
    def build_network_architecture(
                                   architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:


        model = DynUNet(
            spatial_dims=3,
            in_channels=num_input_channels,
            out_channels=num_output_channels,
            kernel_size=[[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
            strides=[[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
            upsample_kernel_size=[[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
            filters=[32, 64, 128, 256, 320, 320],
            res_block=True,
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
                self.initial_lr = float(args.init_lr)

        #for pretrained option to freeze encoder weights (Merlin)
        self.freeze_encoder = args.freeze_encoder if hasattr(args, 'freeze_encoder') else False

        #only for each n_grad_accum the optimizer is optimized (virtually increasing batch size without VRAM overload)
        self.n_grad_accum = torch.tensor(args.n_grad_accum if hasattr(args, 'n_grad_accum') else 1)

        args.cg

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

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=self.initial_lr,
            weight_decay=self.weight_decay
        )
        lr_scheduler = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=150)
        #PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)
        return optimizer, lr_scheduler
