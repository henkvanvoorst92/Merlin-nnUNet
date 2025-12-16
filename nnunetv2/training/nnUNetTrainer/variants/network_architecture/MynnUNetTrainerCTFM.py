from typing import Union, Tuple, List
from torch import nn
import torch
import os
from torch import autocast
from huggingface_hub import hf_hub_download
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.helpers import dummy_context

from batchgenerators.utilities.file_and_folder_operations import join
from nnunetv2.paths import nnUNet_preprocessed, nnUNet_results
from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader

from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
from nnunetv2.training.data_augmentation.custom_transforms.limited_length_multithreaded_augmenter import LimitedLenWrapper
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class

from nnunetv2.training.dataloading.data_loader_3d_random_raters import nnUNetDataLoader3D_channel_sampler

class MynnUNetTrainerCTFM(nnUNetTrainer):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        unpack_dataset: bool = True,
        model_addname: str = None,
        device: torch.device = torch.device("cuda")
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
        self.enable_deep_supervision = False
    
    @staticmethod
    def build_network_architecture(
                                   architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:

        from lighter_zoo import SegResNet

        pretrained = SegResNet.from_pretrained(
            "project-lighter/ct_fm_segresnet"
            )

        if pretrained.encoder.conv_init.in_channels!=num_input_channels:
            pretrained.encoder.conv_init.in_channels = num_input_channels
            new_w = torch.stack([torch.clone(pretrained.encoder.conv_init.weight[0]) for i in range(num_output_channels)], dim=0)
            new_b = torch.stack([torch.clone(pretrained.encoder.conv_init.bias[0]) for i in range(num_output_channels)], dim=0)

            pretrained.encoder.conv_init.weight = nn.Parameter(new_w)
            pretrained.encoder.conv_init.bias = nn.Parameter(new_b)

        if pretrained.up_layers[-1].head.out_channels!=num_output_channels:
            pretrained.up_layers[-1].head.out_channels = num_output_channels
            new_w = torch.stack([torch.clone(pretrained.up_layers[-1].head.weight[0]) for i in range(num_output_channels)], dim=0)
            new_b = torch.stack([torch.clone(pretrained.up_layers[-1].head.bias[0]) for i in range(num_output_channels)], dim=0)

            pretrained.up_layers[-1].head.weight = nn.Parameter(new_w)
            pretrained.up_layers[-1].head.bias = nn.Parameter(new_b)

        model = SegResNet(
            in_channels=num_input_channels,  # set this to what your data uses
            out_channels=num_output_channels,  # how many output classes you want
            blocks_down=pretrained.blocks_down,
            init_filters=pretrained.init_filters
            )

        missing, unexpected = model.load_state_dict(pretrained.state_dict(), strict=False)

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

    def train_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        #self.n_grad_accum = self.n_grad_accum.to(self.device)
        #self.n_accumulated_grads = self.n_accumulated_grads.to(self.device)

        if self.n_accumulated_grads==0:
            self.optimizer.zero_grad(set_to_none=True)
            #print('Optim zero grad')

        # Autocast can be annoying
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context(): #
            output = self.network(data)
            del data
            cur_loss = self.loss(output, target)
            if (self.n_grad_accum > 1):
                l = cur_loss / self.n_grad_accum
            else:
                l = cur_loss

            self.n_accumulated_grads+=1

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            if self.n_accumulated_grads % self.n_grad_accum == 0:
                #print('Backprop')
                self.grad_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
                self.n_accumulated_grads = 0
        else:
            l.backward()
            if self.n_accumulated_grads % self.n_grad_accum == 0:
                #print('Backprop')
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.optimizer.step()
                self.n_accumulated_grads = 0

        return {'loss': l.detach().cpu().numpy()}
