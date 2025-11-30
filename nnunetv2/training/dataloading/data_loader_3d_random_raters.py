import numpy as np
from nnunetv2.training.dataloading.base_data_loader import nnUNetDataLoaderBase
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDataset
from typing import Union, List, Tuple
from batchgenerators.dataloading.data_loader import DataLoader

class nnUNetDataLoader3D_channel_sampler(nnUNetDataLoaderBase):
    def __init__(self,
                 data: nnUNetDataset,
                 batch_size: int,
                 patch_size: Union[List[int], Tuple[int, ...], np.ndarray],
                 final_patch_size: Union[List[int], Tuple[int, ...], np.ndarray],
                 label_manager, #: LabelManager
                 oversample_foreground_percent: float = 0.0,
                 sampling_probabilities: Union[List[int], Tuple[int, ...], np.ndarray] = None,
                 pad_sides: Union[List[int], Tuple[int, ...], np.ndarray] = None,
                 probabilistic_oversampling: bool = False,
                 multichannel_val_loader=False,
                 img_gt_sampling_strategy=(False,False,False, None, None, None)):
        super().__init__(data=data, batch_size=batch_size, patch_size=patch_size,
                         final_patch_size=final_patch_size,label_manager=label_manager,
                         oversample_foreground_percent=oversample_foreground_percent,
                         sampling_probabilities=sampling_probabilities,
                         pad_sides=pad_sides, probabilistic_oversampling=probabilistic_oversampling)


        self.random_gt_sampling, self.random_img_sampling, self.random_gt_img_sampling, self.ix_seg, self.ix_img, self.possible_channels = img_gt_sampling_strategy
        #self.random_gt_sampling: samples ground truth from channel
        #self.random_img_sampling: samples image from channel
        #self.random_gt_img_sampling: samples both image and ground truth from channel with same index!
        #self.ix_seg and self.ix_img: index of the segmentation and image channel if predifined (overrules random sampling)
        #self.possible_channels: list of possible channels to sample from, if None all channels are used
        self.multichannel_val_loader = multichannel_val_loader #True if validation should be performed on all channels per ID

        self.data_shape, self.seg_shape = self.my_determine_shapes()
    #changed because shape should be different
    def my_determine_shapes(self):

        data, seg, properties = self._data.load_case(self.indices[0])
        if self.random_gt_sampling:
            num_color_channels = data.shape[0] #only for multichannel training
        elif self.random_img_sampling or self.random_gt_img_sampling:
            num_color_channels = 1

        if self.multichannel_val_loader:
            #stack all channels in the batch dim for validation of all channels w gt
            data_shape = (self.batch_size*data.shape[0], num_color_channels, *self.patch_size)
            seg_shape = (self.batch_size*seg.shape[0], 1, *self.patch_size)
        else:
            data_shape = (self.batch_size, num_color_channels, *self.patch_size)
            seg_shape = (self.batch_size, 1, *self.patch_size)

        return data_shape, seg_shape

    def multichannel_validation_batch(self, selected_keys):
        #implement loading of all channels in a batch for validation

        data_all = np.zeros(self.data_shape, dtype=np.float32)
        seg_all = np.zeros(self.seg_shape, dtype=np.int16)
        case_properties = []
        ix_count = 0
        new_selected_keys = []
        for j, i in enumerate(selected_keys):
            # oversampling foreground will improve stability of model training, especially if many patches are empty
            # (Lung for example)
            force_fg = self.get_do_oversample(j)

            data, seg, properties = self._data.load_case(i)


            #define what channels should be get
            d_chans = np.arange(data.shape[0]) if self.possible_channels is None else np.array(self.possible_channels)
            s_chans = np.arange(seg.shape[0]) if self.possible_channels is None else np.array(self.possible_channels)

            if self.random_gt_img_sampling:
                #data and seg channels (multiple image-gt pairs) are separate instances in batch
                #img and gt per channel below to eachother
                if seg.shape[0] != data.shape[0]:
                    raise ValueError("Segmentation and image channels do not match!", seg.shape, data.shape)
                data_lst = [data[i:i+1]for i in d_chans]
                seg_lst = [seg[i:i+1] for i in s_chans]
            elif self.random_gt_sampling:
                #random gt always same input img
                data_lst = [data[0:1]for i in s_chans]
                seg_lst = [seg[i:i+1] for i in s_chans]
            elif self.random_img_sampling:
                #ranodm img always same gt
                data_lst = [data[i:i+1]for i in d_chans]
                seg_lst = [seg[0:1] for i in d_chans]

            if len(data_lst) != len(seg_lst):
                raise ValueError("Number of data channels and segmentation channels do not match!", len(data_lst), len(seg_lst))

            case_properties.append(properties)
            for k in range(len(data_lst)):
                data = data_lst[k]
                seg = seg_lst[k]
                # If we are doing the cascade then the segmentation from the previous stage will already have been loaded by
                # self._data.load_case(i) (see nnUNetDataset.load_case)
                shape = data.shape[1:]
                dim = len(shape)
                bbox_lbs, bbox_ubs = self.get_bbox(shape, force_fg, properties['class_locations'])

                # whoever wrote this knew what he was doing (hint: it was me). We first crop the data to the region of the
                # bbox that actually lies within the data. This will result in a smaller array which is then faster to pad.
                # valid_bbox is just the coord that lied within the data cube. It will be padded to match the patch size
                # later
                valid_bbox_lbs = np.clip(bbox_lbs, a_min=0, a_max=None)
                valid_bbox_ubs = np.minimum(shape, bbox_ubs)

                # At this point you might ask yourself why we would treat seg differently from seg_from_previous_stage.
                # Why not just concatenate them here and forget about the if statements? Well that's because segneeds to
                # be padded with -1 constant whereas seg_from_previous_stage needs to be padded with 0s (we could also
                # remove label -1 in the data augmentation but this way it is less error prone)
                this_slice = tuple([slice(0, data.shape[0])] + [slice(i, j) for i, j in zip(valid_bbox_lbs, valid_bbox_ubs)])
                data = data[this_slice]

                this_slice = tuple([slice(0, seg.shape[0])] + [slice(i, j) for i, j in zip(valid_bbox_lbs, valid_bbox_ubs)])
                seg = seg[this_slice]

                padding = [(-min(0, bbox_lbs[i]), max(bbox_ubs[i] - shape[i], 0)) for i in range(dim)]
                padding = ((0, 0), *padding)

                data_all[ix_count] = np.pad(data, padding, 'constant', constant_values=0)
                seg_all[ix_count] = np.pad(seg, padding, 'constant', constant_values=-1)
                new_selected_keys.append(i)
                ix_count += 1

        return {'data': data_all, 'seg': seg_all, 'properties': case_properties, 'keys': np.array(new_selected_keys)}


    def multichannel_train_sampling(self, selected_keys):
        # preallocate memory for data and seg
        data_all = np.zeros(self.data_shape, dtype=np.float32)
        seg_all = np.zeros(self.seg_shape, dtype=np.int16)
        case_properties = []

        for j, i in enumerate(selected_keys):
            # oversampling foreground will improve stability of model training, especially if many patches are empty
            # (Lung for example)
            force_fg = self.get_do_oversample(j)

            data, seg, properties = self._data.load_case(i)

            #sample index
            if len(data.shape)>3 or len(seg.shape)>3:
                ix_seg = None
                if self.random_gt_img_sampling:
                    if seg.shape[0]!=data.shape[0]:
                        raise ValueError("Segmentation and image channels do not match!", seg.shape, data.shape)
                    #rand opt is indices to consider for sampling (can be restricted to not include all)
                    rand_opts = np.arange(data.shape[0]) if self.possible_channels is None else np.array(
                        self.possible_channels)
                    ix = np.random.choice(rand_opts)
                    ix_seg = ix if self.ix_seg is None else self.ix_seg
                    ix_img = ix if self.ix_img is None else self.ix_img
                    # fetch the right channel
                    seg = seg[ix_seg:ix_seg + 1]
                    data = data[ix_img:ix_img + 1]
                elif self.random_gt_sampling:
                    rand_opts = np.arange(seg.shape[0]) if self.possible_channels is None else np.array(
                        self.possible_channels)
                    ix = np.random.choice(rand_opts)
                    ix_seg = ix if self.ix_seg is None else self.ix_seg
                    # fetch the right channel
                    seg = seg[ix_seg:ix_seg + 1]
                elif self.random_img_sampling:
                    rand_opts = np.arange(data.shape[0]) if self.possible_channels is None else np.array(
                        self.possible_channels)
                    ix = np.random.choice(rand_opts)
                    ix_img = ix if self.ix_img is None else self.ix_img
                    data = data[ix_img:ix_img + 1]

                if ix_seg is not None:
                    #alter the properties --> locations of foreground
                    properties = properties.copy() #don't know if this is required
                    #select the foreground coordinates available for sampling
                    coords = properties['class_locations'][1]
                    select_coords = coords[:,0]==ix_seg
                    properties['class_locations'][1] = coords[select_coords]

            case_properties.append(properties)

            # If we are doing the cascade then the segmentation from the previous stage will already have been loaded by
            # self._data.load_case(i) (see nnUNetDataset.load_case)
            shape = data.shape[1:]
            dim = len(shape)
            bbox_lbs, bbox_ubs = self.get_bbox(shape, force_fg, properties['class_locations'])

            # whoever wrote this knew what he was doing (hint: it was me). We first crop the data to the region of the
            # bbox that actually lies within the data. This will result in a smaller array which is then faster to pad.
            # valid_bbox is just the coord that lied within the data cube. It will be padded to match the patch size
            # later
            valid_bbox_lbs = np.clip(bbox_lbs, a_min=0, a_max=None)
            valid_bbox_ubs = np.minimum(shape, bbox_ubs)

            # At this point you might ask yourself why we would treat seg differently from seg_from_previous_stage.
            # Why not just concatenate them here and forget about the if statements? Well that's because segneeds to
            # be padded with -1 constant whereas seg_from_previous_stage needs to be padded with 0s (we could also
            # remove label -1 in the data augmentation but this way it is less error prone)
            this_slice = tuple([slice(0, data.shape[0])] + [slice(i, j) for i, j in zip(valid_bbox_lbs, valid_bbox_ubs)])
            data = data[this_slice]

            this_slice = tuple([slice(0, seg.shape[0])] + [slice(i, j) for i, j in zip(valid_bbox_lbs, valid_bbox_ubs)])
            seg = seg[this_slice]

            padding = [(-min(0, bbox_lbs[i]), max(bbox_ubs[i] - shape[i], 0)) for i in range(dim)]
            padding = ((0, 0), *padding)
            data_all[j] = np.pad(data, padding, 'constant', constant_values=0)
            seg_all[j] = np.pad(seg, padding, 'constant', constant_values=-1)

        return {'data': data_all, 'seg': seg_all, 'properties': case_properties, 'keys': selected_keys}

    def generate_train_batch(self):
        selected_keys = self.get_indices()

        if self.multichannel_val_loader:
            output = self.multichannel_validation_batch(selected_keys)
        else:
            output = self.multichannel_train_sampling(selected_keys)

        return output




if __name__ == '__main__':
    folder = '/media/fabian/data/nnUNet_preprocessed/Dataset002_Heart/3d_fullres'
    ds = nnUNetDataset(folder, 0)  # this should not load the properties!
    dl = nnUNetDataLoader3D(ds, 5, (16, 16, 16), (16, 16, 16), 0.33, None, None)
    a = next(dl)
