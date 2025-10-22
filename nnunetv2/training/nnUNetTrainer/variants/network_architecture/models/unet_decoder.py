import numpy as np
import torch
from torch import nn
from typing import Union, List, Tuple

from typing import Type
import numpy as np
import torch.nn
from torch import nn
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.conv import _ConvNd, _ConvTransposeNd
from torch.nn.modules.dropout import _DropoutNd
from torch.nn.modules.instancenorm import _InstanceNorm

from torch.utils.checkpoint import checkpoint

def maybe_convert_scalar_to_list(conv_op, scalar):
    """
    useful for converting, for example, kernel_size=3 to [3, 3, 3] in case of nn.Conv3d
    :param conv_op:
    :param scalar:
    :return:
    """
    if not isinstance(scalar, (tuple, list, np.ndarray)):
        if conv_op == nn.Conv2d:
            return [scalar] * 2
        elif conv_op == nn.Conv3d:
            return [scalar] * 3
        elif conv_op == nn.Conv1d:
            return [scalar] * 1
        else:
            raise RuntimeError("Invalid conv op: %s" % str(conv_op))
    else:
        return scalar
    
def convert_conv_op_to_dim(conv_op: Type[_ConvNd]) -> int:
    """
    :param conv_op: conv class
    :return: dimension: 1, 2 or 3
    """
    if conv_op == nn.Conv1d:
        return 1
    elif conv_op == nn.Conv2d:
        return 2
    elif conv_op == nn.Conv3d:
        return 3
    else:
        raise ValueError("Unknown dimension. Only 1d 2d and 3d conv are supported. got %s" % str(conv_op))
    
def get_matching_convtransp(conv_op: Type[_ConvNd] = None, dimension: int = None) -> Type[_ConvTransposeNd]:
    """
    You MUST set EITHER conv_op OR dimension. Do not set both!

    :param conv_op:
    :param dimension:
    :return:
    """
    assert not ((conv_op is not None) and (dimension is not None)), \
        "You MUST set EITHER conv_op OR dimension. Do not set both!"
    if conv_op is not None:
        dimension = convert_conv_op_to_dim(conv_op)
    assert dimension in [1, 2, 3], 'Dimension must be 1, 2 or 3'
    if dimension == 1:
        return nn.ConvTranspose1d
    elif dimension == 2:
        return nn.ConvTranspose2d
    elif dimension == 3:
        return nn.ConvTranspose3d
    
class ConvDropoutNormReLU(nn.Module):
    def __init__(self,
                 conv_op: Type[_ConvNd],
                 input_channels: int,
                 output_channels: int,
                 kernel_size: Union[int, List[int], Tuple[int, ...]],
                 stride: Union[int, List[int], Tuple[int, ...]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 nonlin_first: bool = False
                 ):
        super(ConvDropoutNormReLU, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        stride = maybe_convert_scalar_to_list(conv_op, stride)
        self.stride = stride

        kernel_size = maybe_convert_scalar_to_list(conv_op, kernel_size)
        if norm_op_kwargs is None:
            norm_op_kwargs = {}
        if nonlin_kwargs is None:
            nonlin_kwargs = {}

        ops = []

        self.conv = conv_op(
            input_channels,
            output_channels,
            kernel_size,
            stride,
            padding=[(i - 1) // 2 for i in kernel_size],
            dilation=1,
            bias=conv_bias,
        )
        ops.append(self.conv)

        if dropout_op is not None:
            self.dropout = dropout_op(**dropout_op_kwargs)
            ops.append(self.dropout)

        if norm_op is not None:
            self.norm = norm_op(output_channels, **norm_op_kwargs)
            ops.append(self.norm)

        if nonlin is not None:
            self.nonlin = nonlin(**nonlin_kwargs)
            ops.append(self.nonlin)

        if nonlin_first and (norm_op is not None and nonlin is not None):
            ops[-1], ops[-2] = ops[-2], ops[-1]

        self.all_modules = nn.Sequential(*ops)

    def forward(self, x):
        return self.all_modules(x)

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == len(self.stride), "just give the image size without color/feature channels or " \
                                                    "batch channel. Do not give input_size=(b, c, x, y(, z)). " \
                                                    "Give input_size=(x, y(, z))!"
        output_size = [i // j for i, j in zip(input_size, self.stride)]  # we always do same padding
        return np.prod([self.output_channels, *output_size], dtype=np.int64)
    
class StackedConvBlocks(nn.Module):
    def __init__(self,
                 num_convs: int,
                 conv_op: Type[_ConvNd],
                 input_channels: int,
                 output_channels: Union[int, List[int], Tuple[int, ...]],
                 kernel_size: Union[int, List[int], Tuple[int, ...]],
                 initial_stride: Union[int, List[int], Tuple[int, ...]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 nonlin_first: bool = False
                 ):
        """

        :param conv_op:
        :param num_convs:
        :param input_channels:
        :param output_channels: can be int or a list/tuple of int. If list/tuple are provided, each entry is for
        one conv. The length of the list/tuple must then naturally be num_convs
        :param kernel_size:
        :param initial_stride:
        :param conv_bias:
        :param norm_op:
        :param norm_op_kwargs:
        :param dropout_op:
        :param dropout_op_kwargs:
        :param nonlin:
        :param nonlin_kwargs:
        """
        super().__init__()
        if not isinstance(output_channels, (tuple, list)):
            output_channels = [output_channels] * num_convs

        self.convs = nn.Sequential(
            ConvDropoutNormReLU(
                conv_op, input_channels, output_channels[0], kernel_size, initial_stride, conv_bias, norm_op,
                norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, nonlin_first
            ),
            *[
                ConvDropoutNormReLU(
                    conv_op, output_channels[i - 1], output_channels[i], kernel_size, 1, conv_bias, norm_op,
                    norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, nonlin_first
                )
                for i in range(1, num_convs)
            ]
        )

        self.output_channels = output_channels[-1]
        self.initial_stride = maybe_convert_scalar_to_list(conv_op, initial_stride)

    def forward(self, x):
        return self.convs(x)

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == len(self.initial_stride), "just give the image size without color/feature channels or " \
                                                            "batch channel. Do not give input_size=(b, c, x, y(, z)). " \
                                                            "Give input_size=(x, y(, z))!"
        output = self.convs[0].compute_conv_feature_map_size(input_size)
        size_after_stride = [i // j for i, j in zip(input_size, self.initial_stride)]
        for b in self.convs[1:]:
            output += b.compute_conv_feature_map_size(size_after_stride)
        return output


class UNetDecoder(nn.Module):
    def __init__(self,
                 num_classes: int,
                 deep_supervision: bool = False, nonlin_first: bool = False):
        """
        This class needs the skips of the encoder as input in its forward.

        the encoder goes all the way to the bottleneck, so that's where the decoder picks up. stages in the decoder
        are sorted by order of computation, so the first stage has the lowest resolution and takes the bottleneck
        features and the lowest skip as inputs
        the decoder has two (three) parts in each stage:
        1) conv transpose to upsample the feature maps of the stage below it (or the bottleneck in case of the first stage)
        2) n_conv_per_stage conv blocks to let the two inputs get to know each other and merge
        3) (optional if deep_supervision=True) a segmentation output Todo: enable upsample logits?
        :param encoder:
        :param num_classes:
        :param n_conv_per_stage:
        :param deep_supervision:
        """
        super().__init__()
        self.deep_supervision = deep_supervision
        # self.encoder = encoder
        self.num_classes = num_classes

        transpconv_op = get_matching_convtransp(conv_op=nn.Conv3d)

        n_stages_encoder = 6
        feature_size = [64, 64, 256, 512, 1024, 2048]
        self.encode_pooled = StackedConvBlocks(2, nn.Conv3d, 3, 64, (3, 3, 3), 1, norm_op=nn.BatchNorm3d, nonlin=nn.ReLU, nonlin_kwargs={'inplace': True},)
        # feature_size = [64, 64, 256, 512, 1024, 2048]

        # we start with the bottleneck and work out way up
        stages = []
        transpconvs = []
        seg_layers = []
        for s in range(1, n_stages_encoder):
            # if s < n_stages_encoder:
            input_features_below = feature_size[-s]
            input_features_skip = feature_size[-(s + 1)]
            stride_for_transpconv = 2
            if s == n_stages_encoder - 1: # ADDED
                stride_for_transpconv = (2, 2, 1) # ADDED
            transpconvs.append(transpconv_op(
                input_features_below, input_features_skip, stride_for_transpconv, stride_for_transpconv,
            ))
            # input features to conv is 2x input_features_skip (concat input_features_skip with transpconv output)
            # stages.append(StackedConvBlocks(
            #     n_conv_per_stage[s-1], encoder.conv_op, 2 * input_features_skip, input_features_skip,
            #     encoder.kernel_sizes[-(s + 1)], 1, encoder.conv_bias, encoder.norm_op, encoder.norm_op_kwargs,
            #     encoder.dropout_op, encoder.dropout_op_kwargs, encoder.nonlin, encoder.nonlin_kwargs, nonlin_first
            # ))
            # if s == n_stages_encoder - 1:
            #     stages_shape = 67
            # else: 
            stages_shape = 2 * input_features_skip
            stages.append(StackedConvBlocks(2, nn.Conv3d, stages_shape, input_features_skip, (3, 3, 3), 1,
                            norm_op=nn.BatchNorm3d, nonlin=nn.ReLU, nonlin_kwargs={'inplace': True},
                            ))
            seg_layers.append(nn.Conv3d(input_features_skip, 23, 1, 1, 0, bias=True))

        self.stages = nn.ModuleList(stages)
        self.transpconvs = nn.ModuleList(transpconvs)
        self.seg_layers = nn.ModuleList(seg_layers)

    def forward(self, skips):
        """
        we expect to get the skips in the order they were computed, so the bottleneck should be the last entry
        :param skips:
        :return:
        """
        skips = skips[2]
        # for skip_sample in skips:
        #     print("Skip sample shape")
        #     print(skip_sample.shape)
        lres_input = skips[-1]
        seg_outputs = []

        for s in range(len(self.stages)):
            # print("Input before checkpoint")
            # print(lres_input.shape)
            x = checkpoint(self.transpconvs[s], lres_input)
            # x = self.transpconvs[s](lres_input)
            # if s < len(self.stages) - 1:
            # print("Output after checkpoint")
            # print(x.shape)
            # print("Skip shape")
            # print(skips[-(s+2)].shape)

            if s == (len(self.stages) - 1):  # ADDED
                skips_input = self.encode_pooled(skips[-(s+2)])  # ADDED
            else:  # ADDED
                skips_input = skips[-(s+2)]  # ADDED

            x = torch.cat((x, skips_input), 1)

            # print("Shape after concat:")
            # print(x.shape)

            x = checkpoint(self.stages[s], x)

            if self.deep_supervision: # ADDED
                seg_outputs.append(self.seg_layers[s](x)) # ADDED
            elif s == (len(self.stages) - 1): # ADDED
                seg_outputs.append(self.seg_layers[-1](x)) # ADDED

            lres_input = x

        # invert seg outputs so that the largest segmentation prediction is returned first
        # print(seg_outputs)
        # seg_outputs = seg_outputs[::-1]
                
        # Transpose the elements in seg_outputs -- FOR nnU-Net
        seg_outputs = [seg_output.permute(0, 1, 4, 2, 3) for seg_output in seg_outputs]

        
        if not self.deep_supervision: # ADDED
            r = seg_outputs[0] # ADDED
        else: # ADDED 
            r = seg_outputs # ADDED
                    
        # print("Seg outputs")
        # Reverse the list r
        if len(r) > 1:
            r = r[::-1]
        # print([r_item.shape for r_item in r])
        
        return r
        # return lres_input