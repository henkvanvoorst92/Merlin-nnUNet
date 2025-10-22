import math
import sys

import torch
from torch.nn import ReplicationPad3d
import torch.utils.checkpoint as checkpoint
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter


def inflate_conv(conv2d,
                 time_dim=3,
                 time_padding=0,
                 time_stride=1,
                 time_dilation=1,
                 center=False):
    # To preserve activations, padding should be by continuity and not zero
    # or no padding in time dimension
    if conv2d.kernel_size[0] == 7:
        kernel_dim = (3, 7, 7)
        # time_dim = 7
        padding = (1, 3, 3)
        stride = (1, 2, 2)
        # kernel_dim = (7, 7, 7)
        # time_dim = 7
        # padding = (3, 3, 3)
        # stride = (2, 2, 2)
        dilation = (1, 1, 1)
        conv3d = torch.nn.Conv3d(
            conv2d.in_channels,
            conv2d.out_channels,
            kernel_dim,
            padding=padding,
            dilation=dilation,
            stride=stride)
        # Repeat filter time_dim times along time dimension
        weight_2d = conv2d.weight.data
        if center:
            weight_3d = torch.zeros(*weight_2d.shape)
            weight_3d = weight_3d.unsqueeze(2).repeat(1, 1, time_dim, 1, 1)
            middle_idx = time_dim // 2
            weight_3d[:, :, middle_idx, :, :] = weight_2d
        else:
            weight_3d = weight_2d.unsqueeze(2).repeat(1, 1, time_dim, 1, 1)
            weight_3d = weight_3d / time_dim

        # Assign new params
        conv3d.weight = Parameter(weight_3d)
        conv3d.bias = conv2d.bias
    else:
        kernel_dim = (time_dim, conv2d.kernel_size[0], conv2d.kernel_size[1])
        padding = (time_padding, conv2d.padding[0], conv2d.padding[1])
        stride = (time_stride, conv2d.stride[0], conv2d.stride[0])
        dilation = (time_dilation, conv2d.dilation[0], conv2d.dilation[1])
        conv3d = torch.nn.Conv3d(
            conv2d.in_channels,
            conv2d.out_channels,
            kernel_dim,
            padding=padding,
            dilation=dilation,
            stride=stride)
        # Repeat filter time_dim times along time dimension
        weight_2d = conv2d.weight.data
        if center:
            weight_3d = torch.zeros(*weight_2d.shape)
            weight_3d = weight_3d.unsqueeze(2).repeat(1, 1, time_dim, 1, 1)
            middle_idx = time_dim // 2
            weight_3d[:, :, middle_idx, :, :] = weight_2d
        else:
            weight_3d = weight_2d.unsqueeze(2).repeat(1, 1, time_dim, 1, 1)
            weight_3d = weight_3d / time_dim

        # Assign new params
        conv3d.weight = Parameter(weight_3d)
        conv3d.bias = conv2d.bias
    return conv3d


def inflate_linear(linear2d, time_dim):
    """
    Args:
        time_dim: final time dimension of the features
    """
    linear3d = torch.nn.Linear(linear2d.in_features * time_dim,
                               linear2d.out_features)
    weight3d = linear2d.weight.data.repeat(1, time_dim)
    weight3d = weight3d / time_dim

    linear3d.weight = Parameter(weight3d)
    linear3d.bias = linear2d.bias
    return linear3d


def inflate_batch_norm(batch2d):
    # In pytorch 0.2.0 the 2d and 3d versions of batch norm
    # work identically except for the check that verifies the
    # input dimensions

    batch3d = torch.nn.BatchNorm3d(batch2d.num_features)
    # retrieve 3d _check_input_dim function
    batch2d._check_input_dim = batch3d._check_input_dim
    return batch2d


def inflate_pool(pool2d,
                 time_dim=1,
                 time_padding=0,
                 time_stride=None,
                 time_dilation=1):
    if isinstance(pool2d, torch.nn.AdaptiveAvgPool2d):
        pool3d = torch.nn.AdaptiveAvgPool3d((1, 1, 1))
    else:
        kernel_dim = (time_dim, pool2d.kernel_size, pool2d.kernel_size)
        padding = (time_padding, pool2d.padding, pool2d.padding)
        if time_stride is None:
            time_stride = time_dim
        stride = (time_stride, pool2d.stride, pool2d.stride)
        if isinstance(pool2d, torch.nn.MaxPool2d):
            dilation = (time_dilation, pool2d.dilation, pool2d.dilation)
            pool3d = torch.nn.MaxPool3d(
                kernel_dim,
                padding=padding,
                dilation=dilation,
                stride=stride,
                ceil_mode=pool2d.ceil_mode)
        elif isinstance(pool2d, torch.nn.AvgPool2d):
            pool3d = torch.nn.AvgPool3d(kernel_dim, stride=stride)
        else:
            raise ValueError('{} is not among known pooling classes'.format(type(pool2d)))

    return pool3d


abd_windows = dict(
    soft_tissue = (50, 400),
    liver_cect = (80, 150),
    bone = (400, 1800)
)

def apply_window_level(img, windows=abd_windows):
    """
    Apply window leveling to an image with specified windows for each channel.
    
    Parameters:
    img (torch.Tensor): The input image tensor.
    windows (dict): A dictionary containing window settings for each channel.
    
    Returns:
    torch.Tensor: The window-leveled image.
    """
    # Initialize an empty tensor to hold the window-leveled image
    leveled_img = torch.zeros_like(img)
    
    # Iterate over each channel and apply the window level
    for channel, (WL, WW) in enumerate(windows.values()):
        # Calculate the min and max window values
        min_val = WL - (WW / 2)
        max_val = WL + (WW / 2)
        
        # Apply window leveling: Scale the intensity values to be within the window
        # Values below min_val are set to 0, and values above max_val are set to 1
        # Values within the window are scaled linearly between 0 and 1
        leveled_img[:, channel, ...] = torch.clamp((img[:, channel, ...] - min_val) / (max_val - min_val), 0, 1)
    
    return leveled_img

class I3ResNet(torch.nn.Module):
    def __init__(self, resnet2d, frame_nb=16, class_nb=1000, conv_class=False, return_skips=True, vision_ssl=False, classifier_ssl=False, multihead=False, hidden_dim=2048):
        """
        Args:
            conv_class: Whether to use convolutional layer as classifier to
                adapt to various number of frames
        """
        super(I3ResNet, self).__init__()
        self.return_skips = return_skips
        self.conv_class = conv_class

        self.conv1 = inflate_conv(
            resnet2d.conv1, time_dim=3, time_padding=1, center=True)
        self.bn1 = inflate_batch_norm(resnet2d.bn1)
        self.relu = torch.nn.ReLU(inplace=True)
        self.maxpool = inflate_pool(
            resnet2d.maxpool, time_dim=3, time_padding=1, time_stride=2)

        self.layer1 = inflate_reslayer(resnet2d.layer1)
        self.layer2 = inflate_reslayer(resnet2d.layer2)
        self.layer3 = inflate_reslayer(resnet2d.layer3)
        self.layer4 = inflate_reslayer(resnet2d.layer4)
        self.vision_ssl = vision_ssl
        self.classifier_ssl = classifier_ssl
        self.multihead = multihead
        self.hidden_dim = hidden_dim

        if conv_class:
            self.avgpool = inflate_pool(resnet2d.avgpool, time_dim=1)
            if self.multihead:
                self.classifiers = nn.ModuleList()  # Store multiple classifiers
                for i in range(class_nb):  # Assuming num_classifiers is defined
                    self.classifiers.append(torch.nn.Conv3d(
                        in_channels=self.hidden_dim,
                        out_channels=1,
                        kernel_size=(1, 1, 1),
                        bias=True))
            else:
                self.classifier = torch.nn.Conv3d(
                    in_channels=self.hidden_dim,
                    out_channels=class_nb,
                    kernel_size=(1, 1, 1),
                    bias=True)            
                
            
            self.contrastive_head = torch.nn.Conv3d(
                in_channels=self.hidden_dim,
                out_channels=512,
                kernel_size=(1, 1, 1),
                bias=True)
        elif self.classifier_ssl:
            self.avgpool = inflate_pool(resnet2d.avgpool, time_dim=1)
            if self.multihead:
                self.classifiers = nn.ModuleList()
                for i in range(class_nb):
                    self.classifiers.append(torch.nn.Conv3d(
                        in_channels=self.hidden_dim,
                        out_channels=1,
                        kernel_size=(1, 1, 1),
                        bias=True))
            else: 
                self.classifier = torch.nn.Conv3d(
                    in_channels=self.hidden_dim,
                    out_channels=class_nb,
                    kernel_size=(1, 1, 1),
                    bias=True)
        elif vision_ssl:
            self.avgpool = inflate_pool(resnet2d.avgpool, time_dim=1)
            self.rotation_head = torch.nn.Conv3d(
                in_channels=self.hidden_dim,
                out_channels=4,
                kernel_size=(1, 1, 1),
                bias=True)
            self.contrastive_head = torch.nn.Conv3d(
                in_channels=self.hidden_dim,
                out_channels=512,
                kernel_size=(1, 1, 1),
                bias=True)
            
            self.conv = nn.Sequential(
                nn.Conv3d(self.hidden_dim, self.hidden_dim // 2, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm3d(self.hidden_dim // 2),
                nn.LeakyReLU(),
                nn.Upsample(scale_factor=(1, 2, 2), mode="trilinear", align_corners=False),
                nn.Conv3d(self.hidden_dim // 2, self.hidden_dim // 4, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm3d(self.hidden_dim // 4),
                nn.LeakyReLU(),
                nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
                nn.Conv3d(self.hidden_dim // 4, self.hidden_dim // 8, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm3d(self.hidden_dim // 8),
                nn.LeakyReLU(),
                nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
                nn.Conv3d(self.hidden_dim // 8, self.hidden_dim // 16, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm3d(self.hidden_dim // 16),
                nn.LeakyReLU(),
                nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
                nn.Conv3d(self.hidden_dim // 16, self.hidden_dim // 16, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm3d(self.hidden_dim // 16),
                nn.LeakyReLU(),
                nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
                nn.Conv3d(self.hidden_dim // 16, 1, kernel_size=1, stride=1),
            )
        else:
            final_time_dim = int(math.ceil(frame_nb / 16))
            self.avgpool = inflate_pool(
                resnet2d.avgpool, time_dim=final_time_dim)
            self.fc = inflate_linear(resnet2d.fc, 1)

    def forward(self, x):
        skips = []
        # Note: If using nnUNet, then line 125 needs to be commented out.
        x = x.permute(0, 1, 4, 2, 3)
        x = torch.cat((x, x, x), dim=1)
        # x = window_level.apply_window_level(x)
        if self.return_skips:
            skips.append(x.permute(0, 1, 3, 4, 2))
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.return_skips:
            skips.append(x.permute(0, 1, 3, 4, 2))
        x = self.maxpool(x)

        # if x.requires_grad:
        #     x = checkpoint.checkpoint(self.layer1, x)
        #     x = checkpoint.checkpoint(self.layer2, x)
        #     x = checkpoint.checkpoint(self.layer3, x)
        #     x = checkpoint.checkpoint(self.layer4, x)
        # else:
        x = checkpoint.checkpoint(self.layer1, x)
        if self.return_skips:
            skips.append(x.permute(0, 1, 3, 4, 2))
        x = checkpoint.checkpoint(self.layer2, x)
        if self.return_skips:
            skips.append(x.permute(0, 1, 3, 4, 2))
        x = checkpoint.checkpoint(self.layer3, x)
        if self.return_skips:
            skips.append(x.permute(0, 1, 3, 4, 2))
        x = checkpoint.checkpoint(self.layer4, x)
        if self.return_skips:
            skips.append(x.permute(0, 1, 3, 4, 2))

        if self.conv_class:

            x_features = self.avgpool(x)
            
            # return x_features
            
            if self.multihead:
                x_ehr_list = []
                for i in range(len(self.classifiers)):
                    x_ehr = self.classifiers[i](x_features).squeeze(3).squeeze(3).mean(2)
                    x_ehr = F.relu(x_ehr)
                    x_ehr_list.append(x_ehr)
                
                x_ehr = torch.stack(x_ehr_list, dim=1).squeeze()
            else:
                x_ehr = self.classifier(x_features)
                x_ehr = x_ehr.squeeze(3)
                x_ehr = x_ehr.squeeze(3)
                x_ehr = x_ehr.mean(2)
            
            x_contrastive = self.contrastive_head(x_features)
            x_contrastive = x_contrastive.squeeze(3)
            x_contrastive = x_contrastive.squeeze(3)
            x_contrastive = x_contrastive.mean(2)
            
            if self.return_skips:
                return x_contrastive, x_ehr, skips
            else:
                return x_contrastive, x_ehr
        elif self.vision_ssl:
            
            # Getting the output recon
            x_rec = self.conv(x)
            
            # Before do the recon
            x_features = self.avgpool(x)
            
            x_rot = self.rotation_head(x_features).squeeze()
            x_contrastive = self.contrastive_head(x_features).squeeze()
            
            return x_rot, x_contrastive, x_rec
        elif self.classifier_ssl:
            x_features = self.avgpool(x)
            
            if self.multihead:
                x_ehr_list = []
                for i in range(len(self.classifiers)):
                    x_ehr_list.append(self.classifiers[i](x_features).squeeze(3).squeeze(3).mean(2))
                
                x_classifier = torch.stack(x_ehr_list, dim=1).squeeze()
            else:
                x_classifier = self.classifier(x_features).squeeze()
                
            return x_classifier
            
            
            
        else:
            x = self.avgpool(x)
            x_reshape = x.view(x.size(0), -1)
            x = self.fc(x_reshape)
        return x


def inflate_reslayer(reslayer2d):
    reslayers3d = []
    for layer2d in reslayer2d:
        layer3d = Bottleneck3d(layer2d)
        reslayers3d.append(layer3d)
    return torch.nn.Sequential(*reslayers3d)


class Bottleneck3d(torch.nn.Module):
    def __init__(self, bottleneck2d):
        super(Bottleneck3d, self).__init__()

        spatial_stride = bottleneck2d.conv2.stride[0]

        self.conv1 = inflate_conv(
            bottleneck2d.conv1, time_dim=1, center=True)
        self.bn1 = inflate_batch_norm(bottleneck2d.bn1)

        self.conv2 = inflate_conv(
            bottleneck2d.conv2,
            time_dim=3,
            time_padding=1,
            time_stride=spatial_stride,
            center=True)
        self.bn2 = inflate_batch_norm(bottleneck2d.bn2)

        self.conv3 = inflate_conv(
            bottleneck2d.conv3, time_dim=1, center=True)
        self.bn3 = inflate_batch_norm(bottleneck2d.bn3)

        self.relu = torch.nn.ReLU(inplace=True)

        if bottleneck2d.downsample is not None:
            self.downsample = inflate_downsample(
                bottleneck2d.downsample, time_stride=spatial_stride)
        else:
            self.downsample = None

        self.stride = bottleneck2d.stride

    def forward(self, x):
        def run_function(input_x):
            out = self.conv1(input_x)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)
            out = self.relu(out)

            out = self.conv3(out)
            out = self.bn3(out)
            return out

        residual = x

        if self.downsample is not None:
            residual = self.downsample(x)

        if x.requires_grad:
            out = checkpoint.checkpoint(run_function, x)
        else:
            out = run_function(x)

        out = out + residual
        out = self.relu(out)
        return out


def inflate_downsample(downsample2d, time_stride=1):
    downsample3d = torch.nn.Sequential(
        inflate_conv(
            downsample2d[0], time_dim=1, time_stride=time_stride, center=True),
        inflate_batch_norm(downsample2d[1]))
    return downsample3d
