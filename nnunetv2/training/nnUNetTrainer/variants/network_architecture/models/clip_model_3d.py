import torch
from torch import nn
import sys
from transformers import AutoModel, AutoTokenizer, AutoModelForMaskedLM
from torchvision import transforms
import torchvision
import copy
from torch.nn import Parameter

from nnunetv2.training.nnUNetTrainer.variants.network_architecture.models.src import i3res



class Normalize3D(torch.nn.Module):
    def __init__(self, mean, std):
        super(Normalize3D, self).__init__()
        self.mean = mean
        self.std = std

    def forward(self, tensor):
        # Assumes tensor is of shape (C, D, H, W)
        B, C, D, H, W = tensor.shape
        for d in range(D):
            for c in range(C):
                tensor[:, c, d] = (tensor[:, c, d] - self.mean[c]) / self.std[c]
        return tensor


def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )


class ImageEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.use_biomedclip = False
        self.use_ehr = config["use_ehr"]
        self.config = config
        self.use_openclip = False
        self.adapter_bool = config.get("adapter", False)

        if "i3_resnet" in config["architecture"]:
            resnet = torchvision.models.resnet152(pretrained=True)
            model = i3res.I3ResNet(copy.deepcopy(resnet), class_nb=1692, conv_class=True)
            self.i3_resnet = model
                
        # Adding an adapter for few-shot experiments
        if self.adapter_bool:
            reduction = config.get("reduction", 4)
            c_in = config.get("c_in", 512)
            self.adapter = nn.Sequential(
                nn.Linear(c_in, c_in // reduction, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(c_in // reduction, c_in, bias=False),
                nn.ReLU(inplace=True)
            )


    def forward(self, image):

        if "i3_resnet" in self.config["architecture"]:
            
            
            # For nnU-Net
            image = image.permute(0, 1, 3, 4, 2)

            # print("Input Image Shape: ", image.shape)
            contrastive_features, ehr_features, skips = self.i3_resnet(image)
            # # image_features = self.encode_image(image_features)
            # return contrastive_features, ehr_features, skips
            
            if self.adapter_bool:
                contrastive_features = self.adapter(contrastive_features)
            
            # Note: if using nnunet, then 334 needs to be commented out, need use 333
            return contrastive_features, ehr_features, skips
            # return contrastive_features, ehr_features


class TextEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        if "longformer" in config["text_encoder"]:
            self.tokenizer = AutoTokenizer.from_pretrained(
                "yikuan8/Clinical-Longformer"
            )
            self.text_encoder = AutoModel.from_pretrained("yikuan8/Clinical-Longformer")
            self.text_encoder.gradient_checkpointing_enable()
        else:
            raise ValueError("Invalid text encoder.")

        self.linear_layer = nn.Linear(768, 512)

    def forward(self, text_labels):
        text_labels = [text.lower() for text in text_labels]
        
        inputs = self.tokenizer(
                text_labels,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024,
            )
        inputs = {k: v.to(self.text_encoder.device) for k, v in inputs.items()}
        
        if "longformer" in self.config["text_encoder"]:
            text_embeddings = self.text_encoder(**inputs).last_hidden_state[:, 0, :]
            text_embeddings = self.linear_layer(text_embeddings)
        else:
            raise ValueError("Invalid text encoder.")

        return text_embeddings


class Clip3D(nn.Module):
    def __init__(
        self, config, init_logit_scale: float = 1.0, init_logit_bias: float = 0.0
    ):
        super().__init__()
        self.encode_image = ImageEncoder(config)
        self.encode_text = TextEncoder(config)
        self.config = config
        # self.logit_scale = nn.Parameter(torch.ones([]))
        self.logit_scale = nn.Parameter(torch.ones([]) * init_logit_scale)
        # self.logit_bias = nn.Parameter(torch.ones([]) * init_logit_bias)
        self.logit_bias = None  
        self.adapter_bool = config.get("adapter", False)      
        

    def forward(self, image, text):
        image_features, ehr_features = self.encode_image(image)
        text_features = self.encode_text(text)


        # if there is just one dimension, add a batch dimension
        if len(image_features.shape) == 1:
            image_features = image_features.unsqueeze(0)
        if len(text_features.shape) == 1:
            text_features = text_features.unsqueeze(0)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()
        
        if self.adapter_bool:
            return logits_per_image

        if self.config['use_ehr']:
            return (
                image_features,
                ehr_features,
                text_features,
                logits_per_image,
                logits_per_text,
                self.logit_scale.exp(),
                self.logit_bias,
            )
        else:
            # create ehr features as a dummy tensor
            batchsize = image_features.shape[0]
            ehr_features = torch.zeros((batchsize, 1692)).cuda()
            return (
                image_features,
                ehr_features,
                text_features,
                logits_per_image,
                logits_per_text,
                self.logit_scale.exp(),
                self.logit_bias,
            )


