import os
import torch
import torch.nn as nn
from typing import Union, Tuple, List, Mapping, Any
from huggingface_hub import hf_hub_download

from nnunetv2.training.nnUNetTrainer.variants.network_architecture.models import clip_model_3d
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.models import unet_decoder


def download_file(
    repo_id: str,
    filename: str,
    local_dir: str,
):
    os.makedirs(local_dir, exist_ok=True)
    local_file_path = hf_hub_download(
        repo_id=repo_id, filename=filename, local_dir=local_dir
    )
    print(f"{filename} downloaded and saved to {local_file_path}")
    return local_file_path

class MerlinUnet(nn.Module):
    def __init__(self,
                   architecture_class_name: str,
                   arch_init_kwargs: dict,
                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                   num_input_channels: int,
                   num_output_channels: int =3,
                   enable_deep_supervision: bool = False):
        super(MerlinUnet, self).__init__()

        self.deep_supervision = enable_deep_supervision
        self.num_output_channels = num_output_channels

        self.model = self.build_model(
            architecture_class_name,
            arch_init_kwargs,
            arch_init_kwargs_req_import,
            num_input_channels,
            num_output_channels,
            True #allways true because otherwise no pretrained weights loaded, handled in forward
        )


    def build_model(self,
                   architecture_class_name: str,
                   arch_init_kwargs: dict,
                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                   num_input_channels: int,
                   num_output_channels: int =3,
                   enable_deep_supervision: bool = True) -> nn.Module:

        model_config = {
        "architecture": "i3_resnet_clinical_longformer",
        "text_encoder": "clinical_longformer",
        "use_ehr": True
        }
        model = clip_model_3d.Clip3D(model_config)

        # Load in Merlin weights
        file_path = download_file(
            repo_id="stanfordmimi/Merlin",
            filename="i3_resnet_clinical_longformer_best_clip_04-02-2024_23-21-36_epoch_99.pt",
            local_dir=os.path.join(os.path.dirname(__file__), "models"),
        )
        checkpoint = torch.load(file_path)
        model_state_dict = model.state_dict()
        filtered_checkpoint = {k: v for k, v in checkpoint.items() if k in model_state_dict and model_state_dict[k].size() == v.size()}
        missing, unexpected = model.load_state_dict(filtered_checkpoint, strict=False)
        print("Missing keys: ", missing)
        print("Unexpected keys: ", unexpected)

        model = model.encode_image
        #freeze encoder parameters weights
        if hasattr(self, 'freeze_encoder'):
            if self.freeze_encoder:
                for name, param in model.named_parameters():
                    param.requires_grad = False

        decoder = unet_decoder.UNetDecoder(num_classes=num_output_channels, deep_supervision=enable_deep_supervision)
        model = torch.nn.Sequential(model, decoder)

        return model

    def forward(self, x):
        # Pass input through the encoder
        output = self.model(x)

        if not self.deep_supervision:
            output = output[0]
            #make sure for inference to limit to the required number of channels for memory efficiency
            if output.shape[1] != self.num_output_channels:
                output = output[:, :self.num_output_channels, ...]

        return output
    def load_state_dict(
        self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False
        ):

        return self.model.load_state_dict(state_dict, strict=False, assign=assign)
