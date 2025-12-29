
import torch.nn as nn
from torch.nn import functional as F

class SamMed3D_ImageOnly(nn.Module):
    def __init__(self, sam_model, output_classes: int = 1):
        super().__init__()
        self.sam = sam_model
        self.output_classes = output_classes

        # SAM-Med3D has a fixed number of mask tokens
        self.max_classes = self.sam.mask_decoder.mask_tokens.num_embeddings

        if output_classes < 1:
            raise ValueError("output_classes must be >= 1")

        if output_classes > self.max_classes:
            raise ValueError(
                f"SAM-Med3D supports at most {self.max_classes} output classes "
                f"(got {output_classes})."
            )

    def forward(self, image):
        """
        image: (B, C, D, H, W)
        returns: (B, output_classes, D, H, W)
        """

        # 1️⃣ Encode image
        image_embedding = self.sam.image_encoder(image)

        # 2️⃣ Prompt encoder (NO prompts → prompt-free)
        sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
            points=None,
            boxes=None,
            masks=None,
        )

        # 3️⃣ Decode (multi-mask mode)
        low_res_masks, _ = self.sam.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=self.sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=True,
        )
        # low_res_masks: (B, max_classes, D', H', W')

        # 4️⃣ Select only requested number of classes
        low_res_masks = low_res_masks[:, : self.output_classes]

        # 5️⃣ Upsample to input resolution
        full_res_masks = F.interpolate(
            low_res_masks,
            size=image.shape[-3:],
            mode="trilinear",
            align_corners=False,
        )

        return full_res_masks