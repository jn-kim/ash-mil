import math
from typing import List, Tuple, Union

import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn


class RadDinoTokenBackbone(nn.Module):
    def __init__(self, *, freeze: bool = True, input_size: int = 512):
        super().__init__()
        from rad_dino import RadDino

        self.enc = RadDino()
        self.input_size = int(input_size)
        if freeze:
            for p in self.enc.parameters():
                p.requires_grad_(False)

    @property
    def device(self) -> torch.device:
        return next(self.enc.parameters()).device

    def preprocess(self, image_or_images: Union[Image.Image, List[Image.Image]]):
        return self.enc.preprocess(image_or_images)

    def forward(
        self,
        images_or_pixel_values: Union[torch.Tensor, Image.Image, List[Image.Image]],
    ) -> Tuple[torch.Tensor, Tuple[int, int], torch.Tensor]:
        if isinstance(images_or_pixel_values, torch.Tensor):
            pixel_values = images_or_pixel_values
            if pixel_values.device != self.device:
                pixel_values = pixel_values.to(self.device, non_blocking=True)
        else:
            inputs = self.preprocess(images_or_pixel_values).to(self.device)
            pixel_values = inputs["pixel_values"]

        if self.input_size and self.input_size > 0:
            h, w = int(pixel_values.shape[-2]), int(pixel_values.shape[-1])
            if (h, w) != (self.input_size, self.input_size):
                pixel_values = F.interpolate(
                    pixel_values,
                    size=(self.input_size, self.input_size),
                    mode="bilinear",
                    align_corners=False,
                )

        outputs = self.enc.model(pixel_values=pixel_values)
        last_hidden = outputs.last_hidden_state
        cls_token = last_hidden[:, 0]
        patch_tokens = last_hidden[:, 1:]

        t = patch_tokens.shape[1]
        s = int(round(math.sqrt(t)))
        if s * s != t:
            raise ValueError(f"RAD-DINO returned T={t} patch tokens, which is not a perfect square.")
        token_hw = (s, s)

        return patch_tokens, token_hw, cls_token
