"""
Here, we implement various EfficientNet feature extractors.
"""

import torch
from pathlib import Path
from .base import FeatureExtractor
from torchvision.models.efficientnet import efficientnet_b5, EfficientNet_B5_Weights, efficientnet_v2_l, EfficientNet_V2_L_Weights
from torchvision.transforms import Compose, ToTensor
from torch import Tensor, device, cuda, nn
from PIL import Image
from abc import ABC



class EfficientNet_base(ABC):
    dev: device
    model: nn.Module
    transforms: Compose

    def extract(self, batch_size: int=16) -> Tensor:
        """
        Returns a Tensor B x C x H x W. For B5, CHW = (2048, 15, 15) and for V2L
        CHW = (1280, 15, 15).
        """
        images = self.load_images()

        results: list[Tensor] = []

        with torch.no_grad():
            for offset in range(0, len(images), batch_size):
                batch = images[offset:(offset + batch_size)]
                temp = list([self.transforms(Image.open(img).convert('RGB')).to(self.dev) for img in batch])

                features = self.forward(torch.stack(temp, dim=0))
                results.append(features)
        
        return torch.vstack(results)
    

    def forward(self, x: Tensor) -> Tensor:
        """
        Extracts the features of a batch of images that are tensors. The input
        `x` to this method, therefore, must be a 4D tensor. Note that the images
        passed here should have gone through the corresponding transforms-pipeline.
        """
        return self.model.features(x)



class EfficientNet_B5(EfficientNet_base, FeatureExtractor):
    def __init__(self, in_folder: Path, out_folder: Path|None = None, dev: device|None = None) -> None:
        super().__init__(in_folder, out_folder)

        self.dev = dev if isinstance(dev, device) else device('cuda' if cuda.is_available() else 'cpu')

        self.model = efficientnet_b5(weights=EfficientNet_B5_Weights.IMAGENET1K_V1, progress=True).to(self.dev).eval()
        self.transforms = Compose([
            ToTensor(),
            EfficientNet_B5_Weights.IMAGENET1K_V1.transforms()
        ])



class EfficientNet_V2_L(EfficientNet_base, FeatureExtractor):
    def __init__(self, in_folder: Path, out_folder: Path | None = None, dev: device | None = None) -> None:
        super().__init__(in_folder, out_folder)

        self.dev = dev if isinstance(dev, device) else device('cuda' if cuda.is_available() else 'cpu')

        self.model = efficientnet_v2_l(weights=EfficientNet_V2_L_Weights.IMAGENET1K_V1, progress=True).to(self.dev)

        self.transforms = Compose([
            ToTensor(),
            EfficientNet_V2_L_Weights.IMAGENET1K_V1.transforms()
        ])
