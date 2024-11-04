"""
Here, we implement various EfficientNet feature extractors.
"""

import torch
import numpy as np
from pathlib import Path
from .base import FeatureExtractor
from torchvision.models.efficientnet import efficientnet_b5, EfficientNet_B5_Weights
from torchvision.transforms import Compose, ToTensor
from torch import Tensor, device, cuda
from PIL import Image
from typing import Callable


class EfficientNet_B5(FeatureExtractor):
    def __init__(self, in_folder: Path, out_folder: Path|None = None, dev: device|None = None) -> None:
        super().__init__(in_folder, out_folder)

        self.dev = dev if isinstance(dev, device) else device('cuda' if cuda.is_available() else 'cpu')

        self.model = efficientnet_b5(weights=EfficientNet_B5_Weights.IMAGENET1K_V1, progress=True).to(self.dev)
        self.transforms = Compose([
            ToTensor(),
            EfficientNet_B5_Weights.IMAGENET1K_V1.transforms()
        ])
    

    def extract(self) -> Tensor:
        images = self.load_images()

        results: list[Tensor] = []

        with torch.no_grad():
            for img in images:
                temp = self.transforms(Image.open(img).convert('RGB')).to(self.dev)
                results.append(temp)
        
        return torch.vstack(results)
        