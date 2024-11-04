import torch
import numpy as np
from PIL import Image
from pathlib import Path
from .base import FeatureExtractor
from torch import Tensor, device, cuda
from transformers import AutoImageProcessor, ConvNextV2Model
from transformers.image_processing_base import BatchFeature



class ConvNextV2(FeatureExtractor):
    def __init__(self, in_folder: Path, out_folder: Path | None = None, dev: device|None = None) -> None:
        super().__init__(in_folder, out_folder)

        self.dev = dev if isinstance(dev, device) else device('cuda' if cuda.is_available() else 'cpu')
        self.model = ConvNextV2Model.from_pretrained('facebook/convnextv2-huge-22k-512').to(self.dev).eval()
        self.image_processor = AutoImageProcessor.from_pretrained('facebook/convnextv2-huge-22k-512')
    

    def extract(self) -> Tensor:
        images = self.load_images()
        results: list[Tensor] = []

        with torch.no_grad():
            for file in images:
                inputs = np.asarray(Image.open(fp=file).convert('RGB'))
                features = self.forward(inputs)
                results.append(features)
        
        return torch.vstack(results)
    
    def forward(self, x: np.ndarray|Tensor|BatchFeature) -> Tensor:
        if isinstance(x, np.ndarray):
            x = Tensor(x).to(device=self.dev)
        
        preproc = self.image_processor(x, return_tensors='pt').to(self.dev)
        return self.model(**preproc).last_hidden_state
    