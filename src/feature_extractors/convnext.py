import torch
from PIL import Image
from pathlib import Path
from .base import FeatureExtractor
from torch import Tensor, device, cuda, nn
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
                inputs = self.image_processor(Image.open(fp=file).convert('RGB'), return_tensors='pt').to(self.dev)
                features = self.forward(inputs)
                results.append(features)

                # BaseModelOutputWithPoolingAndNoAttention
                # temp = self.model(**inputs).last_hidden_state #.cpu().numpy()
                # if swap_channels_last:
                #     # Channels First  -->> Channels Last
                #     # 1 x 2816 x 16 x 16  -->>  1 x 16 x 2816 x 16  -->>  1 x 16 x 16 x 2816
                #     temp = np.swapaxes(np.swapaxes(temp, 1, 2), 2, 3)
                # # -->> 1 x 720896
                # temp = temp.flatten()
                # results.append(temp)
        
        return torch.vstack(results)
    
    def forward(self, x: Tensor|BatchFeature) -> Tensor:
        if isinstance(x, Tensor):
            return self.model(x).last_hidden_state
        elif isinstance(x, BatchFeature):
            return self.model(**x).last_hidden_state
        
        raise Exception(f'The type of x ({type(x)}) is not supported.')
    