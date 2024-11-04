import numpy as np
from torch import Tensor, is_tensor, int32
from torch.utils.data import Dataset
from pathlib import Path
from typing import Iterable
from re import IGNORECASE, compile
from PIL import Image




class GlobImageDataset(Dataset):
    def __init__(self, folder: str|Path, exts: Iterable[str]=['bmp', 'gif', 'jpe?g', 'png', 'tiff?', 'webp'], seed: int=0, shuffle: bool=True) -> None:
        super().__init__()
        if isinstance(folder, str):
            folder = Path(folder)
        
        assert isinstance(folder, Path) and folder.exists() and folder.is_dir()
        self.folder = folder
        self.exts = exts

        regex = compile(pattern=f'{"|".join(map(lambda ext: f"({ext})", exts))}$', flags=IGNORECASE)
        self.files = list(filter(lambda f: bool(regex.search(f.name)), self.folder.glob('*')))
        self.files.sort()

        if shuffle:
            gen = np.random.default_rng(seed=seed)
            gen.shuffle(self.files)
    

    def __len__(self) -> int:
        return len(self.files)
    

    def __getitem__(self, index: int|Tensor) -> np.ndarray:
        if is_tensor(index):
            assert index.dtype == int32
            index = index.cpu().item()
        
        img = Image.open(self.files[index]).convert('RGB')
        return np.asarray(img)
