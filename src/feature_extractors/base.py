"""
File: FeatureExtractor.py
Author: Sebastian Hönel
"""

from pathlib import Path
from typing import Iterable
from numpy import ndarray, save, transpose
from pandas import DataFrame
from typing import Self
from torch import Tensor, nn
from re import compile, IGNORECASE



class FeatureExtractor(nn.Module):
    """
    This is the mother of all (file-based) feature extractors. It is meant to provide
    common functionality, as well as a common interface for its sub-classes.
    """
    def __init__(self, in_folder: Path, out_folder: Path|None=None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        assert isinstance(in_folder, Path) and (isinstance(out_folder, Path) or out_folder is None)
        self.in_folder = in_folder
        self.out_folder = out_folder
        self.images: list[Path] = []


    def extract(self) -> Tensor:
        raise NotImplementedError('Abstract Method.')


    def save(self, outfile_stemname: str, data: ndarray|Tensor, save_numpy: bool=True, save_csv: bool=True) -> Self:
        if not (save_csv or save_numpy):
            raise Exception('You should save something.')
        
        # Make sure the out-folder exists, recursively
        self.out_folder.mkdir(parents=True, exist_ok=True)
        if isinstance(data, Tensor):
            data = data.cpu().numpy()
        
        if save_numpy:
            outfile_numpy = self.out_folder.joinpath(f'{outfile_stemname}.npy')
            save(file=outfile_numpy, arr=data)
        
        if save_csv:
            outfile_pandas = self.out_folder.joinpath(f'{outfile_stemname}.csv')
            df = DataFrame(transpose(data))
            df.columns = list([f'i_{file.stem}' for file in self.images])
            df.to_csv(path_or_buf=outfile_pandas, index=False)


    def load_images(self, exts: Iterable[str]=['bmp', 'gif', 'jpe?g', 'png', 'tiff?', 'webp'], verbose: bool = True) -> list[Path]:
        self.images.clear()

        regex = compile(pattern=f'{"|".join(map(lambda ext: f"({ext})", exts))}$', flags=IGNORECASE)

        files = list(self.in_folder.glob('*'))
        files.sort()
        for file in files:
            if bool(regex.search(file.name)):
                if verbose:
                    print(f'Adding image: {file.name}')
                self.images.append(file)
        return self.images
