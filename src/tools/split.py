from typing import Literal, Self
from torch import nn, Tensor
import torch

class Split(nn.Module):
    """
    Makes n copies of the input and gives one to each of the nested sub modules.
    During forward, each module's forward() is called. The results are than stacked
    horizontally or vertically (and, therefore, are  expected to be compatible).
    """
    def __init__(self, /, *args: nn.Module, cat_dim: int|None = None, stack: Literal['h', 'v'] = 'v') -> None:
        super().__init__()

        self.stack = stack
        self.cat_dim = cat_dim
        self.modules: list[nn.Module] = list(args)
        if len(self.modules) == 0:
            raise Exception('Need at least one module.')
        
    def forward(self, x: Tensor) -> Tensor:
        temp = list([m(x) for m in self.modules])
        if self.cat_dim is None:
            func = torch.hstack if self.stack == 'h' else torch.vstack
            temp = func(temp)
        else:
            temp = torch.cat(temp, dim=self.cat_dim)

        return temp
    
    def eval(self) -> Self:
        for m in self.modules:
            m.eval()
        return super().eval()
    
    def train(self, mode: bool = True) -> Self:
        for m in self.modules:
            m.train(mode=mode)
        return super().train(mode)
