###########################################################################
# Paste the following 3 lines at the beginning of each app to make it work.
from sys import path; from pathlib import Path; apps = Path(__file__);
while apps.parent.name != 'apps': apps = apps.parent;
path.append(str(apps.parent)); path.append(str(apps.parent.parent)); del apps;
from apps import APPS_DIR, ROOT_DIR;
###########################################################################

"""
Author: Sebastian Hönel
"""

import torch
from torch import manual_seed
from torch import cuda, nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.nn import HuberLoss
from torchsummary import summary
from src.feature_extractors.convnext import ConvNextV2
from src.auto_encoders.depth_ae import ConvNextV2_DepthAE
from src.tools.earlystop import EarlyStopper
from src.data.dataset import GlobImageDataset






def train_loop(loader: DataLoader, model: nn.Module, loss_fn: nn.Module, optimizer: Optimizer):
    size = len(loader.dataset)
    model.train()

    for idx_batch, sample_batched in enumerate(loader):
        # We're training a reconstructing AE, so X==Y
        X = sample_batched.to(dev)
        
        pred = model(X)
        loss = loss_fn(pred, X)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss, current = loss.item(), idx_batch * batch_size + len(X)
        print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(loader: DataLoader, model: nn.Module, loss_fn: nn.Module):
    model.eval()
    num_batches = len(loader)

    test_loss = 0
    with torch.no_grad():
        for idx_batch, sample_batched in enumerate(loader):
            X, y = sample_batched['input'].to(dev), sample_batched['output'].to(dev)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
    
    test_loss /= num_batches
    print(f"\n--- Test Loss:\n{test_loss}")
    return test_loss



from torch import Tensor, device

class TempModel(nn.Module):
    def __init__(self, dev: device, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model = nn.Sequential(
            ConvNextV2(Path(__file__), dev=dev).eval(),
            ConvNextV2_DepthAE(dev=dev).train()
        )
    

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)



if __name__ == '__main__':
    manual_seed(0xbeef)
    batch_size = 8


    train_dataset = GlobImageDataset(folder=ROOT_DIR.joinpath('./src/test/anomalies/test_noise'), seed=1, shuffle=True)
    train_loader = DataLoader(dataset=train_dataset, shuffle=False, batch_size=batch_size)


    dev = 'cuda' if cuda.is_available() else 'cpu'
    # model = ConvNextV2_DepthAE(dev=dev)
    # fe = ConvNextV2(Path(__file__)).eval()
    # summary(model=model, batch_size=8, input_size=(2816, 16, 16))
    
    learning_rate = 5e-4
    # loss_fn = nn.MSELoss() # RMLSELoss()
    loss_fn = HuberLoss()
    model = TempModel(dev=dev)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, amsgrad=True)
    early_stopper = EarlyStopper(patience=35, min_delta=1e-2)

    epochs = 20000
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(loader=train_loader, model=model, loss_fn=loss_fn, optimizer=optimizer)
        # test_loss = test_loop(loader=test_loader, model=model, loss_fn=loss_fn)
        # if early_stopper.early_stop(test_loss):
        #     torch.save(obj=model.state_dict(), f=MODELS_FOLDER.joinpath(f'./{DepthAE.__name__}_ConvNeXt_V2.torch'))
        #     break
