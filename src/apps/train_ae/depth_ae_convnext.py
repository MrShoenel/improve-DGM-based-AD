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






def train_loop(loader: DataLoader, fe: nn.Module, model: nn.Module, loss_fn: nn.Module, optimizer: Optimizer) -> float:
    size = len(loader.dataset)
    fe.eval()
    model.train()

    loss_total = 0.0
    for idx_batch, sample_batched in enumerate(loader):
        # We're training a reconstructing AE, so X==Y
        X = sample_batched.to(dev)
        X_feats = fe(X)
        
        pred = model(X_feats)
        loss = loss_fn(pred, X_feats)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss, current = loss.item(), idx_batch * batch_size + len(X)
        print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        loss_total += loss
    print(f'\nTotal loss:\t{loss_total:>7f}')
    return loss_total


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




if __name__ == '__main__':
    manual_seed(0xbeef)
    batch_size = 32


    train_dataset = GlobImageDataset(folder=ROOT_DIR.joinpath('/home/sehoaa/GLASS-repl_a/glink/ce_blanked/train/good'), seed=1, shuffle=True)
    train_dataset.files = train_dataset.files#[0:80*batch_size]
    train_loader = DataLoader(dataset=train_dataset, shuffle=False, batch_size=batch_size)


    dev = 'cuda' if cuda.is_available() else 'cpu'
    model = nn.DataParallel(ConvNextV2_DepthAE(dev=dev))
    # model.load_state_dict(torch.load(f'./{ConvNextV2_DepthAE.__name__}.torch'))
    fe = nn.DataParallel(ConvNextV2(Path(__file__))).eval()
    # summary(model=model, batch_size=8, input_size=(2816, 16, 16))
    
    learning_rate = 5e-5
    # loss_fn = nn.MSELoss() # RMLSELoss()
    loss_fn = HuberLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, amsgrad=False)
    early_stopper = EarlyStopper(patience=30, min_delta=5e-3)
    print(f'Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')

    epochs = 20000
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        loss = train_loop(loader=train_loader, fe=fe, model=model, loss_fn=loss_fn, optimizer=optimizer)
        if early_stopper.early_stop(loss):
            torch.save(obj=model.state_dict(), f=f'./{ConvNextV2_DepthAE.__name__}.torch')
            break
