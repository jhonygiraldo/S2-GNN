import torch
import torch.nn.functional as F
from torch.optim import Optimizer
from torch_geometric.data import Data


def train(model: torch.nn.Module, data: Data, optimizer: Optimizer) -> float:
    """
    Train the model for one epoch.

    args:
    - model (torch.nn.Module): The model to be trained.
    - data (torch_geometric.data.Data): The data object containing features, labels, and masks.
    - optimizer (torch.optim.Optimizer): The optimizer used for training.

    returns:
    - float: The training loss.
    """
    model.train()
    optimizer.zero_grad()
    logits = model(data)
    loss = F.nll_loss(logits[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def test(model: torch.nn.Module, data: Data) -> dict:
    """
    Evaluate the model on the training, validation, and test sets.

    args:
    - model (torch.nn.Module): The model to be evaluated.
    - data (torch_geometric.data.Data): The data object containing features, labels, and masks.

    returns:
    - dict: A dictionary containing accuracy and loss for the training, validation, and test sets.
    """
    model.eval()
    logits = model(data)
    accs_losses = {}

    for key in ['train', 'val', 'test']:
        mask = data[f'{key}_mask']
        predictions = logits[mask].max(1)[1]
        acc = predictions.eq(data.y[mask]).sum().item() / mask.sum().item()
        loss = F.nll_loss(logits[mask], data.y[mask])
        accs_losses[key] = {'acc': acc, 'loss': loss.item()}

    return accs_losses