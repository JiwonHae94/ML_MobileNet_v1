import torch
from torch.utils.data import DataLoader
from torch import nn
from torch.optim import optimizer
from torch.nn.modules import loss

from datetime import datetime


def train(train_loader: DataLoader,
          model: nn.Module,
          criterion : loss._WeightedLoss,
          optimizer : optimizer.Optimizer,
          device: torch.device):
    """
    Trains model for given batch
    """
    model.train()
    running_loss = 0

    for batch, (X, y) in enumerate(train_loader):
        optimizer.zero_grad()

        X = X.to(device)
        y = y.to(device)

        # Forward pass
        y_hat, _ = model(X)
        loss = criterion(y_hat, y)
        running_loss += loss.item() * X.size(0)

        # Backward pass
        loss.backward()
        optimizer.step()

        print(f"training batch num : {batch}\n running_loss = {loss.item() * X.size(0)}")

    epoch_loss = running_loss / len(train_loader.dataset)
    return model, optimizer, epoch_loss

def validate(valid_loader : DataLoader,
             model : nn.Module,
             criterion : loss._WeightedLoss,
             device : torch.device):
    """
    validates model
    """
    model.eval()
    running_loss = 0

    for batch, (X, y) in enumerate(valid_loader):
        X = X.to(device)
        y = y.to(device)

        #Forward pass
        y_hat, _ = model(X)
        loss = criterion(y_hat, y)
        running_loss += loss.item() * X.size(0)

    epoch_loss = running_loss / len(valid_loader.dataset)
    return model, epoch_loss

def training_loop(model: nn.Module,
                  criterion:loss._WeightedLoss,
                  optimizer:optimizer.Optimizer,
                  train_loader : DataLoader,
                  valid_loader : DataLoader,
                  epochs : int,
                  device: torch.device,
                  print_every : int =1):

    train_losses = []
    valid_losses = []

    # Train model
    for epoch in range(0, epochs):

        # training
        model, optimizer, train_loss = train(train_loader, model, criterion, optimizer, device)
        train_losses.append(train_loss)

        if (train_loss == max(train_losses)):
            torch.save(model.state_dict(), f"lenet_{train_loss}.pth")

        # validation
        with torch.no_grad():
            model, valid_loss = validate(valid_loader, model, criterion, device)
            valid_losses.append(valid_loss)

        if epoch % print_every == (print_every - 1):
            train_acc = get_accuracy(model, train_loader, device=device)
            valid_acc = get_accuracy(model, valid_loader, device=device)

            print(f'{datetime.now().time().replace(microsecond=0)} --- '
                  f'Epoch: {epoch}\t'
                  f'Train loss: {train_loss:.4f}\t'
                  f'Valid loss: {valid_loss:.4f}\t'
                  f'Train accuracy: {100 * train_acc:.2f}\t'
                  f'Valid accuracy: {100 * valid_acc:.2f}')

    return model, optimizer, (train_losses, valid_losses)

def get_accuracy(model : nn.Module,
                 data_loader : DataLoader,
                 device : str):
    '''
    Function for computing the accuracy of the predictions over the entire data_loader
    '''
    correct_pred = 0
    n = 0

    with torch.no_grad():
        model.eval()
        for X, y_true in data_loader:
            X = X.to(device)
            y_true = y_true.to(device)

            _, y_prob = model(X)
            _, predicted_labels = torch.max(y_prob, 1)

            n += y_true.size(0)
            correct_pred += (predicted_labels == y_true).sum()

    return correct_pred.float() / n