import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
def hyper_gradient_descent(model, train_loader, val_loader, num_epochs=10, lr=0.001):

    wandb.init(project="hyper-gradient-descent", entity="mmkipsit")
    wandb.watch(model, log="all")
    criterion = nn.MSELoss()
    val_losses = []
    lmbds = [model.lmbd.data.item()]
    X_val, y_val = next(iter(val_loader))
    X_train, y_train = next(iter(train_loader))
    optimizer = optim.SGD(model.parameters(), lr=lr)
    hyper_optimizer = optim.SGD([model.lmbd], lr=lr)
    for epoch in range(num_epochs):
        model.train()
        model.zero_grad()
        optimizer.zero_grad()
        hyper_optimizer.zero_grad()
        out = model(X_train)
        train_loss = criterion(out.squeeze(), y_train) + model.lmbd * (torch.norm(model.linear.weight, p=2)**2 + torch.norm(model.linear.bias, p=2)**2)
        train_loss.backward()
        wandb.log({"train_loss": train_loss})
        optimizer.step()

        model.eval()
        hyper_optimizer.zero_grad()
        y_pred_val = model(X_val)
        val_loss = criterion(y_pred_val.squeeze(), y_val)
        val_loss.backward()
        hyper_optimizer.step()
        model.lmbd.data.clamp_(0)
        lmbds.append(model.lmbd.data.item())
        val_losses.append(val_loss.item())
        wandb.log({"train_loss": train_loss, "val_loss": val_loss})
       
    return model, val_losses, lmbds
