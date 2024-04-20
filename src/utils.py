import wandb
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np


def plot_hypergradient(val_losses, lmbds):
    iterations = np.arange(1, len(val_losses) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(iterations, val_losses, marker="o")
    plt.xlabel("Iteration")
    plt.ylabel("Validation Loss (MSE)")
    plt.title("Hypergradient Descent - Validation Loss vs Iteration")
    plt.grid(True)

    plt.savefig("../results/hypergradient_valloss.png")

    iterations = np.arange(1, len(lmbds) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, lmbds, marker="o")
    plt.xlabel("Iteration")
    plt.ylabel("Lambda")
    plt.title("Hypergradient Descent - Lambda vs Iteration")
    plt.grid(True)

    
    plt.savefig("../results/hypergradient_lmbd.png")


def plot_grid_search(model, X_train, y_train, X_val, y_val, lmbds):
    """
    Function to plot the grid search results.
    """
    grid = [float(lmbd) for lmbd in lmbds.split(",")]
    wandb.init(project="grid_search", entity="mmkipsit")
    plt.figure(figsize=(10, 6))
    losses = []

    for lmbd in grid:
        model.set_params(alpha=lmbd)
        model.fit(X_train, y_train)
        train_score = model.score(X_train, y_train)
        val_score = model.score(X_val, y_val)
        val_loss = mean_squared_error(y_val, model.predict(X_val))
        losses.append(val_loss)
        wandb.log({"train_score": train_score, "val_score": val_score, "lmbd": lmbd, "val_loss": val_loss})
    wandb.finish()

    plt.plot(grid, losses, marker="o")
    plt.xlabel("Lambda")
    plt.ylabel("Validation Loss (MSE)")
    plt.title("Grid Search - Validation Loss vs Lambda for Ridge Regression")
    plt.xscale("log")
    plt.grid(True)

    
    plt.savefig("../results/grid_search.png")


