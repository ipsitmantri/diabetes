"""
Author: Krishna Sri Ipsit Mantri
Date: 19th April 2024
Purpose: This file is the main file for the project. It will be used to train a ridge regression model to predict diabetes progression.

"""

import argparse
from models import get_model, RidgeRegression
from data import get_data, get_data_tensors
from utils import plot_grid_search, plot_hypergradient
from train import hyper_gradient_descent



def main():
    parser = argparse.ArgumentParser(description='Train a ridge regression model to predict diabetes progression.')
    parser.add_argument('--lmbd', type=float, default=1.0, help='Regularization strength')
    parser.add_argument('--train_size', type=float, default=0.8, help='Fraction of data to use for training')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--lmbd_grid', type=str, default='0.0,0.1,1.0,10.0,100.0', help='Grid of regularization strengths')
    parser.add_argument('--hypergradient', type=bool, default=False, help='Whether to use hypergradient optimization')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs for hypergradient optimization')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for hypergradient optimization')
    args = parser.parse_args()
    
    if args.hypergradient:
        train_loader, val_loader = get_data_tensors(args.train_size, args.seed)
        model = RidgeRegression(10, 1)
        print(model)
        model, val_losses, lmbds = hyper_gradient_descent(model, train_loader, val_loader, args.num_epochs, args.lr)
        plot_hypergradient(val_losses, lmbds)
    else:
        # Get the data
        X_train, X_val, y_train, y_val = get_data(args.train_size, args.seed)

        # Get the model
        model = get_model(args.lmbd)

        # Perform the grid search
        plot_grid_search(model, X_train, y_train, X_val, y_val, args.lmbd_grid)


if __name__ == '__main__':
    main()
