from sklearn.linear_model import Ridge
import torch
import torch.nn as nn
import torch.nn.functional as F

class RidgeRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(RidgeRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=True)
        self.lmbd = nn.Parameter(torch.tensor(15.0), requires_grad=True)

    def forward(self, x):
        return self.linear(x)

def get_model(lmbd=0.5):
    return Ridge(alpha=lmbd)
