from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import torch

def get_data(train_size=0.8, seed=42):
    (X, y) = load_diabetes(return_X_y=True)
    X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=train_size, random_state=seed)
    return X_train, X_val, y_train, y_val


def get_data_tensors(train_size=0.8, seed=42):
    data = load_diabetes()
    X = data.data
    y = data.target

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)


    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    X_train, X_val, y_train, y_val = train_test_split(X_tensor, y_tensor, train_size=train_size, random_state=seed)

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)

    return train_loader, val_loader
