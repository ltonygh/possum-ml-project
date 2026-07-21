import torch
import torch.nn as nn
import pandas as pd
from typing import Tuple, List



class PossumNetwork(nn.Module):
    """
        A unified neural network architecture. Dynamically constructs hidden layers
        and adapts outputs for either classification or regression tasks.
    """
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout_rate: float):
        super().__init__()
        layers = []
        current_dim = input_dim
        
        for _ in range(num_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            current_dim = hidden_dim
            
        layers.append(nn.Linear(hidden_dim, 1))
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

def data_transform(df: pd.DataFrame, feature_list: List[str], label_list: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
        Transform DataFrames into PyTorch float tensors.
    """
    if not feature_list or not label_list:
        raise ValueError("Feature list and Label list cannot be empty.")
    if not all(col in df.columns for col in feature_list + label_list):
        raise ValueError("Some requested columns are missing from the input DataFrame.")
        
    X_tensor = torch.tensor(df[feature_list].values, dtype=torch.float32)
    y_tensor = torch.tensor(df[label_list].values, dtype=torch.float32)
    return X_tensor, y_tensor

def build_model(input_dim: int, hidden_dim: int, num_layers: int = 2, dropout_rate: float = 0.2) -> PossumNetwork:
    """
        Instantiates an untrained instance of the deep network blueprint.
    """
    return PossumNetwork(input_dim, hidden_dim, num_layers, dropout_rate)


class PossumSexClassifier(nn.Module):
    """
        PyTorch neural network for binary classification on possum sex.
    """
    def __init__(self, input_dim: int = 5, hidden_dim: int = 16):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1) # Outputs 1 logit score
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class PossumAgeRegressor(nn.Module):
    """
        PyTorch neural network for linear regression on possum age.
    """
    def __init__(self, input_dim: int = 5, hidden_dim: int = 16):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class PossumHeadLengthRegressor(nn.Module):
    """
        PyTorch neural network for linear regression on possum head length.
    """
    def __init__(self, input_dim: int = 5, hidden_dim: int = 16):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)