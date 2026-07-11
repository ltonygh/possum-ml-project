import torch
import torch.nn as nn

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