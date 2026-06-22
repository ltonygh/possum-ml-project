from typing import List, Tuple
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import logging

logging.basicConfig(level=logging.INFO)



def data_transform(df: pd.DataFrame, features_list: List[str], labels_list: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Validate columns' presence, separate and convert them into Pytorch tensors of features and labels respectively"""

    missing_features = set(features_list) - set(df.columns)
    if missing_features:
        raise ValueError(f"Data transform aborted. Missing structural features: {missing_features}")
    
    missing_labels = set(labels_list) - set(df.columns)
    if missing_labels:
        raise ValueError(f"Data transform aborted. Missing structural labels: {missing_labels}")
    
    unused_features = set(df.columns) - (set(features_list) | set(labels_list))
    if unused_features:
        logging.warning(f"Data transform pipeline warning. Unused columns: {unused_features}")
    

    X_value = df[features_list].values
    y_value = df[labels_list].values

    tensor_X = torch.tensor(X_value, dtype=torch.float32)
    tensor_y = torch.tensor(y_value, dtype=torch.float32)

    if tensor_y.dim() == 1:
        tensor_y = tensor_y.unsqueeze(1)

    return tensor_X, tensor_y



def build_model(input_dim: int, hidden_dim: int = 8) -> nn.Module:
    torch.manual_seed(42)

    model = nn.Sequential(
        nn.Linear(in_features = input_dim, out_featuers = hidden_dim),
        nn.ReLU(),
        nn.Linear(in_features = hidden_dim, out_featuers = 1),
    )

    return model    