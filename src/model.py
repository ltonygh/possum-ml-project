from typing import List, Tuple
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import logging

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



def build_model() -> nn.model:
    model = nn.Sequential(

    )

"""
# Create a tiny mock Pandas DataFrame (similar to your student dataset)
mock_df = pd.DataFrame({
    "Score": [85.0, 75.0, 90.0],
    "RevTime": [12.0, 10.0, 13.0],
    "TargetLength": [89.0, 91.5, 85.0]
})

# Extract features (X) and target (y) values
X_values = mock_df[["Score", "RevTime"]].values
y_values = mock_df["TargetLength"].values

# Convert to standard 32-bit float PyTorch Tensors
X_tensor = torch.tensor(X_values, dtype=torch.float32)
y_tensor_raw = torch.tensor(y_values, dtype=torch.float32)

print(f"Pandas target array shape: {y_values.shape}")
print(f"PyTorch 1D target Tensor shape: {y_tensor_raw.shape}")

# Use .unsqueeze(1) to convert from a flat list into a 2D matrix column
y_tensor_proper = y_tensor_raw.unsqueeze(1)
print(f"PyTorch 2D target Tensor shape after .unsqueeze(1):\n{y_tensor_proper.shape}\n")


print("=== 2. UNDERSTANDING LAYER MATRIX MULTIPLICATION ===")

# Set a fixed seed so your computer generates the exact same initial weights
torch.manual_seed(42)

# Define our input layer. 2 input features (Score, RevTime) -> 4 hidden nodes
linear_layer = nn.Linear(in_features=2, out_features=4)

# Look at the randomly initialized weight matrix under the hood
print(f"Layer weight matrix shape (Out x In): {linear_layer.weight.shape}")

# Feed our 3-row feature matrix into the layer
layer_output = linear_layer(X_tensor)
print(f"Output shape after passing through Linear Layer: {layer_output.shape}")
print(f"Actual matrix values outputted:\n{layer_output}\n")


print("=== 3. UNDERSTANDING NON-LINEAR RELU ACTIVATION ===")

# Instantiate a ReLU activation function
relu_activation = nn.ReLU()

# Push our layer outputs through ReLU
activated_output = relu_activation(layer_output)
print(f"Output values after running through ReLU activation function:")
print(activated_output)
print("(Notice how every single negative number was automatically replaced with 0.0!)\n")


print("=== 4. ASSEMBLING THE COMPLETE NEURAL NETWORK ===")

# Define a full architecture container using nn.Sequential
model = nn.Sequential(
    nn.Linear(in_features=2, out_features=4),  # Layer 1: 2 inputs -> 4 outputs
    nn.ReLU(),                                  # Non-linear threshold flag
    nn.Linear(in_features=4, out_features=1)   # Layer 2: 4 inputs -> 1 final prediction
)

# Run a complete forward pass calculation
predictions = model(X_tensor)
print(f"Final model output shape: {predictions.shape}")
print(f"Predicted target outcomes:\n{predictions}")
"""