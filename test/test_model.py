from typing import List, Tuple
import pandas as pd
import pytest
import numpy as np
import torch
import torch.nn as nn
from src.model import data_transform, build_model



@pytest.fixture
def dummy_df() -> pd.DataFrame:
    """
    Creates a dummy DataFrame already cleaned suitable for data preprocessing to test the following functions:
    - data_transform
    - build_model

    The dummy DataFrame is built with the following attributes:
    - Population:[50, 60, 40]
    - Stress:    [0.8, 0.2, 0.5]
    - Happiness: [10, 48, 20]
    """

    data = {
        "Population":   [50, 60, 40],
        "Stress":       [0.8, 0.2, 0.6],
        "Happiness":    [10, 48, 20]
    }

    return pd.DataFrame(data)



def test_data_transform(dummy_df):
    """Test if function data_transform works as intended"""

    features_list = ["Population", "Stress"]
    labels_list = ["Happiness"]
    X_tensor, y_tensor = data_transform(dummy_df, features_list, labels_list)

    assert X_tensor.dtype == torch.float32
    assert y_tensor.dtype == torch.float32
    assert X_tensor.shape == (3, 2)
    assert y_tensor.shape == (3, 1)

def test_data_transform_no_feature(dummy_df):
    """Test if function data_transform aborts when no features are introduced"""

    features_list = []
    labels_list = ["Happiness"]

    with pytest.raises(ValueError):
        data_transform(dummy_df, features_list, labels_list)

def test_data_transform_missing_features(dummy_df):
    """Test if function data_transform aborts when no labels are introduced"""

    features_list = ["Populations", "Stress"]
    labels_list = ["Happiness"]

    with pytest.raises(ValueError):
        data_transform(dummy_df, features_list, labels_list)

def test_data_transform_unused_features(dummy_df):
    """Test if function data_transform works as intended when certain features are not used"""

    features_list = ["Population"]
    labels_list = ["Happiness"]
    X_tensor, y_tensor = data_transform(dummy_df, features_list, labels_list)

    assert X_tensor.dtype == torch.float32
    assert y_tensor.dtype == torch.float32
    assert X_tensor.shape == (3, 1)
    assert y_tensor.shape == (3, 1)



def test_build_model():
    """Test if function build_model works as intended"""

    input_size = 5
    hidden_size = 12
    batch_size = 4

    model = build_model(input_dim=input_size, hidden_dim=hidden_size)
    assert isinstance(model, nn.Sequential)

    test_input = torch.randn(batch_size, input_size)
    predictions = model(test_input)

    assert predictions.shape == (batch_size, 1)