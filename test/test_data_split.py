import pytest
import numpy as np
import pandas as pd
from src.data_split import split_data



@pytest.fixture
def dummy_dataset_df() -> pd.DataFrame:
    """
        Creates a dummy DataFrame already cleaned to test data splitting.

        The dummy DataFrame is built with the following attributes:
        - Feature 1:        [5.0, 4.5, 3.4, 6.1, 5.7, 4.3, 2.7, 3.9, 1.3, 5.6]
        - Feature 2:        [21,  19,  13,  28,  26,  17,  11,  14, 6, 15]
        - Binary label:     ["T", "T", "F", "T", "T", "F", "F", "F", "F", "T"]
        - Continuous label: [4, 5, 8, 1, 2, 6, 9, 7, 10, 3]
    """

    data = {
        "feat_1": [5.0, 4.5, 3.4, 6.1, 5.7, 4.3, 2.7, 3.9, 1.3, 5.6],
        "feat_2": [21,  19,  13,  28,  26,  17,  11,  14, 6, 15],
        "binary_label": ["T", "T", "F", "T", "T", "F", "F", "F", "F", "T"],
        "continuous_label":  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    }
    return pd.DataFrame(data)



def test_data_split_no_stratify(dummy_dataset_df):
    """
        Test if function data_split works with the continuous labels unstratified.
    """

    features = ["feat_1", "feat_2"]
    target = "continuous_label"

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        df = dummy_dataset_df,
        feature_list = features,
        target_col = target,
        stratify_target = False,
        train_validate_test_ratio = (0.7, 0.1, 0.2),
        random_state = 42
    )

    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(y_train, pd.Series)

    total_samples = len(X_train) + len(X_val) + len(X_test)
    assert total_samples == len(dummy_dataset_df)
    assert len(X_train) == len(y_train)

    assert np.isclose(len(X_train), total_samples * 0.7, atol=2)
    assert np.isclose(len(X_val), total_samples * 0.1, atol=2)
    assert np.isclose(len(X_test), total_samples * 0.2, atol=2)



def test_data_split_stratify(dummy_dataset_df):
    """
        Test if function data_split works with the binary labels stratified.
    """

    features = ["feat_1", "feat_2"]
    target = "binary_label"

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        df = dummy_dataset_df,
        feature_list = features,
        target_col = target,
        stratify_target = True,
        train_validate_test_ratio = (0.6, 0.2, 0.2),
        random_state = 42
    )

    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(y_train, pd.Series)

    total_samples = len(X_train) + len(X_val) + len(X_test)
    assert total_samples == len(dummy_dataset_df)
    assert len(X_train) == len(y_train)

    assert np.isclose(len(X_train), total_samples * 0.6, atol=2)
    assert np.isclose(len(X_val), total_samples * 0.2, atol=2)
    assert np.isclose(len(X_test), total_samples * 0.2, atol=2)

    assert (y_train == "T").sum() == 3
    assert (y_train == "F").sum() == 3
    assert (y_val == "T").sum() == 1
    assert (y_val == "F").sum() == 1
    assert (y_test == "T").sum() == 1
    assert (y_test == "F").sum() == 1


def test_split_data_invalid_ratio(dummy_dataset_df):
    """Ensures a ValueError is raised if ratio array entries do not equal 100%."""
    features = ["feat_1, feat_2"]
    target = "continuous_label"
    
    with pytest.raises(ValueError, match = "Ratio of each set does not sum to 100%"):
        split_data(
            df = dummy_dataset_df,
            feature_list = features,
            target_col = target,
            stratify_target = False,
            train_validate_test_ratio=(0.5, 0.5, 0.5)
        )