import numpy as np
import logging
import pandas as pd
from typing import Tuple, List
from sklearn.model_selection import train_test_split
from src.data_prepare import remove_examples

logging.basicConfig(level=logging.INFO)



def split_data(
    df: pd.DataFrame,
    feature_list: List[str],
    target_col: str,
    stratify_target: bool,
    train_validate_test_ratio: Tuple[float] = (0.7, 0.1, 0.2),
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
        Split the provided DataFrame according to the train-validation-test ratio.
        By default, the random state is defined as 42, and the split ratio for training set, validation set, and test set, is 70%, 10%, and 20% respectively.
        The ratio of validation set can be set to 0.0 to remove validation set.
    """

    if target_col not in df.columns:
        raise KeyError(f"Function split_data aborted: Column '{target_col}' dpes not exist in the DataFrame.")

    train_ratio, val_ratio, test_ratio = train_validate_test_ratio
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("Function load_and_split_raw_data aborted: Ratio of each set does not sum to 100%")
    
    cleaned_df = df.dropna().copy()
    X = cleaned_df[feature_list].apply(pd.to_numeric, errors="coerce")
    y = cleaned_df[target_col]
    
    stratify_test = y if stratify_target == True else None
    
    X_remain, X_test, y_remain, y_test = train_test_split(
        X, y, 
        test_size = test_ratio, 
        random_state = random_state, 
        stratify = stratify_test
    )

    if not val_ratio == 0.0:
        val_fraction = val_ratio / (train_ratio + val_ratio)
        stratify_val = y_remain if stratify_target == True else None

        X_train, X_val, y_train, y_val = train_test_split(
            X_remain, y_remain, 
            test_size = val_fraction, 
            random_state = random_state, 
            stratify = stratify_val
        )
    else:
        X_train, y_train = X_remain, y_remain
        X_val = pd.DataFrame(columns=feature_list)
        y_val = pd.Series(dtype=y.dtype, name=target_col)
    
    logging.info(f"Splitted samples for 3 classes. Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test