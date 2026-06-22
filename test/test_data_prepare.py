import pandas as pd
import pytest
from src.data_prepare import load_dataset, drop_column, remove_examples, update_examples, onehot_encode_columns



@pytest.fixture
def dummy_df() -> pd.DataFrame:
    """
    Creates a dummy DataFrame to test the following functions:
    - drop_column
    - remove_examples
    - update_examples
    - onehot_encode_column

    The dummy DataFrame is built with the following attributes:
    - StudentID:[1, 2, 3]
    - Sex:      [M, F, M]
    - Score:    [80, 75, None]
    - RevTime:  [12, 10, 11]
    - Region:   [MD, RR, MD]
    """

    data = {
        "SID":      [1, 2, 3],
        "Sex":      ["M", "F", "M"],
        "Score":    [85, 75, None],
        "RevTime":  [12, 10, 13],
        "Region":   ["MD", "RR", "MD"]
    }

    return pd.DataFrame(data)



def test_drop_column(dummy_df):
    """Test if function drop_column works as intended"""

    columns = ["SID"]
    new_df = drop_column(dummy_df, columns)

    assert "SID" not in new_df.columns
    assert "Sex" in new_df.columns
    assert new_df.shape[1] == dummy_df.shape[1] - len(columns)



def test_remove_examples(dummy_df):
    """Test if the function remove_examples works as intended"""

    column_name = "Score"
    criteria = "null"
    new_df = remove_examples(dummy_df, column_name, criteria)

    assert new_df["Score"].isnull().sum() == 0
    assert new_df.shape[0] == dummy_df.shape[0] - 1



def test_update_examples(dummy_df):
    """Test if the function update_examples works as intended"""

    column_name =  "Score"
    criteria = "null"
    new_value = 90
    new_df = update_examples(dummy_df, column_name, criteria, None, new_value)

    assert new_df["Score"].isnull().sum() == 0
    assert new_df.loc[2, "Score"] == new_value
    assert new_df.shape[0] == dummy_df.shape[0]



def test_onehot_encode_columns(dummy_df):
    """Test if the function onehot_encode_columns works as intended"""

    column_name = ["Sex", "Region"]
    new_df = onehot_encode_columns(dummy_df, column_name)

    assert "Sex" not in new_df.columns
    assert "Region" not in new_df.columns
    assert "Sex_M" in new_df.columns
    assert "Region_RR" in new_df.columns