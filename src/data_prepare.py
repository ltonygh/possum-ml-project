from pathlib import Path
from typing import List, Any
import os
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import logging

logging.basicConfig(level=logging.INFO)



def load_dataset(file_path: str) -> pd.DataFrame:
    """
        Load the dataset from a CSV file.
    """

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Pssum dataset file not found at: {file_path}")
    
    df = pd.read_csv(file_path)
    logging.info(f"Dataset path {file_path} loaded successfully")

    return df



def drop_column(df: pd.DataFrame, column_list: List[str]) -> pd.DataFrame:
    """
        Drop the items in the list.
    """

    valid_drop = set(df.columns) & set(column_list)
    if not valid_drop:
        raise ValueError("Drop column Aboirted: No matching columns in DataFrame.")
    
    missing_drop = set(column_list) - set(df.columns)
    if missing_drop:
        logging.warning(f"Target column not found in DataFrame: {missing_drop}")

    working_df = df.drop(columns = list(valid_drop))

    logging.info(f"DataFrame columns {list(valid_drop)} dropped successfully")
    return working_df



def remove_examples(df: pd.DataFrame, column_name: str, criteria: str, query_value: Any = None) -> pd.DataFrame:
    """
        Removes the examples that match the criteria in the specified column.
        
        Criteria:
        - null: The value is empty
        - notnull: The value is not empty
        - greater: The value is larger than the provided value
        - smaller: The value is smaller than the provided value
        - equal: The value is equal to the provided value
        - greater_equal: The value is greater or equal to the provided value
        - smaller_equal: The value is smaller or equal to the provided value
    """

    if column_name not in df.columns:
        logging.warning(f"Column {column_name} not found in the DataFrame")
        return df
    
    df_count = len(df)

    if criteria == "null":
        df_filtered = df[df[column_name].notnull()]
    elif criteria == "notnull":
        df_filtered = df[df[column_name].isnull()]
    elif criteria == "greater":
        df_filtered = df[df[column_name] <= query_value]
    elif criteria == "smaller":
        df_filtered = df[df[column_name] >= query_value]
    elif criteria == "equal":
        df_filtered = df[df[column_name] != query_value]
    elif criteria == "greater_equal":
        df_filtered = df[df[column_name] < query_value]
    elif criteria == "smaller_equal":
        df_filtered = df[df[column_name] > query_value]
    else:
        logging.error(f"Criteria '{criteria}' not supported.")
        return df
    
    removed_count = df_count - len(df_filtered)

    logging.info(f"Removed {removed_count} rows using criteria '{criteria}' on column '{column_name}'.")
    return df_filtered



def update_examples(df: pd.DataFrame, column_name: str, criteria: str, value: Any = None, new_value: Any = None) -> pd.DataFrame:
    """
        Updates the examples that match the criteria in the specified column with the provided new value.

        Criteria:
        - null: The value is empty
        - notnull: The value is not empty
        - greater: The value is larger than the provided value
        - smaller: The value is smaller than the provided value
        - equal: The value is equal to the provided value
        - greater_equal: The value is greater or equal to the provided value
        - smaller_equal: The value is smaller or equal to the provided value
    """

    if column_name not in df.columns:
        logging.warning(f"Column {column_name} not found in the DataFrame")
        return df
    
    if new_value is None:
        logging.warning("New value not provided")
        return df
    
    working_df  = df.copy()

    if criteria == "null":
        mask = working_df[column_name].isnull()
    elif criteria == "notnull":
        mask = working_df[column_name].notnull()
    elif criteria == "greater":
        mask = working_df[column_name] > value
    elif criteria == "smaller":
        mask = working_df[column_name] < value
    elif criteria == "equal":
        mask = working_df[column_name] == value
    elif criteria == "greater_equal":
        mask = working_df[column_name] >= value
    elif criteria == "smaller_equal":
        mask = working_df[column_name] <= value
    else:
        logging.error(f"Criteria '{criteria}' not supported.")
        return df
    
    replaced_count = mask.sum()

    if replaced_count > 0:
        working_df.loc[mask, column_name] = new_value

    logging.info(f"Updated {replaced_count} examples in column {column_name} to value {new_value}.")
    return working_df



def onehot_encode_columns(df: pd.DataFrame, column_list: List[str]) -> pd.DataFrame:
    """
        One-hot encode the listed columns.
    """

    valid_encode = set(df.columns) & set(column_list)
    if not valid_encode:
        raise ValueError("One-hot encode column Aboirted: No matching columns in DataFrame.")

    missing_encode = set(column_list) - set(df.columns)
    if missing_encode:
        logging.warning(f"Target column not found in DataFrame: {missing_encode}")
        
    working_df = df.copy()
    encoded_column = {}

    for c in valid_encode:
        encoder = OneHotEncoder(drop="first", sparse_output=False)
        encoded_array = encoder.fit_transform(working_df[[c]])
        new_col_names = encoder.get_feature_names_out([c])

        encoded_df = pd.DataFrame(
            encoded_array, 
            columns=new_col_names, 
            index=working_df.index
        )

        working_df = pd.concat([working_df.drop(columns=[c]), encoded_df], axis=1)
        encoded_column[c] = encoder
        
    logging.info(f"Column {list(valid_encode)} successfully One-Hot encoded.")
    return working_df