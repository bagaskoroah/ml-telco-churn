import pandas as pd
from src.config.config import TARGET

def read_data(input_data: pd.DataFrame) -> pd.DataFrame:
    """
    Load raw dataset from the configured DATA_PATH.

    Returns
    -------
    pd.DataFrame
        Raw dataframe before any cleaning.
    """
    # open the data
    data = pd.read_csv(input_data)

    # define data shape
    print('Data shape:', data.shape)

    return data

def clean_data(raw_data: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw data by removing ID columns, dropping duplicates,
    and handling invalid empty strings in the TotalCharges column.

    Parameters
    ----------
    raw_data : pd.DataFrame
        Raw dataset before cleaning.

    Returns
    -------
    pd.DataFrame
        Cleaned dataset.
    """
    # copy raw data as a backup
    data = raw_data.copy(deep=True)

    # drop customerID column
    data.drop(columns=['customerID'], inplace=True)

    # drop duplicates
    print('\nCleaning data process. . .')
    print('Duplicates found in data:', data.duplicated().sum())

    # remove duplicates and keep the last ones
    data.drop_duplicates(keep='last', inplace=True)

    # re-check data shape after dropping duplicates
    print('Data shape after dropping duplicates:', data.shape)

    return data

def replace_target(cleaned_data: pd.DataFrame) -> pd.DataFrame: 
    """
    Replace the categorical target column (e.g., Yes/No)
    with numerical binary values (1/0).

    Parameters
    ----------
    cleaned_data : pd.DataFrame
        Dataset after cleaning, containing the target column.

    Returns
    -------
    pd.DataFrame
        Dataset with target column converted to numerical values.
    """
    # print category before replace
    print('\nReplacing target categories. . .')
    print('Target categories before replace:', cleaned_data[TARGET].unique())

    # replace categorical to numerical
    cleaned_data[TARGET] = cleaned_data[TARGET].map({'Yes': 1,'No': 0})
    
    # print category after replace
    print('Target categories after replace:', cleaned_data[TARGET].unique())

    return cleaned_data
    
    

