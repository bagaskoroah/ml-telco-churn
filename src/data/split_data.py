import pandas as pd
from typing import Tuple
from sklearn.model_selection import train_test_split

from src.config.config import TEST_SIZE, RANDOM_STATE, TARGET


def split_train_test(cleaned_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split the cleaned dataset into training and test sets.

    Parameters
    ----------
    cleaned_df : pd.DataFrame
        DataFrame that has been cleaned and contains the target column.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
        X_train, X_test, y_train, y_test split using the configured
        TEST_SIZE and RANDOM_STATE.
    """
    X = cleaned_df.drop(columns=[TARGET])
    y = cleaned_df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE)

    print('X train shape:', X_train.shape)
    print('X test shape:', X_test.shape)
    print('y train shape:', y_train.shape)
    print('y test shape:', y_test.shape)

    return X_train, X_test, y_train, y_test