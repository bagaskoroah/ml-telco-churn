import pandas as pd
from sklearn.compose import ColumnTransformer
from src.features.build_pipeline import build_pipeline
from src.config.config import NUMERICAL_COLS, CATEGORICAL_COLS


def build_preprocessing(num_pipeline, cat_pipeline):

    # build preprocessor process
    preprocessor = ColumnTransformer(transformers=[
        ('num', num_pipeline, NUMERICAL_COLS),
        ('cat', cat_pipeline, CATEGORICAL_COLS)
    ])

    return preprocessor