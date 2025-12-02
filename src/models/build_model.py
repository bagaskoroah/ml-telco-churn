from sklearn.dummy import DummyClassifier
from sklearn.model_selection import RandomizedSearchCV
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.metrics import make_scorer, recall_score, precision_score, f1_score
from typing import Any, Tuple, Dict
import time
import numpy as np
import pandas as pd
from src.config import config

def build_baseline(
    X_train: pd.DataFrame, 
    y_train: pd.Series
) -> np.ndarray:
    """
    Build and train a baseline DummyClassifier using a stratified strategy.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training labels.

    Returns
    -------
    np.ndarray
        Predictions from the baseline model on training data.
    """
    # build the baseline object
    base_model = DummyClassifier(strategy='stratified')
    
    # train the model
    base_model.fit(X_train, y_train)

    # predict and evaluate baseline
    base_model_pred = base_model.predict(X_train)

    return base_model_pred

def build_cv_train(
    estimator: Any,
    preprocessor: Any,
    params: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series
) -> Tuple[np.ndarray, Any, Dict[str, Any]]:
    """
    Perform cross-validated model training with preprocessing and SMOTE pipeline.
    Evaluates the best model on training data and returns predictions + best model.

    Parameters
    ----------
    estimator : Any
        Machine learning estimator to train.
    preprocessor : Any
        Preprocessing transformer.
    params : dict
        Hyperparameter search space for RandomizedSearchCV.
    X_train : pd.DataFrame
        Training input features.
    y_train : pd.Series
        Training labels.

    Returns
    -------
    tuple
        y_train_pred : np.ndarray  
            Predictions of the best model on training data.
        best_model : Any  
            Best fitted estimator returned by randomized search.
        best_param : dict  
            Best-found hyperparameters.
    """

    # define start time process
    start_time = time.time()

    # map cv scoring
    scoring = {
        'recall': make_scorer(recall_score),
        'precision': make_scorer(precision_score),
        'f1': make_scorer(f1_score)
    }
    
    # build the model object
    model = ImbPipeline(steps=[
    ('preprocessing', preprocessor),
    ('smote', SMOTE(random_state=123)),
    ('model', estimator)
    ])

    # define cv object
    cv_model = RandomizedSearchCV(
        estimator=model,
        param_distributions=params,
        n_iter=config.N_ITER,
        cv=config.CV,
        scoring=scoring,
        refit='recall',
        n_jobs=config.N_JOBS
        )
    
    # train cv model
    cv_model.fit(X_train, y_train)

    end_time = time.time() - start_time
    end_time = round(end_time/60, 2)

    print(f'Model {estimator.__class__.__name__} has been successfully created for {end_time} minutes.')

    return cv_model

def build_test(
    test_estimator: Any,
    X_test: pd.DataFrame
) -> np.ndarray:
    """
    Generate predictions from the final trained estimator on the test data.

    Parameters
    ----------
    test_estimator : Any
        Trained model used to generate predictions.
    X_test : pd.DataFrame
        Test features.

    Returns
    -------
    np.ndarray
        Predicted labels on the test set.
    """

    # define start time
    start_time = time.time()

    # predict model
    y_test_pred = test_estimator.predict(X_test)

    # calculate process time
    end_time = time.time() - start_time
    end_time = round(end_time/60, 2)

    return y_test_pred
    


