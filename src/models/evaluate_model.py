from typing import Tuple, Any
from sklearn.metrics import recall_score, precision_score, f1_score, \
    confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

def evaluate_baseline(y_train: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float]:
    """
    Calculate baseline evaluation metrics (recall, precision, f1-score).

    Parameters
    ----------
        y_train (np.ndarray): True labels from the training set.
        y_pred (np.ndarray): Predicted labels from the model.

    Returns
    -------
        Tuple[float, float, float]: Recall, precision, and F1 score.
    """
    base_recall = recall_score(y_train, y_pred)
    base_precision = precision_score(y_train, y_pred)
    base_f1 = f1_score(y_train, y_pred)

    return base_recall, base_precision, base_f1

def evaluate_cv_train(
    cv_model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray
) -> Tuple[Any, Any, float, float, float, float, float, float]:
    """
    Evaluate both cross-validation performance and training performance.

    Parameters
    ----------
        cv_model (Any): A fitted sklearn CV model (e.g., GridSearchCV, RandomizedSearchCV).
        x_train (np.ndarray): Feature labels from the training set.
        y_train (np.ndarray): True labels from the training set.

    Returns
    -------
        Tuple[float, float, float, float, float, float]:
            - CV best params
            - CV best model
            - CV recall
            - CV precision
            - CV F1 score
            - Train recall
            - Train precision
            - Train F1 score
    """

    # get the best cv param and best cv model
    best_param = cv_model.best_params_
    best_model = cv_model.best_estimator_

    # predict & evaluate train data
    y_pred = best_model.predict(X_train)

    # evaluate cv
    recall_cv = cv_model.cv_results_['mean_test_recall'].max()
    precision_cv = cv_model.cv_results_['mean_test_precision'].max()
    f1_cv = cv_model.cv_results_['mean_test_f1'].max()

    # evaluate train
    recall_train = recall_score(y_train, y_pred)
    precision_train = precision_score(y_train, y_pred)
    f1_train = f1_score(y_train, y_pred)

    return best_param, best_model, recall_cv, precision_cv, f1_cv, recall_train, precision_train, f1_train

def evaluate_test(y_test: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float]:
    """
    Evaluate model performance on the test set.

    Parameters
    ----------
        y_test (np.ndarray): True labels from the test set.
        y_pred (np.ndarray): Predicted labels from the model.

    Returns
    -------
        Tuple[float, float, float]: Test recall, precision, and F1 score.
    """
    recall_test = recall_score(y_test, y_pred)
    precision_test = precision_score(y_test, y_pred)
    f1_test = f1_score(y_test, y_pred)

    return recall_test, precision_test, f1_test

def confusion(
    y_test: np.ndarray,
    y_pred: np.ndarray
) -> Tuple[ConfusionMatrixDisplay, int, int, int, int]:
    """
    Generate and display a confusion matrix plot, and return the underlying values.

    Parameters
    ----------
        y_test (np.ndarray): True labels from the test set.
        y_pred (np.ndarray): Predicted labels from the model.

    Returns
    -------
        Tuple[
            ConfusionMatrixDisplay,
            int, int, int, int
        ]:
            - ConfusionMatrixDisplay object
            - True Negative (TN)
            - False Positive (FP)
            - False Negative (FN)
            - True Positive (TP)
    """
    # define cm object
    cm = confusion_matrix(y_test, y_pred)
    display = ConfusionMatrixDisplay(confusion_matrix=cm)

    # plot the cm display
    display.plot()

    # unpack the each component value
    tn, fp, fn, tp = cm.ravel()

    # show the plot 
    plt.show()

    return display, tn, fp, fn, tp