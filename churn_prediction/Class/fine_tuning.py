from typing import Union

import numpy as np
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold


def catboost_objective(
    trial: int,
    X: Union[int, float, str],
    y: Union[int, float, str],
    weight: float,
    threshold: float,
    kfold: int = 5,
    selected_score: str = "recall",
) -> float:
    """
    Catboost objective function for hyperparameter tuning.

    Parameters:
        trial (int): The trial number.
        X (Union[int, float, str]): The input features.
        y (Union[int, float, str]): The target variable.
        weight (float): The scale_pos_weight parameter for CatBoostClassifier.
        threshold (float): The threshold for classification.
        kfold (int, optional): The number of folds for stratified k-fold cross-validation. Defaults to 5.
        selected_score (str, optional): The selected score for evaluation. Defaults to "recall".

    Returns:
        float: The mean score of the selected metric across all folds.
    """
    # Catboost parameters
    param_grid = {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
        "depth": trial.suggest_int("depth", 3, 10),
        "n_estimators": trial.suggest_int("n_estimators", 100, 2000, step=100),
    }

    model = CatBoostClassifier(scale_pos_weight=weight, verbose=False, **param_grid)

    # Stratified Kfold
    folds = StratifiedKFold(n_splits=kfold)

    # List of recalls for each fold
    score_list = []

    for train_cv, val_cv in folds.split(X, y):
        # Split into train and validation
        X_train_fold = X.iloc[train_cv]
        y_train_fold = y.iloc[train_cv]
        X_val_fold = X.iloc[val_cv]
        y_val_fold = y.iloc[val_cv]

        # Train the model
        model.fit(X_train_fold, y_train_fold)

        # Predict the validation fold
        y_pred_val = model.predict_proba(X_val_fold)[:, 1]

        if selected_score == "recall":
            score_val = recall_score(y_val_fold, y_pred_val >= threshold)
        elif selected_score == "precision":
            score_val = precision_score(y_val_fold, y_pred_val >= threshold)
        elif selected_score == "f1":
            score_val = f1_score(y_val_fold, y_pred_val >= threshold)
        else:
            print("Please, select a valid score")

        # Add to the list
        score_list.append(score_val)

    mean_score = np.mean(score_list)

    return mean_score
