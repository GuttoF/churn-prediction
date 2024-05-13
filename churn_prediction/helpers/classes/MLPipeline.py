import logging

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    MinMaxScaler,
    OneHotEncoder,
    RobustScaler,
    StandardScaler,
)


class MLPipeline:
    def __init__(self, X, y, model):
        self.X = X
        self.y = y
        self.model = model
        self.pipeline = None

    def build_pipeline(
        self,
        log_list,
        ohe_list,
        robust_scaler_list=[],
        min_max_scaler_list=[],
        standard_scaler_list=[],
    ):
        preprocessing_steps = []
        if ohe_list:
            logging.info("One-hot encoding features: %s", ohe_list)
            preprocessing_steps.append(
                ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
            )

        if log_list:
            logging.info("Log-transforming features: %s", log_list)
            preprocessing_steps.append(("logtransform", FunctionTransformer(np.log1p)))

        scalers = {
            "robustscaler": lambda: RobustScaler(with_centering=False),
            "minmaxscaler": MinMaxScaler,
            "standardscaler": StandardScaler,
        }
        for scaler_type, features in [
            ("robustscaler", robust_scaler_list),
            ("minmaxscaler", min_max_scaler_list),
            ("standardscaler", standard_scaler_list),
        ]:
            if features:
                logging.info("Scaling features %s using %s", features, scaler_type)
                preprocessing_steps.append((scaler_type, scalers[scaler_type]()))

        self.pipeline = Pipeline(steps=preprocessing_steps)

    def train_model(self):
        """
        Trains the model using the built pipeline.

        Raises:
        - ValueError: If the pipeline is not built yet.
        """
        if self.pipeline is None:
            raise ValueError(
                "Pipeline is not built yet. Please build the pipeline first."
            )

        logging.info(f"Training the {self.model}...")
        X_transformed = self.pipeline.fit_transform(self.X)
        self.model.fit(X_transformed, self.y)

    def evaluate_model(self, X, y, threshold=0.5):
        """
        Evaluates the trained model on the given input features and target variable.

        Parameters:
        - X: The input features for evaluation.
        - y: The target variable for evaluation.
        - threshold: The decision threshold for classifying observations as positive. Default is 0.5.
        """
        X_transformed = self.pipeline.transform(X)
        y_probs = self.model.predict_proba(X_transformed)[
            :, 1
        ]  # get probabilities for the positive class
        y_pred = (y_probs >= threshold).astype(
            int
        )  # apply threshold to get final predictions

        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred, average="binary")
        recall = recall_score(y, y_pred, average="binary")
        f1 = f1_score(y, y_pred, average="binary")
        logging.warning(f"Your threshold is {threshold}")
        logging.info(f"Evaluating the {self.model}...")
        print(
            f"Accuracy: {accuracy}\nPrecision: {precision}\nRecall: {recall}\nF1: {f1}"
        )
