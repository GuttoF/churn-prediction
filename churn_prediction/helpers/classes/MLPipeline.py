import logging

import numpy as np
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    MinMaxScaler,
    OneHotEncoder,
    RobustScaler,
    StandardScaler,
)
from yellowbrick.model_selection import FeatureImportances


class MLPipeline:
    """
    A class representing a machine learning pipeline.

    Parameters:
    - X: The input features for training the model.
    - y: The target variable for training the model.
    - model: The machine learning model to be trained.

    Methods:
    - build_pipeline: Builds the preprocessing pipeline based on the specified steps.
    - train_model: Trains the model using the built pipeline.
    - evaluate_model: Evaluates the trained model on the given input features and target variable.
    """

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
        """
        Builds the preprocessing pipeline based on the specified steps.

        Parameters:
        - log_list: A list of features to be log-transformed.
        - ohe_list: A list of features to be one-hot encoded.
        - robust_scaler_list: A list of features to be scaled using RobustScaler.
        - min_max_scaler_list: A list of features to be scaled using MinMaxScaler.
        - standard_scaler_list: A list of features to be scaled using StandardScaler.
        """
        preprocessing_steps = []

        if ohe_list:
            logging.info("One-hot encoding features: %s", ohe_list)
            preprocessing_steps.append(
                ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
            )  # Set sparse_output to False to output a dense matrix

        if log_list:
            logging.info("Log-transforming features: %s", log_list)
            preprocessing_steps.append(("logtransform", FunctionTransformer(np.log1p)))

        scalers = {
            "robustscaler": lambda: RobustScaler(
                with_centering=False
            ),  # Modified to not center data
            "minmaxscaler": MinMaxScaler,
            "standardscaler": StandardScaler,
        }

        for scaler_type, features in [
            ("robustscaler", robust_scaler_list),
            ("minmaxscaler", min_max_scaler_list),
            ("standardscaler", standard_scaler_list),
        ]:
            if features:  # Add scaler to pipeline only if there are features specified
                logging.info("Scaling features %s using %s", features, scaler_type)
                preprocessing_steps.append((scaler_type, scalers[scaler_type]()))

        self.pipeline = Pipeline(steps=preprocessing_steps)

    def apply_boruta(self, n_estimators=500, max_iter=150, alpha=0.05):
        """
        Performs feature selection using the Boruta algorithm and visualizes feature importances.
    
        Args:
            n_estimators (int, optional): The number of trees in the random forest. Defaults to 500.
            max_iter (int, optional): The maximum number of iterations to perform. Defaults to 150.
            alpha (float, optional): The significance level to determine feature importance. Defaults to 0.05.
    
        Returns:
            dict: A dictionary containing the selected features and their importances.
        """
        if not isinstance(self.model, RandomForestClassifier):
            raise ValueError("Boruta only supports RandomForestClassifier.")
    
        self.model.set_params(n_jobs=-1)  # Use all available cores
    
        logging.info("Running Boruta feature selection...")
        X_transformed = self.pipeline.fit_transform(self.X)
    
        # Fixing the numpy bug in the current version
        np.int = np.int32
        np.float = np.float64
        np.bool = np.bool_
    
        # Initialize BorutaPy with adjusted parameters
        boruta_selector = BorutaPy(
            self.model, n_estimators=n_estimators, max_iter=max_iter, alpha=alpha
        )
        boruta_selector.fit(X_transformed, self.y)
        features_selected = boruta_selector.support_
    
        # Get the feature names from the original data
        feature_names = [f'feature{i}' for i in range(X_transformed.shape[1])]  # Modify this line to match how your features are named or passed
    
        # Filter the feature names based on what Boruta selected
        selected_features_names = [name for idx, name in enumerate(feature_names) if features_selected[idx]]
    
        # Using Yellowbrick to visualize feature importances
        visualizer = FeatureImportances(self.model, labels=selected_features_names)
        visualizer.fit(X_transformed, self.y)
        visualizer.show()
    
        return {
            "features_selected": features_selected,
            "features_importances_": self.model.feature_importances_,
        }

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
