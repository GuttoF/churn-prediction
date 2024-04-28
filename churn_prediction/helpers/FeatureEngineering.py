import logging
import os
from typing import Union

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.model_selection import train_test_split

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")


class FeatureEngineering:
    """
    Class for performing feature engineering on a given dataframe.

    Parameters:
    - seed (int): Random seed for reproducibility.
    """

    def __init__(self, seed=None):
        self.seed = seed

    def transform(self, dataframe: pd.DataFrame) -> "FeatureEngineering":
        """
        Perform feature engineering on the given dataframe.

        Parameters:
        - dataframe (pandas.DataFrame): Input dataframe to be transformed.

        Returns:
        - X_train, X_test, X_val, y_train, y_test, y_val, id_train, id_test, id_val (pandas.DataFrame): Transformed dataframes.
        """
        # Drop specified columns
        dataframe.drop(columns=["surname", "row_number"], inplace=True)

        # Define X, y, and ids
        X = dataframe.drop(columns=["exited", "customer_id"])
        y = dataframe["exited"]
        ids = dataframe["customer_id"]

        # Train-test split
        X_train_temp, X_test, y_train_temp, y_test, id_train_temp, id_test = (
            train_test_split(
                X, y, ids, test_size=0.2, random_state=self.seed, stratify=y
            )
        )

        # Validation split
        X_train, X_val, y_train, y_val, id_train, id_val = train_test_split(
            X_train_temp,
            y_train_temp,
            id_train_temp,
            test_size=0.125,
            random_state=self.seed,
            stratify=y_train_temp,
        )

        # List of dataframes for transformations
        dataframes = [X_train, X_test, X_val]

        for dataframe in dataframes:
            # Perform transformations
            dataframe = self._perform_transformations(dataframe)

        return X_train, X_test, X_val, y_train, y_test, y_val, id_train, id_test, id_val

    def _perform_transformations(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Perform feature transformations on the given dataframe.

        Parameters:
        - dataframe (pandas.DataFrame): Input dataframe to be transformed.

        Returns:
        - dataframe (pandas.DataFrame): Transformed dataframe.
        """
        # Non-linear transformations
        dataframe["age_squared"] = dataframe["age"] ** 2
        dataframe["balance_sqrt"] = np.sqrt(dataframe["balance"])
        dataframe["log_credit_score"] = np.log(dataframe["credit_score"])

        # Multiplicative interactions
        dataframe["credit_score_num_of_products"] = (
            dataframe["credit_score"] * dataframe["num_of_products"]
        )
        dataframe["age_balance"] = dataframe["age"] * dataframe["balance"]
        dataframe["engagement_score"] = (
            (dataframe["is_active_member"] * 0.5)
            + (dataframe["has_cr_card"] * 0.3)
            + (dataframe["num_of_products"] * 0.2)
        )

        # Division interactions
        epsilon = 1e-6
        dataframe["customer_value_normalized"] = (
            dataframe["balance"] / dataframe["balance"].max()
            + dataframe["num_of_products"] / dataframe["num_of_products"].max()
            + dataframe["estimated_salary"] / dataframe["estimated_salary"].max()
        )
        dataframe["product_density"] = dataframe["num_of_products"] / (
            dataframe["balance"] + epsilon
        )
        dataframe["balance_salary_ratio"] = (
            dataframe["balance"] / dataframe["estimated_salary"]
        )
        dataframe["credit_score_age_ratio"] = (
            dataframe["credit_score"] / dataframe["age"]
        )
        dataframe["tenure_age_ratio"] = dataframe["tenure"] / dataframe["age"]
        dataframe["credit_salary_ratio"] = (
            dataframe["credit_score"] / dataframe["estimated_salary"]
        )

        # Indicator transformations
        dataframe["balance_indicator"] = np.where(
            dataframe["balance"] >= 150000, "high", "low"
        )

        # Binning transformations
        dataframe["life_stage"] = pd.cut(
            dataframe["age"],
            bins=[0, 20, 35, 50, np.inf],
            labels=["adolescence", "adulthood", "middle_age", "senior"],
        )
        dataframe["cs_category"] = pd.cut(
            dataframe["credit_score"],
            bins=[0, 500, 700, np.inf],
            labels=["low", "medium", "high"],
        )
        dataframe["tenure_group"] = pd.cut(
            dataframe["tenure"],
            bins=[-1, 3, 6, np.inf],
            labels=["new", "intermediate", "long_standing"],
        )

        # Group by transformations
        dataframe["credit_score_gender"] = dataframe.groupby("gender")[
            "credit_score"
        ].transform("mean")
        dataframe["balance_age"] = dataframe.groupby("age")["balance"].transform("mean")
        dataframe["salary_rank_geography"] = dataframe.groupby("geography")[
            "estimated_salary"
        ].rank(method="dense")

        return dataframe

    def report_na(self, dataframe: pd.DataFrame) -> logging.info:
        """
        Reports the number of missing values (NA) in each column of the given dataframe.

        Args:
            dataframe (pd.DataFrame): The dataframe to check for missing values.

        Returns:
            logging.info: A message indicating the number of missing values in each column.
        """
        if not isinstance(dataframe, pd.DataFrame):
            logging.error("Input is not a pandas DataFrame.")
            return

        for column in dataframe.columns:
            na_count = dataframe[column].isna().sum()
            if na_count > 0:
                return logging.info(
                    f"There are {na_count} NA in the '{column}' column."
                )
        return logging.info("There are no NA values in any column.")

    def save_data(
        self, dataframe: Union["float", "int", "str"], file_path: str
    ) -> None:
        """
        Save the given dataframe to a Parquet file.

        Args:
            dataframe (pandas.DataFrame): The dataframe to be saved.
            file_path (str): The path to the output file.

        Returns:
            None

        Raises:
            Exception: If there is an error while saving the data.

        """
        try:
            if os.path.exists(file_path):
                logging.warning("File already exists. It will be overwritten.")

            table = pa.Table.from_pandas(dataframe)
            pq.write_table(table, file_path)
            logging.info("Data successfully saved")
        except Exception as e:
            logging.error(f"Failed to save data: {str(e)}")
