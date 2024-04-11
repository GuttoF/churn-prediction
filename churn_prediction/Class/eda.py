from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


def multiple_histplots(data: Union[int, float, str], rows: int, cols: int) -> plt.Axes:
    """
    Create multiple histogram plots for each column in the given data.

    Parameters:
    - data: The input data, which can be a DataFrame or a Series.
    - rows: The number of rows in the subplot grid.
    - cols: The number of columns in the subplot grid.

    Returns:
    - plot: The matplotlib Axes object containing the subplot grid.
    """
    for i, col in enumerate(data.columns, 1):
        plt.subplot(rows, cols, i)
        plot = sns.histplot(data[col], kde=True)
        plt.ylabel("")

    return plot


def bivariate_bins_churn(
    first: int, last: int, step: int, data: Union[int, float, str], feature: str
) -> plt.Axes:
    """
    Plot bivariate bins for churn analysis.

    Parameters:
    first (int): The starting value of the bins.
    last (int): The ending value of the bins.
    step (int): The step size between the bins.
    data (Union[int, float, str]): The input data for analysis.
    feature (str): The feature to be analyzed.

    Returns:
    plt.Axes: The matplotlib Axes object containing the plot.
    """
    # creating bins
    bins = np.arange(first, last, step)

    # creating an aux dataframe
    aux1 = data[[feature, "exited"]]
    aux1["feature_binned"] = pd.cut(aux1[feature], bins=bins)

    # separating in churn and not churn
    aux2 = aux1.loc[aux1["exited"] == 1, :]
    aux3 = aux1.loc[aux1["exited"] != 1, :]

    # creating subplots
    fig, axes = plt.subplots(1, 2, figsize=(20, 6))
    axes = axes.flatten()

    sns.countplot(x="feature_binned", data=aux2, ax=axes[0]).set_title("Churn")
    plt.xticks(rotation=90)

    sns.countplot(x="feature_binned", data=aux3, ax=axes[0]).set_title("Not in churn")
    plt.xticks(rotation=90)

    return fig


def bivariate_churn_plots(data: Union[int, float, str], feature: str) -> plt.Axes:
    """
    Generate bivariate churn plots.

    Parameters:
    data (Union[int, float, str]): The input data.
    feature (str): The feature to plot against churn.

    Returns:
    plt.Axes: The matplotlib Axes object containing the generated plots.
    """
    # creating subplots
    fig, axes = plt.subplots()
    axes = axes.flatten()

    sns.displot(
        data[data["exited"] == 1][feature], label="churn", color="#1F77B4", ax=axes[0]
    )
    sns.displot(
        data[data["exited"] == 0][feature],
        label="not in churn",
        color="#FF7F0E",
        ax=axes[0],
    )
    plt.legend(["churn", "not in churn"])

    sns.boxplot(data=data, y="exited", x=feature, ax=axes[1])
    plt.legend(["churn", "not in churn"])

    return fig


def correlation_ascending(
    data: Union[int, float, str], col: str, method: str
) -> plt.Axes:
    """
    Plot the correlation of a specific column with other numeric columns in the dataset in ascending order.

    Parameters:
        data (Union[int, float, str]): The dataset to be used for correlation calculation.
        col (str): The column name for which the correlation is to be calculated.
        method (str): The method to be used for correlation calculation.

    Returns:
        plt.Axes: The plot showing the correlation of the specified column with other numeric columns.
    """
    # correlation
    num_attributes = data.select_dtypes(include=["int64", "float64"])
    correlation = num_attributes.corr(method=method)

    correlation_asc = correlation[col].sort_values(ascending=False).to_frame()
    correlation_asc.columns = [""]
    correlation_asc.drop(col, axis=0, inplace=True)
    plot = sns.heatmap(correlation_asc, annot=True, cmap="rocket").set_title(col)

    return plot


def correlation_matrix(data: Union[int, float], method: str) -> plt.Axes:
    """
    Generate a correlation matrix heatmap for numerical attributes in the given data.

    Parameters:
    data (Union[int, float]): The input data containing numerical attributes.
    method (str): The method used to calculate the correlation.

    Returns:
    plt.Axes: The correlation matrix heatmap plot.

    """
    # correlation
    num_attributes = data.select_dtypes(include=["int64", "float64"])
    correlation = num_attributes.corr(method=method)

    # plot
    plot = sns.heatmap(
        correlation, fmt=".2f", vmin=-1, vmax=1, annot=True, cmap="magma", square=True
    )

    return plot


def correlation_matrix(data: Union[int, float], method: str) -> plt.Axes:
    """
    Generate a correlation matrix heatmap for numerical attributes in the given data.

    Parameters:
    data (Union[int, float]): The input data containing numerical attributes.
    method (str): The method used to compute the correlation matrix.

    Returns:
    plt.Axes: The heatmap plot showing the correlation matrix.

    """
    # correlation
    num_attributes = data.select_dtypes(include=["int64", "float64"])
    correlation = num_attributes.corr(method=method)

    # plot
    plot = sns.heatmap(
        correlation, fmt=".2f", vmin=-1, vmax=1, annot=True, cmap="magma", square=True
    )

    return plot


def cramers_v(categorical_attributes: Union[str, int]) -> dict:
    """
    Calculate the Cramer's V correlation coefficient for a given set of categorical attributes.

    Parameters:
    categorical_attributes (Union[str, int]): The categorical attributes to calculate the correlation for.

    Returns:
    dict: A dictionary containing the Cramer's V correlation coefficients for each pair of categorical attributes.
    """
    cat_attributes_list = categorical_attributes.columns.tolist()

    corr_dict = {}

    for i in range(len(cat_attributes_list)):
        corr_list = []
        for j in range(len(cat_attributes_list)):
            ref = cat_attributes_list[i]
            feat = cat_attributes_list[j]
            cm = pd.crosstab(
                categorical_attributes[ref], categorical_attributes[feat]
            ).to_numpy()
            n = cm.sum()
            r, k = cm.shape
            chi2 = stats.chi2_contingency(cm)[0]
            chi2corr = max(0, chi2 - (k - 1) * (r - 1) / (n - 1))
            kcorr = k - (k - 1) ** 2 / (n - 1)
            rcorr = r - (r - 1) ** 2 / (n - 1)
            corr = np.sqrt((chi2corr / n) / (min(kcorr - 1, rcorr - 1)))
            corr_list.append(corr)

        corr_dict[ref] = corr_list

    return corr_dict


def highlight_max(s):
    """
    Highlight the maximum value in a Series.

    Parameters:
    - s: pandas Series
        The Series to be highlighted.

    Returns:
    - list
        A list of CSS styles to apply to each element in the Series.
        The maximum value will be highlighted with a background color of #F15854,
        while other values will have an empty style.

    Example:
    >>> s = pd.Series([1, 2, 3, 2, 1])
    >>> highlight_max(s)
    ['', '', 'background-color: #F15854', '', '']
    """
    if s.dtype in [int, float]:
        is_max = s == s.max()
        return ["background-color: #F15854" if v else "" for v in is_max]
    else:
        return ["" for _ in s]
