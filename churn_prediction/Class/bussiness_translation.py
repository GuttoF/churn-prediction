from typing import Union

import pandas as pd


def churn_rate_per_month(df) -> float: 
    """
    Calculate the monthly churn rate as a percentage based on the given DataFrame.

    Parameters:
    df (pandas.DataFrame): The DataFrame containing the data.

    Returns:
    float: The monthly churn rate as a percentage.
    """
    # Reset index of value counts
    aux1 = df['is_active_member'].value_counts().reset_index()
    
    # Filter for active members
    aux2 = aux1[aux1['is_active_member'] == 1]
    
    # Calculate churn rate
    churn_rate = aux2['is_active_member'].sum() / aux1['is_active_member'].sum()
    
    # Calculate monthly churn rate
    monthly_churn_rate = churn_rate / 12
    
    # Return monthly churn rate as a percentage
    return monthly_churn_rate * 100


def top_clients(scenario_name: str, data: Union[str, int, float], probability: str, prediction:str ,clients_return: str, churn_loss: float, number_of_clients: int, incentive_value: float) -> pd.DataFrame:
    """
    Calculate the top clients based on probability and perform various calculations on the selected clients.

    Parameters:
    - scenario_name (str): The name of the scenario.
    - data (Union[str, int, float]): The input data.
    - probability (str): The column name representing the probability.
    - prediction (str): The column name representing the prediction.
    - clients_return (str): The column name representing the clients' return.
    - churn_loss (float): The churn loss value.
    - number_of_clients (int): The number of top clients to select.
    - incentive_value (float): The incentive value to be sent to the top clients.

    Returns:
    - pd.DataFrame: A DataFrame containing various calculated metrics for the top clients.
    """
    # sort values by probability
    data.sort_values(by = probability, ascending = False, inplace = True)
    
    # select the top clients with the highest probability
    top_value = data.iloc[:number_of_clients, :]

    # send an incentive
    top_value['incentive'] = incentive_value

    # recover per client
    top_value['recover'] = top_value.apply(lambda x: x[clients_return] if x['exited'] == 1 else 0, axis = 1)

    # perfit
    top_value['profit'] = top_value['recover'] - top_value['incentive']

    # total recovered
    recovered_revenue = round(top_value['recover'].sum(), 2)

    # loss recovered in percentage
    loss_recovered = round(recovered_revenue / churn_loss * 100, 2)

    # sum of incentives
    sum_incentives = round(top_value['incentive'].sum(), 2)

    # profit sum
    profit_sum = round(top_value['profit'].sum(), 2)

    # ROI
    roi = round(profit_sum / sum_incentives * 100, 2)

    # calculate possible churn reduction in percentage
    churn_by_model = top_value[(top_value['exited'] == 1) & (top_value[prediction] == 1)]
    churn_real = round((len(churn_by_model) / len(data[data['exited'] == 1]))*100, 2)

    dataframe = pd.DataFrame({  'Scenario': scenario_name,
                                'Recovered Revenue': '$' + str(recovered_revenue),
                                'Loss Recovered': str(loss_recovered) + '%',
                                'Investment': '$' + str(sum_incentives),
                                'Profit': '$' + str(profit_sum),
                                'ROI': str(roi) + '%',
                                'Clients Recovered': str(len(churn_by_model)) + ' clients',
                                'Churn Reduction': str(churn_real) + '%'}, index = [0])

    return dataframe


def knapsack_solver(scenario_name: str, data: Union[str, int, float], prediction: str, clients_return: str, churn_loss: float, W: int, incentive: list) -> pd.DataFrame:
    """
    A knapsack problem algorithm is a constructive approach to combinatorial optimization. Given set of items, each with a specific weight and a value. The algorithm determine each item's number to include with a total weight is less than a given limit.
    reference: https://www.geeksforgeeks.org/python-program-for-dynamic-programming-set-10-0-1-knapsack-problem/
    Parameters:
        W ([int]): [Capacity of the knapsack]
        wt ([list]): [Weight of the item]
        val ([list]): [Values of the item]
        name[str]: Name of the scenario
        data[DataFrame]: [The dataframe with the churn information]
        prediction_col[col string]: [Column of the dataframe with the model prediction]
        clients_return_col[col string]: [Column of the dataframe with the financial return of the clients]
        churn_loss[float]: [The loss of the banking with the churn]
        incentive_col[list]: [Discount value in a list]

    Returns:
        [DataFrame]: [A dataframe with the results calculated]
    """
    # filter clients in churn according model
    data = data[data[prediction] == 1]

    # set parameters for the knapsack function
    val = data[clients_return].astype(int).values # return per client
    wt = data[incentive].values # incentive value per client

    # number of itens in values
    n = len(val)

    # set K with 0 values
    K = [[0 for x in range(W + 1)] for x in range(n + 1)]
    for i in range(n + 1):
        for w in range(W + 1):
            if i == 0 or w == 0:
                K[i][w] = 0
            elif wt[i-1] <= w:
                K[i][w] = max(val[i-1] + K[i-1][w-wt[i-1]], K[i-1][w])
            else:
                K[i][w] = K[i-1][w]
    max_val = K[n][W]

    # select items that maximizes the output
    keep = [False] * n
    res = max_val
    w = W
    for i in range(n, 0, -1):
        if res <= 0: break
        if res == K[i - 1][w]: continue
        else:
            keep[i - 1] = True
            res = res - val[i - 1]
            w = w - wt[i - 1]

    # dataframe with selected clients that maximizes output value
    data = data[keep]

    # Recover per client
    data['recover'] = data.apply(lambda x: x[clients_return] if x['exited'] == 1 else 0, axis = 1)
    
    # Calculate prefit
    data['profit'] = data['recover'] - data['incentive']
    
    # Calculate the total recovered
    recovered_revenue = round(data['recover'].sum(), 2)
    
    # Calculate loss recovered in percent
    loss_recovered = round((recovered_revenue/churn_loss)*100, 2)
    
    # Calculate the sum of incentives
    sum_incentives = round(data['incentive'].sum(), 2)
    
    # Calculate profit sum
    profit = round(data['profit'].sum(), 2)
    
    # Calculate ROI in percent
    roi = round((profit/sum_incentives)*100, 2)
    
    # Calculate possible churn reduction in %
    churn_by_model = data[(data['exited'] == 1) & (data[prediction] == 1)]
    churn_real = round((len(churn_by_model) / len(data[data['exited'] == 1]))*100, 2)
    
    dataframe = pd.DataFrame({ 'Scenario': scenario_name,
                            'Recovered Revenue': '$' + str(recovered_revenue),
                            'Loss Recovered': str(loss_recovered) + '%',
                            'Investment': '$' + str(sum_incentives),
                            'Profit': '$' + str(profit),
                            'ROI': str(roi) + '%',
                            'Clients Recovered': str(len(churn_by_model)) + ' clients',
                            'Churn Reduction': str(churn_real) + '%'}, index = [0])
    
    del K
    return dataframe