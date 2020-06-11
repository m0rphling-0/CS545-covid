from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from statsmodels.tools.eval_measures import rmse, aic
import pandas as pd
import glob, os
import numpy as np


def prepare_data(file):
    # returns a reduced feature data frame from a csv of daily report data 
    # for a state. It assumes a csv with only data from one state created from
    # the johns hopkins daily report data
    data = pd.read_csv(file)
    state = data.loc[0,'Province_State']
    data = data.sort_values(by='Last_Update')
    data = data[['Last_Update','Confirmed','Active','Incident_Rate','People_Tested','Testing_Rate']]
    time =  data['Last_Update']
    data.index = pd.DatetimeIndex(time)
    data = data.drop(columns=['Last_Update'])
    data.index = data.index.date
    data.index = pd.to_datetime(data.index)
    data.index = data.index.to_period('D')
    #print(data)
    return (data, state)

def create_model(data):
    # create var model from the given dataframe
    # expects a dataframe
    # assumes that the data needs to be differenced first
    # returns three tuple containing model, differenced data, and the original data
    data_diffed = data.diff()
    data_diffed = data_diffed.dropna()
    #print(data_diffed)
    model = VAR(data_diffed)
    return (model, data_diffed)

def fit_model(var_model, lag):
    return var_model.fit(lag)

def model_forecast(var_model_fit, days_to_forecast, train_data_diff, train_data):
    cast = var_model_fit.forecast(train_data_diff.values, days_to_forecast)
    last_known_index = len(train_data.index) - 1
    last_known_data  = np.array(train_data.iloc[[last_known_index]])
    cast[0] = cast[0] + last_known_data
    for i in range (1, cast.shape[0]):
        cast[i] = cast[i-1] + cast[i]
    return cast


    
def run_model_series(in_data):
    data_size = len(in_data.index)
    split_index = data_size - int(0.15*data_size)
    train_data = in_data[:split_index]
    test_data = in_data[split_index:]
    min_rmse = 10000000
    best_model = None
    for i in range(1,15):
        model, data_diffed = create_model(train_data)
        model_fit = fit_model(model, i)
        forecast = model_forecast(model_fit, len(test_data.index), data_diffed, train_data)
        error = rmse(forecast, test_data)
        if error[0] < min_rmse:
            min_rmse = error[0]
            best_model = (model, model_fit, forecast, error, test_data, train_data, data_diffed)
    return best_model




    
















