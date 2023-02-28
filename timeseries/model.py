import json
import joblib
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt           
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller

       
def train_test_split(y,train_rate):
    """
    Split train and test data
        Parameters
        ----------
        trainRate: Rate between train and test
        testRate = 1 - trainRate
        Returns
        ----------
    """
    if train_rate is None:
        train_rate =0.8
        # actual - unknown
        msk = (y.isna())
        y_train = y[~msk].copy()
        y_test = y[msk].copy()
    else:
        train_rate = train_rate
        # train-test data
        msk = (y.dropna().index < len(y.dropna())*train_rate)
        y_train = y.dropna()[msk].copy()     
        y_test = y.dropna()[~msk].copy()
    return y_train, y_test

def train(y, order, seasonal_order):
    
    if order is None and seasonal_order is None:
        order = (0,1,1)
        seasonal_order = (0, 1, 0, 12)
    else:
        pass
    model = SARIMAX(y, 
            order=order,
            seasonal_order=seasonal_order,
            trend = 't',
            enforce_stationary = False,
            enforce_invertibility = False)
    model_fit = model.fit()
    return model_fit

def predict(self):
    """
    Parameters
    ----------
    yhat: the fitted value of y across time frame
    y_fc: the predicted value of out-sample y
    Returns
    ----------
    """
    self.y_fc = self.model_fit.forecast(len(self.y_test.index))
    self.yhat = self.model_fit.predict()[1:]

def save_model(model_fit,model_path):
    if model_path is None:
        model_path = 'output/default.sav'
    else:
        model_path = model_path
    joblib.dump(model_fit, model_path)

def save_pred_data(self):
    # drop first value for compatibiltiy with the len(yhat)
    x = self.x[1:].copy()
    y = self.y[1:].copy()
    if len(x) > len(self.yhat):
        self.y_pred = np.concatenate([self.yhat,self.y_fc])
    else:
        self.y_pred = self.yhat
    dt = pd.DataFrame({ "date": x, 
                        "actual": y,
                        "predicted": self.y_pred})
    dt.to_csv('output/predicted.csv')


