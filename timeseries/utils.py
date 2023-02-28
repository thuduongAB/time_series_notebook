import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functools import partial

from statsmodels.tsa.stattools import adfuller, kpss
from pmdarima.arima import auto_arima

def process_data(df):
    df['Date'] = pd.to_datetime(df['Period'], dayfirst=True)
    df.reset_index(inplace = True)
    return df

def nDifferencing(df, nOrder):

    df[str(nOrder)+"-orderdiff"] = df["Sales_quantity"] - df['Sales_quantity'].shift(nOrder)
    return df

def test_stationarity(Y):
    #Perform Dickey–Fuller test:
    print('Results of Dickey Fuller Test:')
    result = adfuller(Y.dropna(),autolag='AIC')
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))

def adf_test(timeseries):
    print ('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput)

def kpss_test(timeseries):
    print ('Results of KPSS Test:')
    kpsstest = kpss(timeseries, regression='c', nlags="auto")
    kpss_output = pd.Series(kpsstest[0:3], index=['Test Statistic','p-value','Lags Used'])
    for key,value in kpsstest[3].items():
        kpss_output['Critical Value (%s)'%key] = value
    print (kpss_output)

def obtain_adf_kpss_results(timeseries, max_d):
    """ 
    Build dataframe with ADF statistics and p-value for time series after applying difference on time series
    
    Args:
        time_series (df): Dataframe of univariate time series  
        max_d (int): Max value of how many times apply difference
        
    Return:
        Dataframe showing values of ADF statistics and p when applying ADF test after applying d times 
        differencing on a time-series.
    
    """
    results=[]

    for idx in range(max_d):
        adf_result = adfuller(timeseries, autolag='AIC')
        kpss_result = kpss(timeseries, regression='c', nlags="auto")
        timeseries = timeseries.diff().dropna()

        if adf_result[1] <= 0.05:
            adf_stationary = True
        else:
            adf_stationary = False

        if kpss_result[1] <= 0.05:
            kpss_stationary = False
        else:
            kpss_stationary = True
            
        stationary = adf_stationary & kpss_stationary
            
        results.append((idx,adf_result[1], kpss_result[1],adf_stationary,kpss_stationary, stationary))
    
    # Construct DataFrame 
    results_df = pd.DataFrame(results, columns=['d','adf_stats','p-value', 'is_adf_stationary','is_kpss_stationary','is_stationary' ])
    
    return results_df
