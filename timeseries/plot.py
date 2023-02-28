import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import lag_plot

from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose

def plain_plot(df, title = "Sale data through years"):
    df.dropna(inplace= True)
    plt.figure(figsize=(15,4), dpi=120)
    plt.plot(df['Date'], df['Sales_quantity'], color='tab:red')
    plt.gca().set(title=title, xlabel='Date', ylabel='Sales_quantity')
    plt.show()

def acf_pacf_plot(Y):
    fig, axes = plt.subplots(1,2,figsize=(12,3), dpi= 120)
    # Plot autocorrelation gragh
    fig = plot_acf(Y.dropna(), lags=18, ax=axes[0])
    # Plot partial autocorrelation gragh
    fig = plot_pacf(Y.dropna(), lags=18, ax=axes[1],method='ywm')
  
def actual_pred_graph(y_test,y_train,yhat, y_fc):
    if y_test is None:
        pass
    else:
        plt.plot(y_test, color = "red", label = 'Test Set')
    plt.plot(y_train, color = "black",label = 'Train Set')
    plt.ylabel('Quantity')
    plt.xlabel('Date')
    plt.xticks(rotation=45)
    plt.title("Actual and Prediction for Sales Data")
    plt.plot(yhat, color='green', label = 'in-sample Predictions')
    plt.plot(y_fc, color='blue', label = 'out-sample predictions')
    plt.legend()

def decompose_seasonaltrend_graph(y,period):
    if period is None:
        period = 36
    else:
        period = period

    # Multiplicative Decomposition 
    multiplicative_decomposition = seasonal_decompose(y, model='multiplicative', period=period)
    # Additive Decomposition
    additive_decomposition = seasonal_decompose(y, model='additive', period=period)
    # Plot
    plt.rcParams.update({'figure.figsize': (12,10)})
    multiplicative_decomposition.plot().suptitle('Multiplicative Decomposition', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    additive_decomposition.plot().suptitle('Additive Decomposition', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def lag_plots(Y):        
    # Plot
    fig, axes = plt.subplots(3, 4, figsize=(12,6), sharex=True, sharey=True, dpi=110)
    fig.subplots_adjust(hspace=0.5)
    for i, ax in enumerate(axes.flatten()[:12]):
        lag_plot(Y, lag=i+1, ax=ax, c='firebrick')
        ax.set_title('Lag ' + str(i+1))
    fig.suptitle('Lag Plots', y=1.05)    


def autocorrelation_graph(self,nlag):
    if nlag is None:
        self.nlag = 18
    else:
        self.nlag = nlag
    # Plot autocorrelation
    fig, axes = plt.subplots(1,2,figsize=(12,3), dpi= 120)
    axes[0].plot(self.y.values)
    axes[0].set_title('Original Series')
    plot_acf(self.y.values,lags=nlag, ax=axes[1])

def rolling_statistics_graph(df,columnname):
    def moving_average(df):
        #Determine annually rolling statistics (12 months)
        df['movingAverage'+columnname] = df[columnname].dropna().rolling(window=12).mean()
        df['movingSTD'+columnname] = df[columnname].dropna().rolling(window=12).std()
    
    #Determine rolling statistics
    moving_average(df)
    
    #Plot
    plt.plot(df[columnname], color='blue', label='Original')
    plt.plot(df['movingAverage'+columnname], color='red', label='Rolling Mean')
    plt.plot(df['movingSTD'+columnname] , color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation'+columnname)
    plt.show()

def actual_pred_graph(y_test, y_train, yhat, y_fc):

    plt.plot(y_test, color = "red", label = 'Test Set')
    plt.plot(y_train, color = "black",label = 'Train Set')
    plt.ylabel('Quantity')
    plt.xlabel('Date')
    plt.xticks(rotation=45)
    plt.title("Actual and Prediction for Sales Data")
    plt.plot(yhat, color='green', label = ' in-sample Predictions')
    plt.plot(y_fc, color='blue', label =  ' out-sample predictions')
    plt.legend()