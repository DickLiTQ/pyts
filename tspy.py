# -*- coding: utf-8 -*-

"""

This pyfile is used to make our time series analysis more convenient. Although statsmodels are powerful enough, it can be quite tedious to remember a lot of modules with different name. If we spend so much time on searching which command should we use in statsmodels, we are just doing something wrong against efficiency. So I try to develop this pyfile based on statsmodels to make things easy.

DickLi
2018.3.22

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.graphics as smg
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
plt.style.use("seaborn")


def __init__():

    return 0

def load_txt(path,header):
    file = open(path,'r')
    lines = file.readlines()
    firstline = lines[0]
    variable = firstline.split()
    if header == "T":
        data = pd.DataFrame(columns=variable,index=range(len(lines[1:])))
        for row, line in enumerate(lines[1:]):
            text = line.split()
            for var in range(len(variable)):
                data.iloc[row,var]=text[var]
    else:
        variable = np.arange(len(variable))
        variable = list(variable)
        data = pd.DataFrame(columns=variable,index=range(len(lines)))
        for row, line in enumerate(lines):
            text = line.split()
            for var in range(len(variable)):
                data.iloc[row,var]=text[var]
    return data.apply(pd.to_numeric,errors="ignore")

def to_num(data):
    return data.apply(pd.to_numeric,errors="ignore")

def dateindex_transfer(data,year,month,day):
    import datetime
    data['Date']=datetime.date(2018,1,1)
    for index in range(data.shape[0]):
        date = datetime.date(int(data[year][index]),int(data[month][index]),int(data[day][index]))
        data['Date'][index] = date
    data.set_index('Date',inplace=True)
    return data

def acf(data,lag):
    fig = plt.figure()
    fig = sm.graphics.tsa.plot_acf(data,lags=lag)
    fig.show()
    return fig

def pacf(data,lag):
    fig = plt.figure()
    fig = sm.graphics.tsa.plot_pacf(data,lags=lag)
    fig.show()

def acfpacf(data,lag):
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    fig = sm.graphics.tsa.plot_acf(data,lags=lag,ax=ax1)
    ax2 = fig.add_subplot(212)
    fig = sm.graphics.tsa.plot_pacf(data,lags=lag,ax=ax2)
    fig.show()

def select_ARMA_order(data,max_ar,max_ma,ic):
    return sm.tsa.arma_order_select_ic(data,max_ar=max_ar,max_ma=max_ma,ic=ic)
    
def ARMA(data,p,q):
    return sm.tsa.ARMA(data,(p,q)).fit()

def ADFuller(data,ic,ct):
    t = adfuller(data,autolag=ic,regression=ct)
    return {'ADF_statistics':t[0],
    'p_value':t[1],
    'lags':t[2],
    'observation':t[3],
    'critical_values':t[4],
    'icbest':t[5]}

def LjungBox(residual,lag):
    t = acorr_ljungbox(residual,lags=lag)
    acfpacf(residual,lag)
    return {'LB_statistics': t[0],'p_value': t[1]}

def LjungBox_ARMA(data,p,q,lag):
    model = ARMA(data,p,q)
    residual = model.fittedvalues - data
    t = acorr_ljungbox(residual,lags=lag)
    acfpacf(residual,lag)
    return {'LB_statistics': t[0],'p_value': t[1]}

def backtest(data,p,q,startpoint,IC):
    MSFE = 0
    predict = 0
    for i in range(1,len(data)-startpoint):
        predict = ARMA(data[i:],p,q).forecast(1)[0]
        MSFE = MSFE + (predict-ARMA(data[i:],1,0).fittedvalues[-1])**2
        data[startpoint+i] = predict
    return MSFE/(len(data)-startpoint)

path = 'C:/Users/DickLi/OneDrive/文档/2017-2018大三下学期/时间序列分析/data/m-unrate-4811.txt'
data = load_txt(path,"T")
#data = to_num(data)
data.columns = ['data']
index = pd.date_range('1/1/1948',periods=766,freq='M')
data.index = index
model = ARMA(data['data'],1,3)
model.summary2()
model.forecast(10)[0]

ADFuller(data['data'],'AIC','c')
    
