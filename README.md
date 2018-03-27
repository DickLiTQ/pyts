# pyts
_Python on Time Series Analysis_

_使用Python来进行时间序列分析_

--------------------------
### Introduction/简介
This pyfile is used to make our time series analysis more convenient. Although statsmodels are powerful enough, it can be quite tedious to remember a lot of modules with different name. If we spend so much time on searching which command should we use in statsmodels, we are just doing something wrong against efficiency. So I try to develop this pyfile based on statsmodels to make things easy.

So this module may like quantmod :)

这个Project将简化我们使用Python进行时间序列分析。尽管我们已经有statsmodels这个强大的统计分析工具，但是要记住不同模块和子模块的不同命令是一个相对麻烦的事情。如果我们花费过多的时间在搜索到底该用statsmodels的哪个部分的命令来完成分析，那么这是非常不具有效率的。因此，我尝试去完成这个**基于statsmodels**的Project来简化我们的使用。

因此这个模块会更像R语言中的quantmod :)

**English and Chinese are both supported in this readme file.**

**这篇readme文件包含英语和中文两种语言。**

-------------------------
### Instruction of Functions/函数说明

#### Import necessary modules/导入所需要的模块
```Python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.graphics as smg
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
plt.style.use("seaborn") # You may change what you like
```
#### txt file loading/txt数据读取
Sometimes data is listed in a txt file and has the following form:

部分数据使用txt记录，并且拥有以下的格式:
```
year mon day gnp
1947 01 01   1780.4
1947 04 01   1778.1
1947 07 01   1776.6
1947 10 01   1804.0
1948 01 01   1833.4
1948 04 01   1867.6
1948 07 01   1877.6
1948 10 01   1880.5
1949 01 01   1854.0
1949 04 01   1847.0
1949 07 01   1867.2
1949 10 01   1848.8
1950 01 01   1923.8
```
To develop a dataframe from such data, we give a loading function below:

针对这样的数据或类似的数据，我们给出以下的加载函数：
```Python
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
```
When the first row in txt file is used to indicate variables, we make _header_ equals "T" and then receive a dataframe with column names.

当第一行是标题（变量）时，我们取header为"T"，可以得到一个包含这些数据内容的dataframe。

#### Date index / 时间指标
In time series analysis, index of date is quite important. From the txt file, we will obtain the datetime as columns named "year", "mon" and "day". A function is designed to get such datetimes and transfer them into a standard form of index.

时间序列分析中，数据的时间指标非常关键，使用从txt读取的数据可能会得到以上year、mon、day形式记录的日期，我们使用下列函数从dataframe中抽取year、month、day并建立time index到dataframe中。
```Python
def dateindex_transfer(data,year,month,day):
    import datetime
    data['Date']=datetime.date(2018,1,1)
    for index in range(data.shape[0]):
        date = datetime.date(int(data[year][index]),int(data[month][index]),int(data[day][index]))
        data['Date'][index] = date
    data.set_index('Date',inplace=True)
    return data
```

#### Plot of ACF and PACF / ACF与PACF绘制
Autoregressive Function (ACF) and Partial Autoregressive Function (PACF) is a powerful tool for us to analyze the correlation between different states. So to plot a clear graph is important. We define three functions to make it easy.

Autoregressive Function (ACF)和Partial Autoregressive Function (PACF)是我们分析时间序列数据相关性的重要手段，其图像的绘制非常关键，我们定义三个函数来简化绘图操作。
```Python
def acf(data,lag):
    fig = plt.figure()
    fig = sm.graphics.tsa.plot_acf(data,lags=lag)
    fig.show()

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
```
*acf()* function returns a graph of ACF, and similarly, *pacf()* for PACF and *acfpacf()* gives both ACF and PACF. For example, if we need a ACF and PACF plot of data with 40 lags, we can use the coding beneath.

*acf()* 返回ACF图，*pacf()* 函数返回PACF图，*acfpacf()* 返回包含ACF和PACF的图。例如我们要绘制某个data的40期lag的ACF和PACF图，则使用以下代码：
```Python
acfpacf(data,40)
```
The result is:


结果为：
![](https://raw.githubusercontent.com/DickLiTQ/pyts/master/acfpacf.png)

#### Test stationary / 检验数据稳定性
Before we establish an ARMA(p,q) model, we are suppose to test whether the data is weak stationary. We examine it by Advanced Dickey Fuller (ADF) test.

在我们建立一个ARMA(p,q)模型时，我们应当先检验数据是否符合弱平稳性。我们使用ADF检验来检验。
```python
def ADFuller(data,ic,ct):
    t = adfuller(data,autolag=ic,regression=ct)
    return {'ADF_statistics':t[0],
    'p_value':t[1],
    'lags':t[2],
    'observation':t[3],
    'critical_values':t[4],
    'icbest':t[5]}
```
For example, we examine a column named "data" in our dataset, using AIC as the choice of lag and use the constant in c:

例如我们检验data数据集中的某个名为data的列，使用AIC最小化作为选取lag的标准，考虑常数c：
```python
ADFuller(data['data'],'AIC','c')
```
The output is

输出结果是
```
{'ADF_statistics': -2.6315992772570356,
 'critical_values': {'1%': -3.4390641198617864,
  '10%': -2.5688179819544312,
  '5%': -2.8653859408474482},
 'icbest': -348.21261544613162,
 'lags': 12,
 'observation': 753,
 'p_value': 0.086640606585698832}
```

#### Choose order (p,q) on AIC and BIC / 根据AIC、BIC选取ARMA(p,q)阶数
We calculate AIC or BIC under different ARMA(p,q) models to find order (p,q) when AIC or BIC take local minimum value. The estimation is based on Maximum Likelihood Estimation Method so in some case it cannot converge.

原理是利用不同ARMA(p,q)下的AIC和BIC值来决定最优的p、q对，基于Maximum Likelihood Estimation，因此可能会得出不收敛的结果。

```Python
def select_ARMA_order(data,max_ar,max_ma,ic):
    return sm.tsa.arma_order_select_ic(data,max_ar=max_ar,max_ma=max_ma,ic=ic)
```
For instance,

例如
```Python
select_ARMA_order(data,3,3,'aic')
```
The output is

结果为
```
{'aic':             0           1           2           3
   0           725.521096  727.368619  727.420572  729.002785
   1           727.325969  728.806353  729.167261  730.875017
   2           727.409239  729.407495  731.071447         NaN
   3           729.405918         NaN  729.042475  719.520655,
 'aic_min_order': (3, 3)                                      }
```

#### Build an ARMA(p,q) / 建立ARMA(p,q)模型
According to our data and order (p,q), we can fit a ARMA(p,q) model.

根据data和所得(p,q)获得ARMA(p,q)模型。
```Python
def ARMA(data,p,q):
    return sm.tsa.ARMA(data,(p,q)).fit()
```
For example,

例如，
```Python
model = ARMA(data,3,3)
model.summary2()
```
The output is 

可以得到模型的信息
```
Results: ARMA
====================================================================
Model:              ARMA             BIC:                 -219.5541 
Dependent Variable: data             Log-Likelihood:      129.70    
Date:               2018-03-22 20:30 Scale:               1.0000    
No. Observations:   766              Method:              css-mle   
Df Model:           5                Sample:              01-31-1948
Df Residuals:       761                                   10-31-2011
Converged:          1.0000           S.D. of innovations: 0.204     
AIC:                -247.4012        HQIC:                -236.682  
----------------------------------------------------------------------
               Coef.    Std.Err.      t       P>|t|     [0.025   0.975]
----------------------------------------------------------------------
const         5.8372     0.7056     8.2729   0.0000    4.4543   7.2201
ar.L1.data    0.9867     0.0060   164.8929   0.0000    0.9749   0.9984
ma.L1.data    0.0356     0.0374     0.9517   0.3416   -0.0377   0.1090
ma.L2.data    0.2046     0.0328     6.2346   0.0000    0.1403   0.2690
ma.L3.data    0.1558     0.0367     4.2447   0.0000    0.0839   0.2278
-----------------------------------------------------------------------------
                Real           Imaginary          Modulus          Frequency
-----------------------------------------------------------------------------
AR.1            1.0135             0.0000           1.0135             0.0000
MA.1            0.5256            -1.5614           1.6474            -0.1983
MA.2            0.5256             1.5614           1.6474             0.1983
MA.3           -2.3643            -0.0000           2.3643            -0.5000
====================================================================
```
10-steps forecast:

进行10期预测：
```Python
model.forecast(10)[0]
```
The result is

得到
```
array([ 8.63304374,  8.55667961,  8.49778608,  8.46231854,  8.4273238 ,
        8.39279556,  8.35872761,  8.3251138 ,  8.29194808,  8.25922447])
```

#### LjungBox Test / 进行LjungBox检验
Usually we use LjungBox to test the correlation of time series residual.

常使用LjungBox来检验时间序列模型残差相关性。
```python
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
```
We are required to calculate the residual manually while using *LjungBox()*. If our model is ARMA(p,q), we can use *LjungBox_ARMA()* for convenience.

如果使用*LjungBox()*，我们需要手动计算残差。如果我们的模型是ARMA(p,q)模型，则可以使用*LjungBox_ARMA()*。

#### Backtest / 进行Backtest
backtest is an important method in out-sample test.

backtest是out-sample test中的重要手段，是选取模型的标准之一。我们定义backtest函数：

```python
def backtest(data,p,q,startpoint,IC):
    MSFE = 0
#    predict = 0
    startpoint = startpoint - 1 # i th elements has an index of i-1
    test = np.zeros_like(data)
    train = np.zeros_like(data)
    test[startpoint:] = data[startpoint:]
    train[:startpoint] = data[:startpoint]
    for i in range(len(data)-startpoint):
        predict = ARMA(train[i:startpoint+i],p,q).forecast(1)[0]
        MSFE = MSFE + (predict-test[startpoint+i])**2
        train[startpoint+i] = predict
    return MSFE/(len(data)-startpoint)
```

