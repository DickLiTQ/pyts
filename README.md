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
    if header == "T":
        firstline = lines[0]
        variable = firstline.split()
        data = pd.DataFrame(columns=variable,index=range(len(lines[1:])))
    else:
        data = pd.DataFrame(index=range(len(lines)))
    for row, line in enumerate(lines[1:]):
        text = line.split()
        for var in range(len(variable)):
            data.iloc[row,var]=text[var]
    return data

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



