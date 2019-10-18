# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 15:16:36 2019

@author: user1
"""

#import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
dataset= pd.read_csv('hour.csv')
#dataset.plot()  # plots all columns against index
dataset.plot(kind='scatter', y = 'cnt',x='temp',)
#dataset.set(xlim=(xmin, xmax), ylim=(ymin, ymax))
sns.regplot(dataset['temp'],dataset['cnt'])
plt.xlabel('Temperature')
plt.ylabel('Usage Count')
plt.title('Scatter plot - Usage Count/temperature',fontsize=16)

dataset.plot(kind='scatter', y = 'cnt',x='atemp',) # scatter plot)
sns.regplot(dataset['atemp'],dataset['cnt'])
plt.xlabel('Adjusted Temperature')
plt.ylabel('Usage Count')
plt.title('Scatter plot - Usage Count/Adjusted temperature',fontsize=16)

dataset.plot(kind='scatter', y = 'cnt',x='weathersit',) # scatter plot
sns.regplot(dataset['weathersit'],dataset['cnt'])
plt.xlabel('Weather Situation')
plt.ylabel('Usage Count')
plt.title('Scatter plot -Usage Count/Weather Situation',fontsize=16)

dataset.plot(kind='scatter', y = 'cnt',x='windspeed',) # scatter plot
sns.regplot(dataset['windspeed'],dataset['cnt'])
plt.xlabel('Wind Speed')
plt.ylabel('Usage Count')
plt.title('Scatter plot - Usage Count/Windspeed',fontsize=16)

dataset.plot(kind='scatter', y = 'cnt',x='casual',) # scatter plot
sns.regplot(dataset['casual'],dataset['cnt'])
plt.title('Scatter plot - Usage Count/Casual Users',fontsize=16)

dataset.plot(kind='scatter', y = 'cnt',x='registered',) # scatter plot
sns.regplot(dataset['registered'],dataset['cnt'])
plt.title('Scatter plot -Usage Count/registered Users',fontsize=16)

dataset.plot(kind='scatter', y = 'cnt',x='workingday',) # scatter plot
sns.regplot(dataset['workingday'],dataset['cnt'])
plt.title('Scatter plot -Usage Count/Work day',fontsize=16)

dataset.plot(kind='scatter', y = 'cnt',x='holiday',) # scatter plot
sns.regplot(dataset['holiday'],dataset['cnt'])
plt.title('Scatter plot -Usage Count/holiday',fontsize=16)

dataset.plot(kind='scatter', y = 'cnt',x='yr',) # scatter plot
sns.regplot(dataset['yr'],dataset['cnt'])
plt.title('Scatter plot -Usage Count/Year',fontsize=16)
dataset.plot(kind='scatter', y = 'cnt',x='hum',) # scatter plot
sns.regplot(dataset['hum'],dataset['cnt'])
plt.title('Scatter plot -Usage Count/Humidity',fontsize=16)