
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, median_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
get_ipython().run_line_magic('matplotlib', 'inline')
df= pd.read_csv(r"C:\Users\user1\Desktop\Machine learning\hour.csv")
#for d in [df]: # loop to apply the change to both dfs
   # d["date"] = pd.to_datetime(d['date'])
    #print("Column changed to: ", d.date.dtype.name)


# In[2]:


X= df.iloc [: , : -1]. values
y= df.iloc [: , 14].values


# In[3]:


def assign_label(hour):
    if hour > 6 and hour <= 12:
        return 1
    elif hour > 12 and hour <= 18:
        return 2
    elif hour > 18 and hour <= 24:
        return 3
    else:
        return 4


# In[4]:


df['time_label'] = df['hr'].apply(assign_label)


# In[5]:


columns = df.columns.drop(['casual','registered','cnt'])
columns


# In[6]:


import math

#Sample 80% of the data randomly and assigns it to train.
eighty_percent_values = math.floor(df.shape[0]*0.8)
train = df.sample(n=eighty_percent_values, random_state = 1)

#Selects the remaining 20% to test.
test = df.drop(train.index)
train.shape[0] + test.shape[0] == df.shape[0]


# In[7]:


lr = LinearRegression()
lr.fit(train[columns], train['cnt'])
predictions_test = lr.predict(test[columns])
mse_test = mean_squared_error(test['cnt'], predictions_test)
mse_test


# In[8]:


predictions_train = lr.predict(train[columns])
#mean squared error
mse_train = mean_squared_error(train['cnt'], predictions_train)
mse_train


# In[9]:


lr = LinearRegression()
lr.fit(train[columns], train['cnt'])
lm_r2 = r2_score(test['cnt'], predictions_test)
lm_r2


# In[10]:


lm_r2 = r2_score(train['cnt'], predictions_train)
lm_r2


# In[11]:


#mean absolute error train
lm_mean_ae = mean_absolute_error(train['cnt'], predictions_train)
lm_mean_ae


# In[12]:


#mean absolute error test
lm_mean_ae = mean_absolute_error(test['cnt'], predictions_test)
lm_mean_ae


# In[13]:


# Median Absolute Error train
lm_median_ae = median_absolute_error(train['cnt'], predictions_train)
lm_median_ae


# In[14]:


# Median Absolute Error test
lm_median_ae = median_absolute_error(test['cnt'], predictions_test)
lm_median_ae


# In[15]:


tree = DecisionTreeRegressor(min_samples_leaf=5)
tree.fit(train[columns], train['cnt'])
predictions = tree.predict(test[columns])
mse = mean_squared_error(test['cnt'], predictions)
mse


# In[16]:


#mean absolute error test
tree = DecisionTreeRegressor(min_samples_leaf=5)
tree.fit(train[columns], train['cnt'])
predictions = tree.predict(test[columns])
dt_mean_ae = mean_absolute_error(test['cnt'], predictions)
dt_mean_ae


# In[17]:


tree = DecisionTreeRegressor(min_samples_leaf=5)
tree.fit(train[columns], train['cnt'])
predictions = tree.predict(test[columns])
# Median Absolute Error test
dt_median_ae = median_absolute_error(test['cnt'], predictions)
dt_median_ae


# In[18]:


#R2 Score
tree = DecisionTreeRegressor(min_samples_leaf=5)
tree.fit(train[columns], train['cnt'])
predictions = tree.predict(test[columns])
dt_r2 = r2_score(test['cnt'], predictions)
dt_r2


# In[19]:


tree = RandomForestRegressor(min_samples_leaf=2, n_estimators=250)
tree.fit(train[columns], train['cnt'])
predictions = tree.predict(test[columns])
mse = mean_squared_error(test['cnt'], predictions)
mse


# In[20]:


#mean absolute error
rf_mean_ae = mean_absolute_error(test['cnt'], predictions)
rf_mean_ae


# In[21]:


# Median Absolute Error test
rf_median_ae = median_absolute_error(test['cnt'], predictions)
rf_median_ae


# In[22]:


tree = RandomForestRegressor(min_samples_leaf=2, n_estimators=250)
tree.fit(train[columns], train['cnt'])
predictions = tree.predict(test[columns])
rf_r2 = r2_score(test['cnt'], predictions)
rf_r2


# In[23]:


mse_leaf=[]
for i in range(1, 10):
    tree = RandomForestRegressor(min_samples_leaf=i, n_estimators= 250)
    tree.fit(train[columns], train['cnt'])
    predictions = tree.predict(test[columns])
    mse = mean_squared_error(test['cnt'], predictions)
    mse_leaf.append(mse)
mse_leaf


# In[24]:


n_trees = [250, 500, 750]
mse_trees=[]
for i in n_trees:
    tree = RandomForestRegressor(min_samples_leaf=1, n_estimators=i)
    tree.fit(train[columns], train['cnt'])
    predictions = tree.predict(test[columns])
    mse = mean_squared_error(test['cnt'], predictions)
    mse_trees.append(mse)
mse_trees


# In[33]:


from scipy import stats
#import numpy as np
z = np.abs(stats.zscore(X))
print (z)


# In[34]:


threshold = 3
print(np.where(z > 3))


# In[27]:


print(z[17341][11])


# In[42]:


#Removing the outliers
c= X[(z < 3).all(axis=1)]


# In[43]:


print(columns)

