
# coding: utf-8

# In[18]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from sklearn import linear_model
df= pd.read_csv(r"C:\Users\user1\Desktop\Machine learning\hour.csv")


# In[19]:


X = df.drop('cnt', axis=1)
y = df['cnt']


# In[20]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)


# In[22]:


from sklearn.svm import SVC
svclassifier = SVC(kernel='linear')
svclassifier.fit(X_train, y_train)


# In[23]:


y_pred = svclassifier.predict(X_test)


# In[24]:


from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))

