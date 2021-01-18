#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.utils import shuffle
from sklearn import metrics


# In[2]:


data_set = pd.read_csv("http://bit.ly/w-data")


# In[6]:


print(data_set)


# In[7]:


null_values = data_set.isnull().sum()
print(null_values)


# In[5]:


data_set = shuffle(data_set)


# In[8]:


print(data_set)


# In[9]:


data_set.plot(x='Hours', y='Scores', style='o')  
plt.title('Plot of No of hours studied VS Percentage Scored')  
plt.xlabel('No of hours studied')  
plt.ylabel('Percentage Score')  
plt.show()


# In[11]:


x = data_set.iloc[:, :-1].values
y = data_set.iloc[:, -1].values


# In[12]:


from sklearn.model_selection import train_test_split  
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)


# In[13]:


regressor = LinearRegression()  
regressor.fit(x_train, y_train)


# In[14]:


line = (regressor.coef_*x)+(regressor.intercept_)
plt.scatter(x, y, c="red")
plt.plot(x, line, c="black");
plt.xlabel('No of hours studied')  
plt.ylabel('Percentage Score')
plt.title('Scatter plot of No of hours studied VS Percentage Scored') 
plt.show()


# In[15]:


print(x_test) # Testing data - In Hours
y_pred = regressor.predict(x_test) # Predicting the scores


# In[16]:


# Comparing Actual Outcome vs Predicted Outcome
df = pd.DataFrame({'Actual Outcome': y_test, 'Predicted Outcome': y_pred})  
print(df)


# In[20]:


hour=9.25
pred = regressor.predict([[hour]])
print(pred)
print(f"No of Hours the student studies in a day = {hour}")
print(f"Predicted Percentage of the student = {pred[0]}")


# In[21]:


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred)) 
print('R square score:', regressor.score(x_test, y_test))


# In[ ]:




