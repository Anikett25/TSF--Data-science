#!/usr/bin/env python
# coding: utf-8

# ## Task 1 - Prediction using supervised ML

# In[1]:


# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# Reading data from remote link
data = "http://bit.ly/w-data"
s_table = pd.read_csv(data)

s_table.head()


# In[3]:


# Plotting the distribution of scores
s_table.plot(x='Hours', y='Scores', style='o')  
plt.title('Hours vs Marks')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.show()


# **There is a positive relation between number of hours studied and percentage scored**

# In[4]:


#Preparing the data
X = s_table.iloc[:, :-1].values
y = s_table.iloc[:, 1].values


# In[5]:


#Split data into training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[6]:


#To fit the regression line
regressor = LinearRegression()
regressor.fit(X_train, y_train)


# In[7]:


# Plotting the regression line
line = regressor.coef_*X+regressor.intercept_
plt.scatter(X, y)
plt.plot(X, line)
plt.show()


# In[8]:


print(X_test)
y_pred = regressor.predict(X_test)


# In[9]:


# Comparing Actual data and predicted data
new_tab = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
new_tab


# In[21]:


#plot the actual vs predicted
plt.scatter(x = X_test, y = y_test, color='blue')
plt.plot(X_test, y_pred, color='black')
plt.title('Actual vs Predicted')
plt.xlabel('Hours')
plt.ylabel('Percentage')
plt.show()


# In[22]:


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))


# **To predict the percentage if a student studies for 9.25 hrs/day**

# In[23]:


hrs = [9.25]
s_pred = regressor.predict([hrs])
print("No of Hours = {}".format(hours))
print("Predicted Score = {}".format(round(s_pred[0], 3)))

