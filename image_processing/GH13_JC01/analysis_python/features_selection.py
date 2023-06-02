# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 08:53:40 2023
Features Engineering: Guidance from Hands on Machine Learning Book
Feature Selection for LFW prediction
@author: jcard
"""

import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from pandas.plotting import scatter_matrix

# Data for the 3 harvest dates
raw_data = pd.read_csv('harvest_complete.csv')


# Separate all the data in 2 data frames: One for the response variable / second with all the predictors
response = raw_data.iloc[:,6]
predictors = raw_data.iloc[:,11:24]

# Check if any of our variables is missing data. 
# There are 84 instances for all predictors
predictors.info()

#Describe method: This will generate a summary of each numerical attribute.

predictors.describe()

# To go further in our analysis we could plot a histogram for each numerical variable.
# histogram shows the number of instances (in vertical axis) that have a give value range(horizontal axis)
## REMEMBER: Here we are just showing histograms for the predictors. 
response.hist(bins = 25, figsize = (10,7))
#plt.show()

predictors.hist(bins = 25, figsize = (10,7))
#plt.show()

# Use scikit learn to split our dataset in multiple subsets.
# The simplest function is train_test_split() // random_state argument sets the random generator seed. 
# This means that our test size will be 20% of our data while the train set will be 80% of our data. 
train_set,test_set = train_test_split(all_var, test_size = 0.2, random_state= 42)


## FROM NOW ON WE WILL BE WORKING WITH OUR TRAINING SET (80 % of all our sample)

allvar_train= train_set.copy()

# Taking a look at correlations
corr_matrix =  allvar_train.corr()
allvar_train.info()
# Using corr_matrix, we will check specifically how much each predictor correlates with biomass measurements (LFW,LDW,LA)

corr_LFW = corr_matrix['LFW_g'].sort_values(ascending=False)
corr_LFW
corr_LDW = corr_matrix['LDW_g'].sort_values(ascending=False)
corr_LDW
corr_LA = corr_matrix['LA_g'].sort_values(ascending=False)
corr_LA

# Another way to show correlation between attributes is to use the pandas scatter_matrix() function. It plots every numerical attribute 
# against other numerical attribute. 
## We could do one graph for each 
## attributes = ['median_house_value','median_income','total_rooms','housing_median_age'], and use it next to df inside brackets
scatter_matrix(allvar_train,figsize=(12,8),alpha = 0.2)


# FIT A MODEL FOR LA BASED ON THE LIST OF FEATURES GIVEN

feature_list= predictors.columns.tolist()

ratings = all_var.loc[:,'LA_g']
features = all_var.loc[:,feature_list]
    
    # 
X_train, X_test, y_train, y_test = train_test_split(features, ratings, test_size = 0.2, random_state = 1)
    
    # don't worry too much about these lines, just know that they allow the model to work when
    # we model on just one feature instead of multiple features. Trust us on this one :)
if len(X_train.shape) < 2:
  X_train = np.array(X_train).reshape(-1,1)
  X_test = np.array(X_test).reshape(-1,1)
    
    # 
model = LinearRegression()
model.fit(X_train,y_train)
    
    # 
print('Train Score:', model.score(X_train,y_train))
print('Test Score:', model.score(X_test,y_test))
    
# print the model features and their corresponding coefficients, from most predictive to least predictive
print(sorted(list(zip(feature_list,model.coef_)),key = lambda x: abs(x[1]),reverse=True))
    

y_predicted = model.predict(X_test)
    
plt.scatter(y_test,y_predicted)
plt.xlabel('LA_mm2')
plt.ylabel('Predicted LA_mm2')
plt.ylim(1,5)
plt.show()


