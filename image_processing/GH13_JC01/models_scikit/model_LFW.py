# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 18:55:06 2023
Fitting our first multiple linear regression using scikit image and pandas to create datasets. 
Objective: Generate statistical performance description for a multiple linear regression and a random forest model for one response variable.
RESPONSE VARIABLE: Leaf Fresh Weight.  
@author: jcard
"""

import pandas as pd
import numpy as np 
import os
import math
import matplotlib.pyplot as plt

raw_data = pd.read_csv("C:/Users/jcard/OneDrive - University of Georgia/kinect_imaging/GH13_JC01/GH13_JC01_R/harvest_complete.csv")

# Use function info to display details about our dataframe. 
# Columns: 6,7,8 are our response variables. 
# Notice that the information is complete in all of our columns (a value for each of the 165 plants)
raw_data.info()

# Keep columns that contains only predictors (image-derived predictors) and response variable. 
dataset = raw_data.iloc[:,[6] + list(range(11,raw_data.shape[1]))]
dataset.info()

# Add additional predictors obtained by doing calculations using the other variables such as: 
## compactness_idx= 4*(plant_area/Perimeter^2)
dataset['compactness_idx'] = 4*math.pi*(dataset['plant_area']/(dataset['plant_perimeter']**2))

# Let's create a summary table for each numerical predictor
summary_num = dataset.describe()

# Now lets plot a histogram for each numerical variable. Number of plants in vertical axis that have a given value. 

dataset.hist(bins = 40, figsize = (20,15))
plt.show()

# We are having an issue here since we dont have a bell shape distribution of our variables. 
# ATTENTION: We need to solve this issue. The problem  is that we are using information from all harvests all togethe. 

# FROM HERE: We start using sklearn library 
# We are generating here one data set with a proportion of all data and a test set with an smaller proportion

from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(dataset, test_size = 0.2, random_state =42)

# Let's generate a copy of our training data set
copytrain = train_set.copy()

# Looking for correlations
# We will compute the standard correlation coefficient (Pearson's r)  between every pair of predictors using:
# corr() method:

corr_matrix  =copytrain.corr()

# Now we want to check specifically how much each variable correlates with the LEAF FRESH WEIGHT:
corr_matrix['LFW_g'].sort_values(ascending=False)

# Another way to show correlation between variables is to use the pandas scatter_matrix() function. It plots every numerical attribute 
# against other numerical attribute. 
# Here we will focus on just the few most promisin attributes that seem most correlated with the LEAF FRESH WEIGHT.


from pandas.plotting import scatter_matrix

# HERE WE WILL SELECT THE MOST PROMISING PREDICTORS
variables = ['LFW_g','plant_ellipse_minor_axis','height_mm','plant_area','plant_height']
scatter_matrix(copytrain[variables],figsize=(12,8),alpha = 0.5)

# We can zoom in in the most promising varaible that is plant_ellipse_minor_axis and see its correlation plot:

copytrain.plot(kind = 'scatter',x ="plant_ellipse_minor_axis", y = "LFW_g", alpha = 0.5)

# Let's separate predictors and the labels. 
features = copytrain.drop("LFW_g", axis = 1) # predictors
labels = copytrain["LFW_g"]


# Feature Scaling:  Algorithms do not perform well when numerical attributes have different scales. 
# We have two alternatives: 1) min- max 2) Standardization
from sklearn.preprocessing import StandardScaler
# from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# Just features on a similar scale. We create a new dataframe dropping the predictor variable.

features_sc = scaler.fit_transform(features)

# TRAINING A LINEAR MODEL: 
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(features_sc, labels)

# trying on a few new instances
some_data = features.iloc[:5]
some_data_sc = scaler.fit_transform(some_data)
some_labels = labels.iloc[:5]

print("Predictions:", lin_reg.predict(some_data_sc))
print("Labels:", list(some_labels))

## Evaluating the model on the training set:
## OBTAINING REGRESSION MODEL RMSE
from sklearn.metrics import mean_squared_error
linear_predictions = lin_reg.predict(features_sc)
lin_mse = mean_squared_error(labels,linear_predictions)
lin_rmse = np.sqrt(lin_mse)
print("We obtain a typical prediction error (RMSE) of:", lin_rmse)

# Probably the model  is not powerfull enough. Let's try a more powerful model: Decision Tree Regresor, capable of finding complex non linear relationships
from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor()
tree_reg.fit(features_sc, labels)

# Once again we evaluate it on the training set: 
tree_predictions = tree_reg.predict(features_sc)
tree_mse = mean_squared_error(labels,tree_predictions)
tree_rmse = np.sqrt(tree_mse)
print("Using a decision tree model, we obtain a typical prediction error (RMSE) of:", tree_rmse)

