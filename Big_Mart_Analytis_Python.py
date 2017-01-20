# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 15:14:32 2017

@author: Parag.Mehta
"""

import pandas as pd
import numpy as np

# Load dataset
train = pd.read_csv("D:\\Project\\parag Personal\\aegis\\Kaggle\\BIG_MART\\Train.csv")
test = pd.read_csv("D:\\Project\\parag Personal\\aegis\\Kaggle\\BIG_MART\\Test.csv")

# describe dataset
train.head()
test.head()
train.describe()
test.describe()

# check for missing values
pd.isnull(train)
pd.isnull(test)
train.isnull().values.any()
test.isnull().values.any()

#Total NA values
train.isnull().sum().sum()
test.isnull().sum().sum()


#Check which columns have  null
test.isnull().any()
train.isnull().any()

train.isnull().sum()
test.isnull().sum()

# Combine Datasets
train['source'] = 'Train'
test['source'] = 'Test'
combined = pd.concat([train,test], ignore_index = True)
train.shape
test.shape
combined.shape

# Check NA values in combined
combined.isnull().sum()

# Unique values in every dolumn
combined.apply(lambda x: len(x.unique()))
categorical_columns = [x for x in combined.dtypes.index if combined.dtypes[x] == "object"]

# Getting values of columns ignoring id and source
categorical_columns = [x for x in categorical_columns if x not in ['Item_Identifier','source']]

# print frequency of categories
for col in categorical_columns:
    print ('\nFrequency of Categories for varible %s' %col)
    print (combined[col].value_counts())
    

# Impute missing values of weight by average weight.
avg_weight = combined.pivot_table(values = 'Item_Weight', index = 'Item_Identifier')
# Row number of item weights column whcih has null values
miss_na = combined['Item_Weight'].isnull()
#original missing values
miss_na.sum()

combined.loc[miss_na, 'Item_Weight'] = combined.loc[miss_na,'Item_Identifier'].apply(lambda x: avg_weight[x])

# Impute missing values of outlet_size by mode
from scipy.stats import mode
# General mode of all outlet size irrespective of outlet type
combined['Outlet_Size'].mode()

# Mode per outlet type
miss_na1 = combined['Outlet_Size'].isnull()
combined.loc[miss_na1, 'Outlet_Size'] = combined.loc[miss_na1,'Outlet_Type' ].apply(lambda x: mode(x))

print (sum(combined['Outlet_Size'].isnull()))

# Visibility has '0' as values whcih cannot be true. Replacing all 0's in visibility by mean of that
visibility_avg =  combined.pivot_table(values = 'Item_Visibility', index = 'Item_Identifier')
miss_na2 = combined['Item_Visibility'] == 0
sum(miss_na2)
combined.loc[miss_na2, 'Item_Visibility'] = combined.loc[miss_na2, 'Item_Identifier'].apply(lambda x : visibility_avg[x])

# creating a new column which will show individual products mean visiblity across stores.
combined['visibility_average'] = combined.apply(lambda x: x['Item_Visibility']/visibility_avg[x['Item_Identifier']], 
                                    axis=1)
combined['visibility_average'].describe()

#
combined['Item_Type'].unique()
combined['Item_Type_combined'] = combined['Item_Identifier'].apply(lambda x: x[0:2])

combined['Item_Type_combined'] = combined['Item_Type_combined'].map({'FD':'Food','NC':'Non-Consumable','DR':'Drinks'})
combined['Item_Type_combined'].value_counts()

# create a new column to determine years of operation of a store
combined['year_operation'] = 2013 - combined['Outlet_Establishment_Year']
combined['year_operation'].describe()

# Assignning new levels to item fat content & regularising levels
combined["Item_Fat_Content"].value_counts()
combined['Item_Fat_Content'] = combined['Item_Fat_Content'].replace({'LF':'Low Fat','low fat':'Low Fat','reg':'Regular'})

# Seperating Fat content from non consumables
combined.loc[combined["Item_Type_combined"] == "Non-Consumable", 'Item_Fat_Content'] = "Non-Edible"

# Creating new column outlet which is same as outlet_identifier
combined['outlet'] = combined['Outlet_Identifier']

# One hot encoding of few columns
item_fat_dummies = pd.get_dummies(combined['Item_Fat_Content'],prefix='Item_Fat_Content')
combined = pd.concat([combined,item_fat_dummies],axis=1)
combined.drop('Item_Fat_Content',axis=1,inplace=True)

Outlet_Location_Type_dummies = pd.get_dummies(combined['Outlet_Location_Type'],prefix='Outlet_Location_Type')
combined = pd.concat([combined,Outlet_Location_Type_dummies],axis=1)
combined.drop('Outlet_Location_Type',axis=1,inplace=True)

Outlet_Type_dummies = pd.get_dummies(combined['Outlet_Type'],prefix='Outlet_Type')
combined = pd.concat([combined,Outlet_Type_dummies],axis=1)
combined.drop('Outlet_Type',axis=1,inplace=True)

Item_Type_Combined_dummies = pd.get_dummies(combined['Item_Type_combined'],prefix='Item_Type_combined')
combined = pd.concat([combined,Item_Type_Combined_dummies],axis=1)
combined.drop('Item_Type_combined',axis=1,inplace=True)

combined['Outlet_Size'] = combined['Outlet_Size']
Outlet_Size_dummies = pd.get_dummies(combined['Outlet_Size'],prefix='Outlet_Size')
combined = pd.concat([combined,Outlet_Size_dummies],axis=1)
combined.drop('Outlet_Size',axis=1,inplace=True)


combined['outlet'] = combined['outlet']
Outlet_Size_dummies = pd.get_dummies(combined['outlet'],prefix='outlet')
combined = pd.concat([combined,Outlet_Size_dummies],axis=1)
combined.drop('outlet',axis=1,inplace=True)

combined.dtypes


# Seperate train & test dataset
new_train = combined[combined['source'] == "Train" ]
new_test = combined.loc[combined['source'] =="Test" ]

# removing unwanted columns
new_train.drop(['source'], axis =1, inplace = True)
new_test.drop(['source', 'Outlet_Identifier'], axis =1, inplace = True)
new_test.drop(['Item_Outlet_Sales', 'Outlet_Identifier'], axis =1, inplace = True)
new_test.drop(['Outlet_Identifier'], axis =1, inplace = True)
new_train.drop(['Outlet_Identifier'], axis =1, inplace = True)
new_test.drop(['Item_Type'], axis =1, inplace = True)
new_train.drop(['Item_Type'], axis =1, inplace = True)
new_train.drop(['Outlet_Establishment_Year'], axis =1, inplace = True)
new_test.drop(['Outlet_Establishment_Year'], axis =1, inplace = True)
new_train.drop(['Item_Identifier'], axis =1, inplace = True)
new_test.drop(['Item_Identifier'], axis =1, inplace = True)

new_test.dtypes
new_train.dtypes

#Export files as modified versions:
new_train.to_csv("D:\\Project\\parag Personal\\aegis\\Kaggle\\BIG_MART\\Python\\train_modified.csv",index=False)
new_test.to_csv("D:\\Project\\parag Personal\\aegis\\Kaggle\\BIG_MART\\Python\\test_modified.csv",index=False)

# Modelling

# convert data to numpy array for random forest to accept it and seperating target variable

cols = list(new_train.columns.values) # predictors
del cols[1] # removing Item_Outlet_Sales column as its target
target = ['Item_Outlet_Sales'] # target values

new_train_arr = new_train.as_matrix(cols) # convert train to numpy array
targets_arr = new_train.as_matrix(target) # convert test to numpy array

# Building Random forest regression model
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=200,max_depth=5, min_samples_leaf=100,n_jobs=4)
rf.fit(new_train_arr, targets_arr)


# Predict training set
train_results = rf.predict(new_train_arr)

# MSE sqr error before CV
from sklearn.metrics import mean_squared_error
np.sqrt(mean_squared_error(train_results, targets_arr))

#new_train['predictions'] = results

# feature importance
for cols, imp in zip(cols, rf.feature_importances_):
    print(cols, imp)

#Simple K-Fold cross validation. 10 folds.
from sklearn import cross_validation,metrics

# Scoring MSE
cross_val = cross_validation.cross_val_score(rf, new_train_arr, targets_arr, cv=20, scoring='mean_squared_error')
score_cv = np.sqrt(np.abs(cross_val))

# MSE for all CB- average
print (np.sqrt(metrics.mean_squared_error(targets_arr, train_results)))

# CV scores, mean, std, min & max
# STD states the modes does not varry a lot with changing data sets
print ((np.mean(score_cv),np.std(score_cv),np.min(score_cv),np.max(score_cv)))

# Making test set into np array
new_test_arr = new_test.as_matrix(cols)

# predicting test set
test_predict = rf.predict(new_test_arr)

# creating output file
new_train = combined[combined['source'] == "Train" ]
new_test = combined.loc[combined['source'] =="Test" ]
output = pd.concat([new_test['Item_Identifier'], new_test['Outlet_Identifier']], axis=1, keys=['Item_Identifier', 'Outlet_Identifier'])
output.head()
output['predictions_sales'] = test_predict
output.to_csv('D:\\Project\\parag Personal\\aegis\\Kaggle\\BIG_MART\\Python\\predict_output_noCV.csv', index=False)
