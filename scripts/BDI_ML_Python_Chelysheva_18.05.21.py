#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 17 09:00:00 2021

@author: Irina Chelysheva
More Machine Learning in Python with scikit-learn
BDI Python Code Clinic
18/05/2021, 11:00

References:
Jason Brownlee, Machine Learning Mastery with Python
Abhini Shetye, Feature Selection with sklearn and Pandas

"""

###1) Import the libraries

from sklearn.datasets import load_boston
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
#%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso

###2) Load the dataset
x = load_boston()
print(x.keys())
print(x.feature_names)
print(x.data.shape)
print(x.DESCR)
df = pd.DataFrame(x.data, columns = x.feature_names)
#add and visualise the tagret - output (MEDV)
df["MEDV"] = x.target
plt.hist(x.target)
plt.title('Boston Housing Prices and Count Histogram')
plt.xlabel('price ($1000s)')
plt.ylabel('count')
plt.show()
X = df.drop("MEDV",1)   #Feature Matrix
y = df["MEDV"]          #Separate Target Variable
df.head()
#Split into train and test sets
#Important: we do it before the feature selection
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
all_train = pd.concat([X_train, y_train], axis=1)

###3) Feature selection

##3a) Filter method
#Using Pearson Correlation
plt.figure(figsize=(12,10))
cor = all_train.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()
#Correlation with output variable - we use both positive and negative corr!
cor_target = abs(cor["MEDV"])
#Selecting highly correlated features
relevant_features = cor_target[cor_target>0.5]
relevant_features
#RM, PTRATIO and LSTAT are highly correlated with the output variable MEDV
print(all_train[["LSTAT","PTRATIO"]].corr())
print(all_train[["RM","LSTAT"]].corr())
#RM and LSTAT are highly correlated with each other->keep only one of them
#-> LSTAT and PTRATIO are the features to use

##3b) Wrapper method
#3b1) Backward Elimination
#Adding constant column of ones, mandatory for sm.OLS model - Ordinary Least Squares
X_1 = sm.add_constant(all_train)
#Fitting sm.OLS model
model = sm.OLS(y_train,X_1).fit()
model.pvalues
cols = list(X.columns)
pmax = 1
while (len(cols)>0):
    p= []
    X_1 = X[cols]
    X_1 = sm.add_constant(X_1)
    model = sm.OLS(y,X_1).fit()
    p = pd.Series(model.pvalues.values[1:],index = cols)      
    pmax = max(p)
    feature_with_p_max = p.idxmax()
    if(pmax>0.05):
        cols.remove(feature_with_p_max)
    else:
        break
selected_features_BE = cols
print(selected_features_BE)
#AGE and INDUS had a highest p-value and been removed

#3b2) Recursive Feature Elimination
model = LinearRegression()
#Initializing RFE model, 7 is a random number of features
rfe = RFE(model, 7)
#Transforming data using RFE
X_rfe = rfe.fit_transform(X_train,y_train)  
#Fitting the data to model
model.fit(X_rfe,y_train)
print(rfe.support_)
print(rfe.ranking_)

#Optimal number of features
nof_list=np.arange(1,13)            
high_score=0
#Variable to store the optimum features
nof=0           
score_list =[]
for n in range(len(nof_list)):
    X_subtrain, X_subtest, y_subtrain, y_subtest = train_test_split(X_train,y_train, test_size = 0.3, random_state = 1)
    model = LinearRegression()
    rfe = RFE(model,nof_list[n])
    X_train_rfe = rfe.fit_transform(X_subtrain,y_subtrain)
    X_test_rfe = rfe.transform(X_subtest)
    model.fit(X_train_rfe,y_subtrain)
    score = model.score(X_test_rfe,y_subtest)
    score_list.append(score)
    if(score>high_score):
        high_score = score
        nof = nof_list[n]
print("Optimum number of features: %d" %nof)
print("Score with %d features: %f" % (nof, high_score)) #accuracy

#Getting back using 11 features for RFE
cols = list(X_train.columns)
model = LinearRegression()
#Initializing RFE model
rfe = RFE(model, 11)             
#Transforming data using RFE
X_rfe = rfe.fit_transform(X_train,y_train)  
#Fitting the data to model
model.fit(X_rfe,y_train)              
temp = pd.Series(rfe.support_,index = cols)
selected_features_rfe = temp[temp==True].index
print(selected_features_rfe)
#AGE and B removed


##3c) Embedded Method using Lasso (hint: you can also try Ridge!)
reg = LassoCV()
reg.fit(X_train, y_train)
#print("Best alpha using built-in LassoCV: %f" % reg.alpha_)
print("Best score using built-in LassoCV: %f" %reg.score(X_train,y_train))
coef = pd.Series(reg.coef_, index = X.columns)
print("LASSO picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")
imp_coef = coef.sort_values()
matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)
imp_coef.plot(kind = "barh")
plt.title("Feature importance using Lasso Model")
#LASSO eliminated 3 variables - NOX, CHAS and INDUS

###4) Choosing the best regression model
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

# Build models to check the algorithms
models = []
models.append(('LR', LinearRegression()))
models.append(('R', Ridge()))
models.append(('LASSO', Lasso()))
models.append(('EN', ElasticNet()))
models.append(('KN', KNeighborsRegressor()))
models.append(('DTR', DecisionTreeRegressor(random_state=1)))
models.append(('SVR', SVR(gamma='auto'))) #support vector regression
# Evaluate each model
results = []
names = []
seed = 7
kfold = model_selection.KFold(n_splits=10, random_state=seed)
scoring = 'neg_mean_squared_error'
for name, model in models:
	cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='neg_mean_squared_error')
	results.append(cv_results)
	names.append(name)
	print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
# Compare algorithms
plt.boxplot(results, labels=names)
plt.title('Algorithm Comparison')
plt.show()
#DTR performs the best

#Subset different features based on feature selection
X_a = X[["LSTAT","PTRATIO"]]
X_b1 = X.drop(["AGE","TAX","B"],1)
X_b2 = X.drop(["AGE","B"],1)
X_c = X.drop(["NOX", "CHAS", "INDUS"],1)

Xa_train, Xa_test, ya_train, ya_test = train_test_split(X_a, y, test_size=0.3, random_state=1)
Xb1_train, Xb1_test, yb1_train, yb1_test = train_test_split(X_b1, y, test_size=0.3, random_state=1)
Xb2_train, Xb2_test, yb2_train, yb2_test = train_test_split(X_b2, y, test_size=0.3, random_state=1)
Xc_train, Xc_test, yc_train, yc_test = train_test_split(X_c, y, test_size=0.3, random_state=1)

# Evaluate each model again
results = []
names = []
seed = 7
kfold = model_selection.KFold(n_splits=10, random_state=seed)
scoring = 'neg_mean_squared_error'
for name, model in models:
	cv_results = cross_val_score(model, Xb2_train, yb2_train, cv=kfold, scoring='neg_mean_squared_error')
	results.append(cv_results)
	names.append(name)
	print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
# Compare algorithms
plt.boxplot(results, labels=names)
plt.title('Algorithm Comparison')
plt.show()
# DTR peforms best - with b2 feature selection (RFE)
# Evaluate the performance on test set
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
model = DecisionTreeRegressor(random_state=1)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print(mean_squared_error(y_test, predictions))
print(mean_absolute_error(y_test, predictions))
#Check best algorith on selected features
model.fit(Xb2_train, yb2_train)
predictions = model.predict(Xb2_test)
print(mean_squared_error(yb2_test, predictions))
print(mean_absolute_error(yb2_test, predictions))

#Summary: 
#Recursive Feature Elimination was the most precise method for feature selection
#DecisionTreeRegression was the best perfoming algorithm for our data

#####################

#Small add-on
# mutual feature selection
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_regression
def select_features(X_train, y_train, X_test):
	# configure to select a subset of features
	fs = SelectKBest(score_func=mutual_info_regression, k=11)
	# learn relationship from training data
	fs.fit(X_train, y_train)
	# transform train input data
	X_train_fs = fs.transform(X_train)
	# transform test input data
	X_test_fs = fs.transform(X_test)
	return X_train_fs, X_test_fs, fs
 
# feature selection
X_train_fs, X_test_fs, fs = select_features(X_train, y_train, X_test)
# fit the model
model = DecisionTreeRegressor(random_state=1)
model.fit(X_train_fs, y_train)
# evaluate the model
predictions = model.predict(X_test_fs)
# evaluate predictions
print(mean_squared_error(y_test, predictions))
print(mean_absolute_error(y_test, predictions))

