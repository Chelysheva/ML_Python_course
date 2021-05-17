#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 16 13:00:00 2021

@author: Irina Chelysheva

Machine Learning in Python with scikit-learn
BDI Python Code Clinic
17/05/2020, 11:00

References:
Jason Brownlee, Machine Learning Mastery with Python
René Laqua, kaggle Python notebook

"""

### 1) Check the versions of libraries and import them
 
# Python version
import sys
print('Python: {}'.format(sys.version))
# scipy
import scipy
print('scipy: {}'.format(scipy.__version__))
# numpy
import numpy
print('numpy: {}'.format(numpy.__version__))
# matplotlib
import matplotlib
print('matplotlib: {}'.format(matplotlib.__version__))
# pandas
import pandas
print('pandas: {}'.format(pandas.__version__))
# scikit-learn
import sklearn
print('sklearn: {}'.format(sklearn.__version__))

### 2) Load dataset
from pandas import read_csv
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'glu', 'pres', 'skin', 'ins', 'bmi', 'pedi', 'age', 'class']
dataset = read_csv(url, names=names)

# Full dataset information:
# Consists of several medical predictor variables and one target variable (class)
# Attributes:
#1. Number of times pregnant
#2. Plasma glucose concentration a 2 hours in an oral glucose tolerance test
#3. Diastolic blood pressure (mm Hg)
#4. Triceps skin fold thickness (mm)
#5. 2-Hour serum insulin (mu U/ml)
#6. Body mass index (weight in kg/(height in m)^2)
#7. Diabetes pedigree function
#8. Age (years)
#9. Class variable (0 or 1)

### 3) Discover the dataset
# Shape
print(dataset.shape)
# Head
print(dataset.head(20))
# Descriptions
print(dataset.describe())
# Class distribution
print(dataset.groupby('class').size())

### 4) Clean-up the dataset
# Replace zero/invalid values with the mean in the group
# It does not seem to improve the results in this case, might be useful in others
dataset_nozeros = dataset.copy()
zero_fields = ['glu', 'pres', 'skin', 'ins', 'bmi'] 
dataset[zero_fields] = dataset[zero_fields].replace(0, numpy.nan)
dataset[zero_fields] = dataset[zero_fields].fillna(dataset_nozeros.mean())
print(dataset.describe())  # check that there are no invalid values left

### 5) Visualize the data
from matplotlib import pyplot
# Box plots - gives us a clear idea of the distribution of the input attributes
dataset.plot(kind='box', subplots=True, layout=(3,3), sharex=False, sharey=False)
pyplot.show()
# Histograms
dataset.hist()
pyplot.show()

#Hint: you can use sklearn.preprocessing functions to scale the data
#import sklearn.preprocessing

### 6) Compare the algorithms
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Split-out validation dataset
array = dataset.values
# Separate the last - class column from training/testing data
X = array[:,0:8]
Y = array[:,8]
test_size = 0.25
# Pass an int for reproducible output across multiple function calls
seed = 5
# Split dataset in a stratified fashion
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed, stratify=dataset['class'])

# Build models to check the algorithms
models = []
models.append(('LR', LogisticRegression(solver='liblinear')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))
# Evaluate each model
results = []
names = []
for name, model in models:
	kfold = StratifiedKFold(n_splits=10, random_state=seed)
	cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
	results.append(cv_results)
	names.append(name)
	print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
# Compare algorithms
pyplot.boxplot(results, labels=names)
pyplot.title('Algorithm Comparison')
pyplot.show()

### 7) Use the best model to make predictions and evaluate it
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

# Make predictions on validation dataset
model = LinearDiscriminantAnalysis()
model.fit(X_train, Y_train)
predictions = model.predict(X_test)

# Evaluate predictions
print(accuracy_score(Y_test, predictions))
print(classification_report(Y_test, predictions))

# Plot confusion matrix
print(confusion_matrix(Y_test, predictions))
conf_matrix = confusion_matrix(Y_test, predictions)
pyplot.figure()
pyplot.matshow(conf_matrix, cmap='Pastel1')
for x in range(0, 2):
    for y in range(0, 2):
        pyplot.text(x, y, conf_matrix[x, y])
        
pyplot.ylabel('expected label')
pyplot.xlabel('predicted label')
pyplot.show()

# Calculate Sensitivity and Specificity
TP = conf_matrix[1, 1]
TN = conf_matrix[0, 0]
FP = conf_matrix[0, 1]
FN = conf_matrix[1, 0]

print("Sensitivity: %.4f" % (TP / float(TP + FN)))
print("Specificity  : %.4f" % (TN / float(TN + FP)))

# Null accuracy - accuracy to be achieved by always predicting the most frequent class
most_outcome = dataset['class'].median()
prediction_most = [most_outcome for i in range(len(Y_test))]
print(accuracy_score(Y_test, prediction_most))

### 8) Save the finalized model to disk
import pickle
filename = 'finalized_model.sav'
pickle.dump(model, open(filename, 'wb'))
 
# At some point later...
 
# Load the model back from disk
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_test, Y_test)
print(result)

####### Additional imrpovements #######

### 9) Improving the model by changing the classification threshold
# Print the first 10 predicted responses
model.predict(X_test)[0:10]
# Print the first 10 predicted probabilities of class membership
model.predict_proba(X_test)[0:10]
# Histogram of predicted probabilities
save_predictions_proba = model.predict_proba(X_test)[:, 1]
pyplot.hist(save_predictions_proba, bins=10)
pyplot.xlim(0,1) # x-axis limit from 0 to 1
pyplot.title('Histogram of predicted probabilities')
pyplot.xlabel('Predicted probability of diabetes')
pyplot.ylabel('Frequency')
pyplot.show()

# Change threshold: 
# Predict diabetes if the predicted probability is greater than 0.3
from sklearn.preprocessing import binarize

# Return 1 for all values greater than 0.3, otherwise - 0
prediction2 = binarize(save_predictions_proba.reshape(-1, 1), 0.3)
print(accuracy_score(Y_test, prediction2))

conf_matrix2 = confusion_matrix(Y_test, prediction2)

pyplot.figure()
pyplot.matshow(conf_matrix2, cmap='Pastel1')
for x in range(0, 2):
    for y in range(0, 2):
        pyplot.text(x, y, conf_matrix2[x, y])
        
pyplot.ylabel('expected label')
pyplot.xlabel('predicted label')
pyplot.show()

TP = conf_matrix2[1, 1]
TN = conf_matrix2[0, 0]
FP = conf_matrix2[0, 1]
FN = conf_matrix2[1, 0]

print("new Sensitivity: %.4f" % (TP / float(TP + FN)))
print("new Specificity  : %.4f" % (TN / float(TN + FP)))

### 10) Plot ROC curve
from sklearn.metrics import roc_curve, auc
# Input: first argument - true values, second argument - predicted probabilities
# Output: FPR, TPR, thresholds
# FPR: False Positive Rate
# TPR: True Positive Rate
FPR, TPR, thresholds = roc_curve(Y_test, save_predictions_proba)

# Plot!
pyplot.figure(figsize=(10,5))
pyplot.plot(FPR, TPR)
pyplot.xlim([0.0, 1.0])
pyplot.ylim([0.0, 1.0])
pyplot.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
pyplot.plot(FPR, TPR, lw=2, label='LinearDiscriminantAnalysis (AUC = %0.2f)' % auc(FPR, TPR))
pyplot.title('ROC curve for diabetes classifier')
pyplot.xlabel('False Positive Rate (1 - Specificity)')
pyplot.ylabel('True Positive Rate (Sensitivity)')
pyplot.grid(True)
pyplot.legend(loc="lower right")