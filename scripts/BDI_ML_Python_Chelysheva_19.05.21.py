#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 18 16:00:00 2021

@author: Irina Chelysheva
More Machine Learning in Python with scikit-learn
BDI Python Code Clinic
19/05/2021, 11:00

References:
Dr. Melanie Krause

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Neural networks, scaling, PCA for dimentiality reduction, image classification

###Breast cancer dataset
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
print(cancer.keys())
print(cancer.feature_names)
print(cancer.data.shape)
print(cancer.DESCR)
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=42)
from sklearn.neural_network import MLPClassifier

#Build neural network - train and test the model
#Start with the default settings 
mlp = MLPClassifier(random_state=2)
mlp.fit(X_train, y_train)
print('Accuracy on training set:', round(mlp.score(X_train, y_train),4))
#Accuracy on training set: 0.9366
print('Accuracy on test set:', round(mlp.score(X_test, y_test),4))
#Accuracy on test set: 0.9301

#Neural networks are very sensitive to the scaling of the data
#Subtract the mean and divide by the standard deviation so that all features are standardized
mean_on_train = X_train.mean(axis=0)
std_on_train = X_train.std(axis=0)
X_train_scaled = (X_train - mean_on_train) / std_on_train
X_test_scaled = (X_test - mean_on_train) / std_on_train
X_train_scaled.std(axis=0)

#repeat
mlp = MLPClassifier(random_state=2)
mlp.fit(X_train_scaled, y_train)
#ConvergenceWarning: Stochastic Optimizer: Maximum iterations (200) reached and the optimization hasn’t converged yet.
print('Accuracy on training set:', round(mlp.score(X_train_scaled, y_train),4))
#Accuracy on training set: 0.993
print('Accuracy on test set:', round(mlp.score(X_test_scaled, y_test),4))
#Accuracy on test set: 0.965
#Scaling improved the results!

#Increase the number of iterations
mlp = MLPClassifier(random_state=2, max_iter=1000)
mlp.fit(X_train_scaled, y_train)
print('Accuracy on training set:', round(mlp.score(X_train_scaled, y_train),4))
#Accuracy on training set: 0.9977
print('Accuracy on test set:', round(mlp.score(X_test_scaled, y_test),4))
#Accuracy on test set: 0.965

#We can see the 143 predicted values with predict:
mlp.predict(X_test_scaled)
#FInd misclassified observations
diff = y_test - mlp.predict(X_test_scaled)
diff[diff!= 0]
#just 2 out of the 143 test samples are incorrectly predicted
list(diff).index(-1)
#sample 135 is 0 but predicted to be 1
list(diff).index(1)

#3100 weights of the neural network can be obtained with coefs_. 
#we have 30 features and one hidden layer with 100 nodes
mlp.coefs_[0].shape
#gives the weights linking each input feature to each node of the hidden layer
mlp.coefs_[1].shape
#gives the weights when combining the nodes into the final prediction

#height map can visualize the weights linking the 30 features to the hidden layer
plt.figure(figsize=(20, 5))
plt.imshow(mlp.coefs_[0], interpolation='none', cmap='inferno')
plt.yticks(range(30), cancer.feature_names)
plt.xlabel("Node")
plt.ylabel("Input feature")
plt.colorbar()

#PCA - Unsupervised approach
#We are not dividing into training and test set
#use automatic scaler
from sklearn.preprocessing import StandardScaler
cancer = load_breast_cancer()
scaler = StandardScaler()
scaler.fit(cancer.data)
X_scaled = scaler.transform(cancer.data)

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(X_scaled)
X_pca = pca.transform(X_scaled)
print('Original shape:', X_scaled.shape)
#Original shape: (569, 30)
print('Reduced shape:', X_pca.shape)
#Reduced shape: (569, 2)
print(X_pca) #PC1 and PC2 for each of the 569 sample points

#plot PCA, coloring by target variable
mask0 = np.where(cancer.target == 0)
Xmal_pca = X_pca[mask0]
mask1 = np.where(cancer.target == 1)
Xben_pca = X_pca[mask1]
plt.rcParams.update({'font.size': 16})
plt.figure(figsize=(8, 8))
plt.scatter(Xmal_pca[:, 0], Xmal_pca[:, 1],
c='red', label = 'Malignant')
plt.scatter(Xben_pca[:, 0], Xben_pca[:, 1],
c='blue', label = 'Benign')
plt.legend(cancer.target_names, loc="best")
plt.gca().set_aspect("equal")
plt.xlabel("First principal component")
plt.ylabel("Second principal component")

#check the weights of the features in in component
print(pca.components_)
#heatmap of the weights
plt.rcParams.update({'font.size': 10})
plt.imshow(pca.components_, cmap='viridis')
plt.yticks([0, 1], ["First component", "Second component"])
plt.xticks(range(len(cancer.feature_names)),
cancer.feature_names, rotation=60, ha= 'right')
plt.xlabel("Feature")
plt.ylabel("Principal components")
plt.colorbar()

#how much variance is explained by PC1 and PC2?
print(pca.explained_variance_ratio_)
#63.3 % is explained by just 2 components!

#repeat PCA with default settings (30 PCs)
pcafull = PCA()
pcafull.fit(X_scaled)
X_pcafull = pcafull.transform(X_scaled)

#scree plot shows the share of variance accounted for by each PC
plt.title("Scree Plot")
plt.plot(pcafull.explained_variance_ratio_, linewidth=3.0)
plt.xlabel("Number of Principal Components")
plt.ylabel("Variance Explained")

print(np.cumsum(pcafull.explained_variance_ratio_, axis = 0))
#90 % of variance with 7 components, 95% with 10

#use PCA for dimentiality reduction
pca = PCA(n_components=2, whiten=True, random_state=2)
pca.fit(X_train_scaled)
X_train_scaled_pca = pca.transform(X_train)
X_test_scaled_pca = pca.transform(X_test)
print('X_train_scaled_pca.shape:', X_train_scaled_pca.shape)

mlp = MLPClassifier(random_state=2, max_iter=1000)
mlp.fit(X_train_scaled_pca, y_train)
#ConvergenceWarning: Stochastic Optimizer: Maximum iterations (200) reached and the optimization hasn’t converged yet.
print('Accuracy on training set:', round(mlp.score(X_train_scaled_pca, y_train),4))
print('Accuracy on test set:', round(mlp.score(X_test_scaled_pca, y_test),4))
#Accuracy is close to 90 with just 2 PCs!


###Image classification
from sklearn.datasets import fetch_lfw_people
#face images of celebrities from the early 2000s
people = fetch_lfw_people(min_faces_per_person=20, resize=0.7)
print(people.images.shape)
(3023, 87, 65)
target_names = people.target_names
print('Number of classes:', len(people.target_names))
#Number of classes: 62 - people
#plot some
fig, ax = plt.subplots(1,5, figsize =(15,8),
subplot_kw={'xticks': (), 'yticks': ()})
for i in range(5):
    ax[i].imshow(people.images[200*i], cmap= 'gray')
    ax[i].set_title(people.target_names[people.target[200*i]])
plt.show()
#look at pictures from same person
names_list = people.target_names[people.target]
Beckham_occur = np.where(names_list== 'David Beckham')
Beckham5 = [x[15:20] for x in Beckham_occur]
fig, ax = plt.subplots(1,5, figsize =(15,8),
subplot_kw={'xticks': (), 'yticks': ()})
for i in range(5):
    ax[i].imshow(people.images[Beckham5[0][i]], cmap= 'gray')
    ax[i].set_title(people.target_names
      [people.target[Beckham5[0][i]]])
plt.show()

counts = np.bincount(people.target)
print(counts)
#keep only <50 images for each person
#mask is applied to select the respective entries of the people.data and people.target arrays.
mask = np.zeros(people.target.shape, dtype=np.bool)
for target in np.unique(people.target):
    mask[np.where(people.target == target)[0][:50]] = 1
    X_people = people.data[mask]
    y_people = people.target[mask]
    X_people = X_people / 255 #rescaling pixels to 0 to 1
print('Size of final data set:', X_people.shape)
#Size of final data set: (2063, 5655) - 5655 pixels

#apply neural network without PCA
X_train, X_test, y_train, y_test = train_test_split(X_people, y_people,
stratify=y_people, random_state=2)
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
mlp = MLPClassifier(random_state=2, alpha = 2, max_iter= 1000)
mlp.fit(X_train_scaled, y_train)
print('Accuracy on training set:', mlp.score(X_train_scaled, y_train))
print('Accuracy on test set:', mlp.score(X_test_scaled, y_test))

#use PCA to reduce the dimentions
pca = PCA(n_components=80, whiten=True, random_state=2) 
#with 80 PCs, the data set is now of the form (2063, 80) rather than (2063,5655)
pca.fit(X_train_scaled)
X_train_scaled_pca = pca.transform(X_train_scaled)
X_test_scaled_pca = pca.transform(X_test_scaled)
mlp2 = MLPClassifier(random_state=2, alpha=2, max_iter= 1000) #alpha - penalisation, look it up youself
mlp2.fit(X_train_scaled_pca, y_train)
print('Accuracy on training set:', mlp2.score(X_train_scaled_pca, y_train))
print('Accuracy on test set:', mlp2.score(X_test_scaled_pca, y_test))
#We improved both - the accuracy and speed!