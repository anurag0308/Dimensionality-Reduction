# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 00:31:26 2019

@author: Anurag sharma
"""
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler 
#importing_the_dataset
data=pd.read_csv("H:\\FDS\\Iris1.csv", header= None)

#Normalising_the_data
data1 = data.iloc[:,1:5]
scaler = MinMaxScaler(feature_range=(0, 1))
data2 = scaler.fit_transform(data1)
A = data2

#code_for_svd
x1 = (A.T).dot(A)
evals1,evect1 = np.linalg.eig(x1)
idx = evals1.argsort()[::-1]   
eigenValues = evals1[idx]
eigenVectors = evect1[:,idx]
sigma = np.sqrt(np.diag(eigenValues))
sigma.resize((A.shape[0],A.shape[1]))
x3=A.dot(A.T)
evals2,evect2 = np.linalg.eig(x3)
U = np.asmatrix(evect2)
V= np.asmatrix(evect1)
newA=U@sigma@V.T

#finding_eror
error=mean_squared_error(newA,A)