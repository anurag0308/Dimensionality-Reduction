# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 20:08:11 2019

@author: Anurag sharma
"""

import pandas as pd
import numpy as np
import sklearn.preprocessing as sp
iris = pd.read_csv("H:\\FDS\\Iris.csv")
print (iris)
data=iris.values
x=data[:,0],data[:,1],data[:,2],data[:,3],data[:,4]
print (x)