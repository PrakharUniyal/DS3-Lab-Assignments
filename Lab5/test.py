# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 16:08:31 2019

@author: prakhar
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn import model_selection
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
data = pd.read_csv("DiabeticRetinipathy.csv")

X = data.values[:,0:-1]
Y = data.values[:,-1:]

