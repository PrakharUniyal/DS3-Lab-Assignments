#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 20:58:13 2019

@author: murali
"""
import pandas as pd
import numpy as np

import math as m
from math import sqrt

import matplotlib.pyplot as plt

import collections

from sklearn import metrics
from sklearn import mixture
from sklearn import  datasets
from sklearn import decomposition
from sklearn.model_selection import train_test_split


#Clustering
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN 
from matplotlib.colors import ListedColormap

#Regression
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from matplotlib.tri import Triangulation

#Autoregression
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.ar_model import AR
