# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 15:11:44 2019
@author: prakhar
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt

def nanvals(data):print(data.isnull().sum(),'\n'+'-'*28,'\nTotal:',' '*16,data.isnull().sum().sum())

data = pd.read_csv("landslide_data2_miss.csv",sep=',')
odata = pd.read_csv("landslide_data2_original.csv",sep=',')

#List of attribute names.
cols=list(data)

#Number of empty entries in each tuple and the frequency of number of empty entries.
mdat = [len(cols)-i for i in list(data.apply(lambda x: x.count(), axis=1))]
freq=[mdat.count(i) for i in range(9)]

#Printing the tuples having upto 8 values missing.
n0data=data[data.apply(lambda x: x.count()>=len(cols)-8, axis=1)]
print(n0data,n0data.iloc[:,0].count())

#Plotting number of missing values in each tuple (upto 8).
plt.xlabel("no. of missing values")
plt.ylabel("no. of tuples")
plt.plot([i for i in range(1,9)],freq[1:])
plt.show()

#Printing the tuples having more than 50% values missing.
n1data=data[data.apply(lambda x: x.count()<=len(cols)/2, axis=1)]
print(n1data,n1data.iloc[:,0].count())

#Removing tuples according to given constraints.
nanvals(data) #Initial missing values in each attribute.
n2data=data.dropna(thresh=1+len(cols)/2)
n2data=n2data.dropna(subset=['stationid'])
print("Number of miss_vals after removing tuples:")
nanvals(n2data) #Missing values in each attribute now.

#Filling missing values in numerical data columns with median.
numcols=[i for i in cols if (data[i].dtype==np.int64 or data[i].dtype==np.float64) and i!="is_goal" ]
for numcol in numcols: data[numcol].fillna(data[numcol].median(),inplace=True)

#Comparing data.describe for original value data and the filled missing data.
for i in dict(data.describe()).keys():print("For MISS_VAL DATA:\n",dict(data.describe())[i],"\n\nFor ORIGINAL DATA:\n",dict(odata.describe())[i],"\n\n","-"*28)

#Comparing Box plots and calculating RMSE.
for i in cols[2:]:
    print("\n\n----",i,"----")
    plt.boxplot([data[i],odata[i]])
    plt.show()
    print("RMSE:",sqrt(mean_squared_error(odata[i], data[i])))
    
#Manipulating entries in temperature attribute.
temp = data["temperature"]

#Filling Zeroes
plt.hist(temp.fillna(0))
plt.show()

#Filling Median
plt.hist(temp.fillna(temp.median()))
plt.show()

#Filling using Interpolation
plt.hist(temp.interpolate())
plt.show()

#Temperature recorded at stationID t10.
sdata=data[data["stationid"]=="t10"]
plt.plot([i for i in range(124)],list(sdata['temperature']))
