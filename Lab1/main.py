"""
Created on Wed Aug 21 19:27:10 2019
@author: prakhar
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

data=pd.read_csv("winequality-red.csv",sep=";")

print(data.describe(),"\n\n")

cols=list(data)

for i in cols: 
    if(i!='quality'):
        plt.scatter(data['quality'],data[i])
        plt.title(i)
    plt.show()


for i in cols:print(i,(16-len(i))*' ',"\t:\t",np.corrcoef(data['quality'],data[i])[0][1])

pd.DataFrame.hist(data,figsize=(10,10))

data.groupby('quality').hist('pH')

