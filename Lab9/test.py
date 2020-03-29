# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 14:16:43 2019

@author: prakhar
"""
import matplotlib.pyplot as plt
import numpy as np

x = [i for i in range(-10,11)]
y = [i**3 for i in range(-10,11)]

plt.plot(x,x)
plt.plot(x,y)

print(np.corrcoef(x,y))