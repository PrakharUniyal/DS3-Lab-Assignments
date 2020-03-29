# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 22:20:51 2019

@author: prakhar
"""
import numpy as np
import matplotlib.pyplot as plt

x = [1,2,3]
y = [4,5,6]

n = np.array([x,y])

print(n.shape[0])

#-0.86057496  0.50932381

plt.quiver([-1],0.86057496/-0.50932381)




