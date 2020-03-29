# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 21:42:11 2019
@author: prakhar
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.preprocessing import PolynomialFeatures
from mpl_toolkits.mplot3d import Axes3D

degrees=[2,3,4,5]
def train_test_split(data,test_size,seed): return model_selection.train_test_split(data.iloc[:,0:-1],data.iloc[:,-1:],test_size=test_size,random_state=seed,shuffle=True)
def rms(y_test,y_pred): return sqrt(mean_squared_error(y_test,y_pred))

data = pd.read_csv("atmosphere_data.csv")

X_train, X_test, Y_train, Y_test = train_test_split(data,0.3,42)

pd.concat([X_train,Y_train],axis=1).to_csv("atmosphere_train.csv")
pd.concat([X_test,Y_test],axis=1).to_csv("atmosphere-test.csv")

X1_train = pd.DataFrame(X_train["pressure"])
X1_test = pd.DataFrame(X_test["pressure"])

print("----------Simple Linear Regression----------")
regressor=LinearRegression()
regressor.fit(X1_train,Y_train)
Y_pred_train=regressor.predict(X1_train)
Y_pred_test=regressor.predict(X1_test)
    
plt.scatter(X1_train,Y_train)
plt.scatter(X1_train,Y_pred_train,color='red')
plt.title("Best Fit Line for Train Data")
plt.xlabel("Pressure Value");plt.ylabel("temperature")
plt.show()

print("Root Mean Square Error on train data:",rms(Y_train,Y_pred_train))
print("Root Mean Square Error on test data: ",rms(Y_test,Y_pred_test))

plt.scatter(Y_test,Y_pred_test)
plt.title("Test Data: Actual v/s Predicted")
plt.xlabel("Actual temperature");plt.ylabel("Predicted temperature")
plt.show()

print("----------Simple Non-Linear Regression----------")
trainaccs=[]
testaccs=[]

X_poly_train = [PolynomialFeatures(degree = p).fit_transform(X1_train) for p in degrees]
X_poly_test  = [PolynomialFeatures(degree = p).fit_transform(X1_test) for p in degrees] 

nregressor = [LinearRegression() for i in range(len(degrees))]
for i in range(len(degrees)):nregressor[i].fit(X_poly_train[i],Y_train)

Y_pred_train = [nregressor[i].predict(X_poly_train[i]) for i in range(len(degrees))]
Y_pred_test  = [nregressor[i].predict(X_poly_test[i]) for i in range(len(degrees))]

for i in range(len(degrees)):
    plt.scatter(X1_train,Y_train)
    plt.scatter(X1_train,Y_pred_train[i],color='red')
    plt.title("Best Fit curve of degree "+str(degrees[i])+" for Train Data")
    plt.xlabel("Pressure Value");plt.ylabel("temperature")
    plt.show()

for i in range(len(degrees)):
    print("degree=",degrees[i])
    print("Root Mean Square Error on train data:",rms(Y_train,Y_pred_train[i]))
    trainaccs.append(rms(Y_train,Y_pred_train[i]))
    print("Root Mean Square Error on test data: ",rms(Y_test,Y_pred_test[i]))
    testaccs.append(rms(Y_test,Y_pred_test[i]))

plt.bar(degrees,trainaccs);plt.show()
plt.bar(degrees,testaccs);plt.show()

plt.scatter(Y_test,Y_pred_test[testaccs.index(min(testaccs))])
plt.title("Test Data: Actual v/s Predicted by regression model of degree "+str(2+testaccs.index(min(testaccs))))
plt.xlabel("Actual temperature");plt.ylabel("Predicted temperature")
plt.show()


print("----------Multivar Linear Regression----------")
mregressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred_train=regressor.predict(X_train)
Y_pred_test=regressor.predict(X_test)
    
print("Root Mean Square Error on train data:",rms(Y_train,Y_pred_train))
print("Root Mean Square Error on test data: ",rms(Y_test,Y_pred_test))

plt.scatter(Y_test,Y_pred_test)
plt.title("Test Data: Actual v/s Predicted")
plt.xlabel("Actual temperature");plt.ylabel("Predicted temperature")
plt.show()


print("----------Multivar Polynomial Regression----------")
trainaccs=[]
testaccs=[]

X_poly_train = [PolynomialFeatures(degree = p).fit_transform(X_train) for p in degrees]
X_poly_test  = [PolynomialFeatures(degree = p).fit_transform(X_test) for p in degrees] 

nregressor = [LinearRegression() for i in range(len(degrees))]
for i in range(len(degrees)):nregressor[i].fit(X_poly_train[i],Y_train)

Y_pred_train = [nregressor[i].predict(X_poly_train[i]) for i in range(len(degrees))]
Y_pred_test  = [nregressor[i].predict(X_poly_test[i]) for i in range(len(degrees))]

for i in range(len(degrees)):
    print("degree=",degrees[i])
    print("Root Mean Square Error on train data:",rms(Y_train,Y_pred_train[i]))
    trainaccs.append(rms(Y_train,Y_pred_train[i]))
    print("Root Mean Square Error on test data: ",rms(Y_test,Y_pred_test[i]))
    testaccs.append(rms(Y_test,Y_pred_test[i]))

plt.bar(degrees,trainaccs);plt.show()
plt.bar(degrees,testaccs);plt.show()

plt.scatter(Y_test,Y_pred_test[testaccs.index(min(testaccs))])
plt.title("Test Data: Actual v/s Predicted by regression model of degree "+str(2+testaccs.index(min(testaccs))))
plt.xlabel("Actual temperature");plt.ylabel("Predicted temperature")
plt.show()


print("----------Final Analysis----------")
correlations = data.corr()['temperature'].drop('temperature')
c = sorted([abs(i) for i in dict(correlations).values()])
impattr = ['']*2
for i in correlations.keys():
    if abs(dict(correlations)[i])== c[-1]: impattr[0]=i
    if abs(dict(correlations)[i])== c[-2]: impattr[1]=i

X2_train = pd.DataFrame(X_train[impattr])
X2_test = pd.DataFrame(X_test[impattr])

print("----------A----------")    
mregressor=LinearRegression()
regressor.fit(X2_train,Y_train)
Y_pred_train=regressor.predict(X2_train)
Y_pred_test=regressor.predict(X2_test)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(list(X2_train[impattr[0]]),list(X2_train[impattr[1]]),Y_train,zdir='z', s=20, c=None, depthshade=True)
ax.scatter(list(X2_train[impattr[0]]),list(X2_train[impattr[1]]),Y_pred_train,zdir='z', s=20, c=None, depthshade=True)
plt.show()

print("Root Mean Square Error on train data:",rms(Y_train,Y_pred_train))
print("Root Mean Square Error on test data: ",rms(Y_test,Y_pred_test))

plt.scatter(Y_test,Y_pred_test)
plt.title("Test Data: Actual v/s Predicted")
plt.xlabel("Actual temperature");plt.ylabel("Predicted temperature")
plt.show()

print("----------B----------")
trainaccs=[]
testaccs=[]

X_poly_train = [PolynomialFeatures(degree = p).fit_transform(X2_train) for p in degrees]
X_poly_test  = [PolynomialFeatures(degree = p).fit_transform(X2_test) for p in degrees] 

nregressor = [LinearRegression() for i in range(len(degrees))]
for i in range(len(degrees)):nregressor[i].fit(X_poly_train[i],Y_train)

Y_pred_train = [nregressor[i].predict(X_poly_train[i]) for i in range(len(degrees))]
Y_pred_test  = [nregressor[i].predict(X_poly_test[i]) for i in range(len(degrees))]

for i in range(len(degrees)):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(list(X2_train[impattr[0]]),list(X2_train[impattr[1]]),Y_train,zdir='z', s=20, c=None, depthshade=True)
    ax.scatter(list(X2_train[impattr[0]]),list(X2_train[impattr[1]]),Y_pred_train[i],zdir='z', s=20, c=None, depthshade=True)
    plt.show()

for i in range(len(degrees)):
    print("degree=",degrees[i])
    print("Root Mean Square Error on train data:",rms(Y_train,Y_pred_train[i]))
    trainaccs.append(rms(Y_train,Y_pred_train[i]))
    print("Root Mean Square Error on test data: ",rms(Y_test,Y_pred_test[i]))
    testaccs.append(rms(Y_test,Y_pred_test[i]))

plt.bar(degrees,trainaccs);plt.show()
plt.bar(degrees,testaccs);plt.show()

plt.scatter(Y_test,Y_pred_test[testaccs.index(min(testaccs))])
plt.title("Test Data: Actual v/s Predicted by regression model of degree "+str(2+testaccs.index(min(testaccs))))
plt.xlabel("Actual temperature");plt.ylabel("Predicted temperature")
plt.show()