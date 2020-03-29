# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 01:23:58 2019
@author: prakhar
"""

import numpy as np
import pandas as pd
from matplotlib.pyplot import figure,show
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn import model_selection
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix


def load_dataset(path_to_file): return pd.read_csv(path_to_file)

def normalization(dataframe): return MinMaxScaler().fit_transform(dataframe.values)

def standardisation(dataframe): return StandardScaler().fit_transform(dataframe.values)

def saveas(data,file_name,func):
    data.iloc[:,0:-1] = func(data.iloc[:,0:-1])
    data.to_csv(file_name)
    
def train_test_split(data,test_size,seed): return model_selection.train_test_split(data.values[:,0:-1],data.values[:,-1:],test_size=test_size,random_state=seed,shuffle=True)
#    X0 = data[data[list(data)[-1]] == 0]
#    X1 = data[data[list(data)[-1]] == 1]
#    
#    X0_train,X0_test,Y0_train,Y0_test=model_selection.train_test_split(X0.iloc[:,0:-1],X0.iloc[:,-1:],test_size=test_size,random_state=seed)
#    X1_train,X1_test,Y1_train,Y1_test=model_selection.train_test_split(X1.iloc[:,0:-1],X1.iloc[:,-1:],test_size=test_size,random_state=seed)
#    
#    X_train = pd.concat([X0_train,X1_train])
#    Y_train = pd.concat([Y0_train,Y1_train])
#    X_test = pd.concat([X0_test,X1_test])
#    Y_test = pd.concat([Y0_test,Y1_test])
#    
#    return X_train,X_test,Y_train,Y_test
#    
    
def classification(X_train, X_test, Y_train,k):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, np.ravel(Y_train))
    return knn.predict(X_test)

def percentage_accuracy(test,pred): return (100*accuracy_score(test,pred))

def print_confusion_matrix(i,test,pred): print("K="+str(1+2*i),"\nConfusion Matrix:\n",confusion_matrix(test,pred))

def main():
    filename = "SteelPlateFaults-2class"
        
    #Loaded the data.
    data = load_dataset(filename+".csv")
    
    #Saved the normalized and standardized data in seperate file.
    saveas(data.copy(),filename+"-Normalised.csv",normalization)
    saveas(data.copy(),filename+"-Standardised.csv",standardisation)
    
    #Loaded that data as well.
    ndata = load_dataset(filename+"-Normalised.csv")
    sdata = load_dataset(filename+"-Standardised.csv")
    datasets = [data, ndata, sdata]
    
    #K={1,3,5,7,9,...,19,21}
    neighbors = np.arange(1,23,2)
    
    #Some preprocessing of plots.
    xwidth = np.min(np.diff(neighbors))/3
    fig=figure()
    ax = fig.add_subplot(111)
    
    
    for i in range(3):
        #Splitting the data for training and testing.
        X_train, X_test, Y_train, Y_test = train_test_split(datasets[i],0.3,42)#Test_size=0.3, Seed=42
        #Finding accuracy and Confusion Matrices for each K.
        accuracy = np.empty(len(neighbors))
        for j,k in enumerate(neighbors):
            #Classifying the test tuples.
            Y_pred = classification(X_train, X_test, Y_train,k)
            
            #Comparing them with the actual classes of test tuples.
            accuracy[j] = percentage_accuracy(Y_test,Y_pred)
            
            #Confusion Matrix.
            print_confusion_matrix(j,Y_test,Y_pred)
            
            #Accuracy 
            print("Percentage Accuracy=",percentage_accuracy(Y_test,Y_pred),"\n")
        
        #Max accuracy in different values of K.
        print("Best Accuracy at K=",1+2*list(accuracy).index(accuracy.max()),":",accuracy.max(),"\n")
    
        #Plot.
        ax.bar(neighbors+(int(i-1)*xwidth),accuracy,xwidth/1.2)
    
    show()
    
    
if __name__ == "__main__":
    main()