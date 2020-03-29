# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 16:10:34 2019
@author: Prakhar
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn import model_selection
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score,confusion_matrix
from scipy.stats import multivariate_normal



def load_dataset(path_to_file): return pd.read_csv(path_to_file)

def standardisation(dataframe): return StandardScaler().fit_transform(dataframe.values)
    
def train_test_split(data,test_size,seed): return model_selection.train_test_split(data.iloc[:,0:-1],data.iloc[:,-1],test_size=test_size,random_state=seed) 
    
def KNNclassification(X_train, X_test, Y_train,k):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, np.ravel(Y_train))
    return knn.predict(X_test)

def GNBclassification(X_train, X_test, Y_train):
    gnb = GaussianNB()
    gnb.fit(X_train, np.ravel(Y_train))
    return gnb.predict(X_test)

def percentage_accuracy(test,pred): return (100*accuracy_score(test,pred))

def print_confusion_matrix(i,test,pred): print("\nK="+str(1+2*i),"\nConfusion Matrix:\n",confusion_matrix(test,pred))

def main():
    filename = "SteelPlateFaults-2class"
        
    #Loaded the data.
    data = load_dataset(filename+".csv")
    
    
    knnaccs=[]
    mnaccs=[]
    for i in range(1,len(list(data))):
        cdata = data.copy()
        
        pca = PCA(n_components=i)
        
        #Standardized the Attributes(Other than class label)
        X = cdata.iloc[:,0:-1]
        Y = cdata.iloc[:,-1]
        X = standardisation(X)
        
        dc = pca.fit_transform(X)
        reduced_X = pd.DataFrame(data = dc,columns=['comp '+str(n) for n in range(i)])
        reduced_data = pd.concat([reduced_X,Y], axis = 1)
    
        #K={1,3,5,7,9,...,19,21}
        neighbors = np.arange(1,23,2)
        
        #Splitting the data for training and testing.
        X_train, X_test, Y_train, Y_test = train_test_split(reduced_data,0.3,42)#Test_size=0.3, Seed=42
        
        #Finding accuracy and Confusion Matrices for each K using KNN Classifier.
        
        print("\nFor KNN CLassifierrs:")
        accuracy = np.empty(len(neighbors))
        for j,k in enumerate(neighbors):
            #Classifying the test tuples.
            Y_pred_knn = KNNclassification(X_train, X_test, Y_train,k)
            #Comparing them with the actual classes of test tuples.
            accuracy[j] = percentage_accuracy(Y_test,Y_pred_knn)
            #Confusion Matrix.
            print_confusion_matrix(j,Y_test,Y_pred_knn)
            #Accuracy
            print("%.2f" %percentage_accuracy(Y_test,Y_pred_knn),"\b% ")
            
        print()
        #Max accuracy in different values of K.
        print("\nBest Accuracy with KNN at K=",1+2*list(accuracy).index(accuracy.max()),":","%.2f" %accuracy.max(),"\b%")
        knnaccs.append(accuracy.max())
        
        ############################################
        
        train_data = pd.concat([X_train,Y_train],axis=1)
        
        dist0 = train_data[train_data[list(train_data)[-1]]==0].iloc[:,0:-1]
        dist1 = train_data[train_data[list(train_data)[-1]]==1].iloc[:,0:-1]
        
        m0,c0 = list(dist0.mean()),np.array(dist0.cov())
        m1,c1 = list(dist1.mean()),np.array(dist1.cov())

        xx = list(X_test.values)
        Y_pred = []
        
        for i in range(len(Y_test)):
            if multivariate_normal.pdf(xx[i], mean=m0, cov=c0, allow_singular=True)<multivariate_normal.pdf(xx[i], mean=m1, cov=c1,allow_singular=True):
                Y_pred.append(1)
            else:
                Y_pred.append(0)

        ############################################
        
        print("\nMultivariate_Normal Distribution Classification")
        print(confusion_matrix(Y_test,Y_pred))
        print("%.2f" %percentage_accuracy(Y_test,Y_pred),"\b% ")
        mnaccs.append(percentage_accuracy(Y_test,Y_pred))
    
    #Plotting the accuracies over varying dimensions of data.
    plt.bar([i for i in range(27)],knnaccs)    
    plt.show()
    plt.bar([i for i in range(27)],mnaccs)
    plt.show()
    
    print("Best Accuracy for KNN:",max(knnaccs))
    print("Best Accuracy for Multi_norm:",max(mnaccs))

if __name__ == "__main__":
    main()