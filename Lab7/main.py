# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 23:12:05 2019
@author: Prakhar
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn import model_selection
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

#Loads the dataset
def load_dataset(path_to_file): return pd.read_csv(path_to_file)

#Splitting the data
def train_test_split(data,test_size,seed): return model_selection.train_test_split(data.iloc[:,0:-1],data.iloc[:,-1],test_size=test_size,random_state=seed) 

#Standardizing the non-class attributes of data
def standardisation(dataframe): return StandardScaler().fit_transform(dataframe.values)
def stdatts(dataframe): dataframe.iloc[:,0:-1] = standardisation(dataframe.iloc[:,0:-1])

#Makes 2 gmm models for the given data of 2 classes 
def makegmm(c1,c2,q): return GaussianMixture(n_components=q).fit(c1),GaussianMixture(n_components=q).fit(c2)

#Classifies the data into the two different classes
def classifieddata(data_train,c1,c2): return data_train[data_train[list(data_train)[-1]]==c1].iloc[:,0:-1],data_train[data_train[list(data_train)[-1]]==c2].iloc[:,0:-1]

#Output
def percentage_accuracy(test,pred): return (100*accuracy_score(test,pred))
def print_result(i,test,pred): print("\nK="+str(1+2*i),"\t ",[list(confusion_matrix(test,pred)[i]) for i in range(len(confusion_matrix(test,pred)))],"%.2f" %percentage_accuracy(test,pred),"\b%",end='')


def KNNclassification(X_train, X_test, Y_train,k):
    #Makes prediction using KNN CLassifier for a given k.
    knn = KNeighborsClassifier(n_neighbors=k)#Model
    knn.fit(X_train, np.ravel(Y_train))#Fit
    return knn.predict(X_test)#Prediction

def GMMclassification(X_train0,X_train1,X_test,Y_test,q):
    #Makes prediction using GMM Classifier for a given q.
    gmm0,gmm1 = makegmm(X_train0,X_train1,q)
    #print(gmm0.covariances_)
    #Predictions
    Y_pred=[int(gmm0.score_samples(X_test)[i]<gmm1.score_samples(X_test)[i]) for i in range(len(X_test))]
    #Outputs
    print("\nQ="+str(q)," ",[list(confusion_matrix(Y_test,Y_pred)[i]) for i in range(len(confusion_matrix(Y_test,Y_pred)))],' ',"%.2f" %percentage_accuracy(Y_test,Y_pred),"\b%",end='')
    return percentage_accuracy(Y_test,Y_pred)

def KNNwork(neighbors,X_train,X_test,Y_train,Y_test):
    #Finding accuracy and Confusion Matrices for each K using KNN Classifier.
    print("\nFor KNN CLassifiers:")
    accuracy = np.empty(len(neighbors))
    for j,k in enumerate(neighbors):
        #Classifying the test tuples.
        Y_pred_knn = KNNclassification(X_train, X_test, Y_train,k)
        #Comparing them with the actual classes of test tuples.
        accuracy[j] = percentage_accuracy(Y_test,Y_pred_knn)
        #Confusion Matrix and Accuracy.
        print_result(j,Y_test,Y_pred_knn)
    print()
    #Max accuracy in different values of K.
    print("\nBest Accuracy with KNN at K=",1+2*list(accuracy).index(accuracy.max()),":","%.2f" %accuracy.max(),"\b%\n")
    return accuracy.max()

def GMMwork(q_values,X_train,X_test,Y_train,Y_test):
    #Finding accuracy and Confusion Matrices for each Q using GMM Classifier.
    print("For GMM Classifier:")
    #Classifying data
    X_train0,X_train1=classifieddata(pd.concat([X_train,Y_train],axis=1),0,1)
    accuracy = []
    #Applying GMM Classification by comparing log probabilities for each data sample in test.
    for q in q_values:accuracy.append(GMMclassification(X_train0,X_train1,X_test,Y_test,q))
    #Comparing with unimodal.
    print("\nUnimodal Gaussian distribution:",end='')
    GMMclassification(X_train0,X_train1,X_test,Y_test,1)
    print("\nBest Accuracy with GMM at Q=",2**(1+list(accuracy).index(max(accuracy))),":","%.2f" %max(accuracy),"\b%\n")
    return max(accuracy)

def main():
    
    filename = "SteelPlateFaults-2class"
            
    #Loaded the data.
    data = load_dataset(filename+".csv")
    stdatts(data)
    
    #K={1,3,5,7,9,...,19,21}
    neighbors = np.arange(1,23,2)
    #Q={2,4,8,16}
    q_values = [2,4,8,16]
    
    knnacc=[]
    gmmacc=[]
    
    
    #Splitted the data.
    X_train, X_test, Y_train, Y_test = train_test_split(data,0.3,42)
    #KNN Classification
    knnacc.append(KNNwork(neighbors,X_train,X_test,Y_train,Y_test))
    #GMM CLassification
    gmmacc.append(GMMwork(q_values,X_train,X_test,Y_train,Y_test))
    
    for i in range(1,len(list(data))):
        print("\n\n-------------",i,"--------------\n")
        print("n_dimensions="+str(i),end='')
        reduced_data = pd.concat([pd.DataFrame(data = PCA(n_components=i).fit_transform(data.copy().iloc[:,0:-1]),columns=['comp '+str(n) for n in range(i)]),data.copy().iloc[:,-1]], axis = 1)    

        #Splitted the data.
        X_train, X_test, Y_train, Y_test = train_test_split(reduced_data,0.3,42)    
        #KNN Classification
        knnacc.append(KNNwork(neighbors,X_train,X_test,Y_train,Y_test))  
        #GMM CLassification
        gmmacc.append(GMMwork(q_values,X_train,X_test,Y_train,Y_test))
       
    #Plotting the max accuracies data.
    plt.bar([i for i in range(len(list(data)))],knnacc)
    plt.show()
    plt.bar([i for i in range(len(list(data)))],gmmacc)
    plt.show()

if __name__ == "__main__":
    main()