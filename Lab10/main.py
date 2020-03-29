# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 12:21:14 2019
@author: Prakhar
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics.cluster import contingency_matrix
from scipy.spatial.distance import cdist
from sklearn.metrics import homogeneity_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def purity_score(y_true, y_pred): return np.sum(np.amax(contingency_matrix(y_true, y_pred), axis=0))/np.sum(contingency_matrix(y_true, y_pred))

dataset = pd.read_csv("iris.csv")
dataX = StandardScaler().fit_transform(dataset.iloc[:,1:5])

mapp = {'Iris-setosa':0, 'Iris-versicolor':1, 'Iris-virginica':2}

true_labels = dataset.iloc[:,5].replace(mapp)
data = PCA(n_components=2).fit_transform(dataX)

for i in range(3):
    plt.scatter(data[true_labels == i,0],data[true_labels == i,1])
plt.show()
#Original Analysis
n_clusters = 3

#K-Means:
kmeans = KMeans(n_clusters=n_clusters)
kmeans.fit(data)
k_means_labels = kmeans.labels_
colors1 = ['red','magenta','pink']
for i in range(n_clusters): plt.scatter(data[k_means_labels == i,0], data[k_means_labels == i,1], s=100, c=colors1[i],marker='.')
plt.show()
print("sum of squared distances of samples:",kmeans.inertia_)

#GMM:
gmm = GaussianMixture(n_components=n_clusters)
gmm.fit(data)
gmm_labels = gmm.predict(data)
colors2 = ['blue','indigo','cyan']
for i in range(n_clusters): plt.scatter(data[gmm_labels == i,0], data[gmm_labels == i,1], s=100, c=colors2[i],marker='.')
plt.show()

print("\nPurity Scores:")
print("K-Means:",purity_score(true_labels,k_means_labels))
print("GMM:",purity_score(true_labels,gmm_labels))
print("-----------------------------------------------")

#Elbow Method for K-Means and GMM:
distortions_k_means = []
distortions_gmm = []
K= [2,3,4,5,6,7,8,9]
for k in K:
    modelk = KMeans(n_clusters=k)
    modelk.fit(data)
    distortions_k_means.append(sum(np.min(cdist(data, modelk.cluster_centers_,'euclidean'),axis=1)) / data.shape[0])
    k_means_labels = modelk.labels_
    for i in range(k): plt.scatter(data[k_means_labels == i,0], data[k_means_labels == i,1], s=100,cmap='rainbow',marker='.')
    plt.show()
    print("sum of squared distances of samples:",modelk.inertia_)

    modelg = GaussianMixture(n_components=k)
    modelg.fit(data)
    distortions_gmm.append(modelg.score(data))
    gmm_labels = modelg.predict(data)
    for i in range(k): plt.scatter(data[gmm_labels == i,0], data[gmm_labels == i,1], s=100,cmap='rainbow',marker='.')
    plt.show()

plt.plot(K,distortions_k_means,marker='*');plt.show()
plt.plot(K,distortions_gmm,marker='*');plt.show()
print("-----------------------------------------------")