import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN as DBS
from sklearn.cluster import AgglomerativeClustering as AGC
from sklearn import metrics
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def purity_score(y_true, y_pred):     
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)       
    row_ind, col_ind = linear_sum_assignment(-contingency_matrix) 
    return contingency_matrix[row_ind, col_ind].sum() / np.sum(contingency_matrix)

dataset = pd.read_csv("iris.csv")
#dataX = dataset.iloc[:,1:5]
dataX = StandardScaler().fit_transform(dataset.iloc[:,1:5])

mapp = {'Iris-setosa':0, 'Iris-versicolor':1, 'Iris-virginica':2}

true_labels = dataset.iloc[:,5].replace(mapp)
data = PCA(n_components=2).fit_transform(dataX)
data = np.array(data)

print("_____________K-Means_____________")
kmeans = KMeans(n_clusters=3)
kmeans.fit(data)

y_km = kmeans.fit_predict(data)

print("purity_score:",purity_score(true_labels, y_km))

plt.scatter(data[:,0],data[:,1],c=y_km,cmap="rainbow",s=10)
plt.show()

print("_____________Agglomerative Clustering_____________")
agc = AGC(n_clusters=3)

agc.fit(data)

y_agc = agc.fit_predict(data)

print("purity_score:",purity_score(true_labels, y_agc))

plt.scatter(data[:,0],data[:,1],c=y_agc,cmap="rainbow",s=10)
plt.show()

print("_____________DBSCAN_____________")

for i in [0.05,0.5,0.95]:
    pscores = []
    for j in [1,5,10,20]: 
        dbs = DBS(eps=i,min_samples=j)
        dbs.fit(data)
        y_dbs = dbs.fit_predict(data)

        print("purity_score: "+"%.4f"%purity_score(true_labels, y_dbs))
        pscores.append(purity_score(true_labels, y_dbs))
        plt.scatter(data[:,0],data[:,1],c=y_dbs,cmap="rainbow",s=10)
        plt.show()
    plt.plot([1,5,10,20],pscores)
    plt.show()