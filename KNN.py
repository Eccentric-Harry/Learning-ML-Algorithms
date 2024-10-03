import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

X, y = make_blobs (n_samples=300, centers=3, cluster_std=0.6, random_state=0)
plt.scatter (X[:, 0], X[:, 1], s=50) 
plt.title("Data Before Clustering") 
plt.show()

kmeans =  KMeans(n_clusters=3) 
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

plt.scatter (X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')

centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=200, alpha=0.75, marker='x') 
plt.title("Data After K-Means Clustering")
plt.show()