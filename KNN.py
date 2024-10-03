import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Sample data: You can replace this with any dataset you'd like to use
X = np.array([[1, 2], [2, 3], [4, 5], [8, 7], [10, 10], 
              [9, 8], [12, 15], [11, 11], [10, 12], [13, 13]])

# Ask user for the number of clusters
n_clusters = int(input("Enter the number of clusters: "))

# Create the KMeans object with the given number of clusters
kmeans = KMeans(n_clusters=n_clusters)

# Fit the model
kmeans.fit(X)

# Predict the cluster labels for the data points
labels = kmeans.predict(X)

# Retrieve the cluster centers
centers = kmeans.cluster_centers_

# Plot the data points and cluster centers
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='rainbow', marker='o', label='Data Points')
plt.scatter(centers[:, 0], centers[:, 1], s=200, c='black', marker='x', label='Centers')

# Display plot with labelsx
plt.title(f'K-Means Clustering with {n_clusters} Clusters')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()
plt.show()