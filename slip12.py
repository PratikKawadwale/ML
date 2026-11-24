#slip12,16
# Write a Python program to implement Agglomerative clustering on a synthetic dataset.

import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering

# Generate synthetic dataset
X, y = make_blobs(
    n_samples=100,
    centers=3,
    random_state=42
)

# Apply Agglomerative Clustering
model = AgglomerativeClustering(n_clusters=3)
labels = model.fit_predict(X)

# Display clusters
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='rainbow')
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Agglomerative Clustering")
plt.show()

# Print cluster labels
print("Cluster Labels:")
print(labels)