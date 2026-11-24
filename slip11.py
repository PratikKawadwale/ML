#slip11,13
# Write a Python program to implement K-means clustering model and classify the
# employees into various income groups or clusters. Preprocess data if require (i.e.
# drop missing or null values). Use the Elbow method and Silhouette Score to find
# the value of k.

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Load dataset
data = pd.read_csv("position_sal.csv")

# Preprocess data
data = data.dropna()

# Select salary column
X = data[['salary']]

# Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Elbow Method
wcss = []

for k in range(1, 6):
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    model.fit(X_scaled)
    wcss.append(model.inertia_)

plt.plot(range(1, 6), wcss, marker='o')
plt.xlabel("Number of Clusters (K)")
plt.ylabel("WCSS")
plt.title("Elbow Method")
plt.show()

# Silhouette Score
for k in range(2, 6):
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    print("K =", k, "Silhouette Score =", score)

# Choose best K using Silhouette Score
best_k = 2
best_score = -1

for k in range(2, 6):
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)

    if score > best_score:
        best_score = score
        best_k = k

# Final KMeans model
final_model = KMeans(n_clusters=best_k, random_state=42, n_init=10)
data['Income_Group'] = final_model.fit_predict(X_scaled)

# Display employees and their income groups
print("\nEmployee Income Groups:")
print(data[['position', 'salary', 'Income_Group']])

print("\nBest K:", best_k)
print("Best Silhouette Score:", best_score)
