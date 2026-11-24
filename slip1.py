#slip1,2
# Write a Python program to reduce 4D data to 2D data using PCA for iris flower
# dataset. Use Logistic regression and train the model to predict new flowers based
# on given measurements.
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Read CSV file
data = pd.read_csv("iris.csv")

# Select input and output
X = data[['SepalLengthCm', 'SepalWidthCm',
          'PetalLengthCm', 'PetalWidthCm']]
y = data['Species']

# Scale data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Reduce 4D to 2D
pca = PCA(n_components=2)
X = pca.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Check accuracy
print("Accuracy:", model.score(X_test, y_test))

# Predict new flower
new_flower = [[5.1, 3.5, 1.4, 0.2]]
new_flower = scaler.transform(new_flower)
new_flower = pca.transform(new_flower)

result = model.predict(new_flower)

print("Predicted Flower:", result[0])
