#slip7 ,8
# Write a Python program to classify the iris flowers dataset using SVM and find out
# the flower type depending on the given input data, like sepal length, sepal width,
# petal length, and petal width. Find the accuracy of all SVM kernels.
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Load CSV dataset
data = pd.read_csv("iris.csv")

# Input and output
X = data[['SepalLengthCm', 'SepalWidthCm',
          'PetalLengthCm', 'PetalWidthCm']]

y = data['Species']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Standardize data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# SVM kernels
kernels = ['linear', 'poly', 'rbf', 'sigmoid']

best_accuracy = 0
best_model = None

for kernel in kernels:
    model = SVC(kernel=kernel)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred) * 100

    print(kernel, "Accuracy:", accuracy)

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model

# Predict new flower
new_flower = [[5.1, 3.5, 1.4, 0.2]]

new_flower = scaler.transform(new_flower)

prediction = best_model.predict(new_flower)

print("Predicted Flower:", prediction[0])