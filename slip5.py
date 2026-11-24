#slip5 ,6
# Write a Python program to fit the Polynomial linear regression models to
# Salary_positions.csv data. Also, predict the salaries of level 11 and level 12
# employees.
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Load dataset
data = pd.read_csv("position_sal.csv")

# Select input and output
X = data[['level']]
y = data['salary']

# Create polynomial features
poly = PolynomialFeatures(degree=4)
X_poly = poly.fit_transform(X)

# Train model
model = LinearRegression()
model.fit(X_poly, y)

# Predict level 11
salary11 = model.predict(poly.transform([[11]]))

# Predict level 12
salary12 = model.predict(poly.transform([[12]]))

print("Salary for Level 11:", salary11[0])
print("Salary for Level 12:", salary12[0])

# Display graph
plt.scatter(X, y, color='red')
plt.plot(X, model.predict(X_poly), color='blue')

plt.xlabel("Level")
plt.ylabel("Salary")
plt.title("Polynomial Regression")
plt.show()