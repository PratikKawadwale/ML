#slip3,20
# Write a Python program to implement multiple Linear Regression for predicting
# house price.
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Read dataset
data = pd.read_csv("house.csv")

# Input and output
X = data[['area']]
y = data['price']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Accuracy
print("Accuracy:", model.score(X_test, y_test))

# Predict house price
new_house = pd.DataFrame({'area': [3300]})
prediction = model.predict(new_house)

print("Predicted House Price:", prediction[0])
