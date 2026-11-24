#slip17,18
# Write a Python program to implement Decision Tree classifier model on data
# which is extracted from images that were taken from genuine and forged
# banknote-like specimens.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load CSV file (without column headings)
data = pd.read_csv("data_banknote_authentication.csv", header=None)

# Assign column names
data.columns = ['variance', 'skewness', 'curtosis',
                'entropy', 'class']

# Remove missing values
data = data.dropna()

# Split input and output
X = data[['variance', 'skewness', 'curtosis', 'entropy']]
y = data['class']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Create Decision Tree model
model = DecisionTreeClassifier(random_state=42)

# Train model
model.fit(X_train, y_train)

# Predict test data
y_pred = model.predict(X_test)

# Calculate accuracy
print("Accuracy:", accuracy_score(y_test, y_pred) * 100)

# Predict new banknote
new_data = [[3.6216, 8.6661, -2.8073, -0.44699]]
prediction = model.predict(new_data)

if prediction[0] == 0:
    print("Banknote is Genuine")
else:
    print("Banknote is Forged")
