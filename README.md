# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import required libraries such as Pandas, Matplotlib, and sklearn.
2. Load the dataset and define features (X) and target (y).
3. Split the dataset into training and testing sets.
4. Train the Decision Tree Classifier model.
5. Predict the output and evaluate accuracy.
6. Visualize the Decision Tree using plot_tree().

## Program:
# Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
# Developed by: VIJIYALAKSHMI A
# RegisterNumber: 212225240185
```
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report

# Employee dataset
data = {
    'Age': [25, 30, 45, 35, 22, 40, 28, 50, 23, 31],
    'Salary': [40000, 50000, 80000, 60000, 35000, 90000, 45000, 100000, 30000, 52000],
    'Years': [1, 5, 10, 7, 2, 12, 3, 15, 1, 6],
    'Churn': [1, 0, 0, 0, 1, 0, 1, 0, 1, 0]
}

df = pd.DataFrame(data)

# Features and Target
X = df[['Age', 'Salary', 'Years']]
y = df['Churn']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Creation
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# -------- Decision Tree Diagram --------
plt.figure(figsize=(12,6))
plot_tree(model,
          feature_names=['Age', 'Salary', 'Years'],
          class_names=['No Churn', 'Churn'],
          filled=True)
plt.title("Decision Tree for Employee Churn")
plt.show()
```

## Output:
<img width="871" height="752" alt="Screenshot 2026-05-11 185939" src="https://github.com/user-attachments/assets/9d062b4a-9c1c-4578-9c68-64845a2b1c83" />

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
