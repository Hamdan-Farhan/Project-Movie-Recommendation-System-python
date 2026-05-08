# Import necessary libraries
from sklearn.tree import DecisionTreeClassifier, plot_tree
import numpy as np
import matplotlib.pyplot as plt

# Prepare data: features and labels
# Features: [Genre (0: Action, 1: Comedy, 2: Drama)
# Duration (0: Short, 1: Medium, 2: Long)
# Age Rating (0: Family, 1: Adults)]

X = [
    [0, 2, 1],  # Action, Long, Adults
    [1, 1, 0],  # Comedy, Medium, Family
    [2, 2, 1],  # Drama, Long, Adults
    [0, 0, 0],  # Action, Short, Family
    [1, 0, 0],  # Comedy, Short, Family
    [2, 1, 1],  # Drama, Medium, Adults
]

# Corresponding movie names (target values)
y = [
    "Mad Max: Fury Road",
    "Home Alone",
    "The Godfather",
    "Kung Fu Panda",
    "Minions",
    "Forrest Gump",
]

# Create a Decision Tree Classifier model
model = DecisionTreeClassifier(criterion="gini", max_depth=3, random_state=42)
model.fit(X, y)

# Visualize the Decision Tree
plt.figure(figsize=(12, 8))
plot_tree(model, feature_names=["Genre", "Duration", "Age Rating"], class_names=y, filled=True)
plt.title("Decision Tree for Movie Recommendation")
plt.show()

# Predict a movie based on user input
# Input features: [Genre, Duration, Age Rating]
# Genre: 0 = Action, 1 = Comedy, 2 = Drama
# Duration: 0 = Short, 1 = Medium, 2 = Long
# Age Rating: 0 = Family, 1 = Adults
user_input = np.array([[1, 0, 0]])  # Example: Comedy, Short, Family
prediction = model.predict(user_input)

print(f"The recommended movie for you is: {prediction[0]}")


