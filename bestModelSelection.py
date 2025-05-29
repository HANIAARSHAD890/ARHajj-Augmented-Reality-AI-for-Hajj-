import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import joblib
df = pd.read_csv("encoded_dataset.csv")

# Select one-hot encoded health condition columns
health_cols = [
    "Health_Condition_Dehydration",
    "Health_Condition_Fainting",
    "Health_Condition_Heatstroke",
    "Health_Condition_Injured",
    "Health_Condition_Normal"
]

df["Health_Condition"] = df[health_cols].idxmax(axis=1).str.replace("Health_Condition_", "")
# 1. Load your dataset (replace with your actual file path)
df.drop(columns=health_cols, inplace=True)

# 2. Define features (X) and target (y)
target_col = "Health_Condition"
X = df.drop(columns=[target_col])
y = df[target_col]

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Define models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "KNN": KNeighborsClassifier(),
    "SVM": SVC(),
    "Naive Bayes": GaussianNB(),
}

# 5. Train and evaluate models
results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average='weighted')
    results.append((name, acc, f1))
    print(f"--- {name} ---")
    print("Accuracy:", acc)
    print("F1 Score:", f1)
    print(classification_report(y_test, preds))
    print()

# 6. Visual comparison
results_df = pd.DataFrame(results, columns=["Model", "Accuracy", "F1 Score"])
results_df.sort_values("F1 Score", ascending=False, inplace=True)

# Plot
plt.figure(figsize=(10, 6))
plt.barh(results_df["Model"], results_df["F1 Score"], color='skyblue')
plt.xlabel("F1 Score")
plt.title("Model Comparison for Predicting Health Condition")
plt.gca().invert_yaxis()
# plt.show()

# 7. Best model
best_model = results_df.iloc[0]
print("Best model based on F1 score:\n", best_model)

# Get the trained Decision Tree model from your models dictionary
decision_tree_model = models["Decision Tree"]

# # Save the trained Decision Tree model
# joblib.dump(decision_tree_model, "decision_tree_model.pkl")

# print("Decision Tree model saved successfully as 'decision_tree_model.pkl'")
############################################################################DECISION TREE BEST MODEL#######################@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@############