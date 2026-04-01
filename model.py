import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("finalized_dataset.csv")
print("Shape:", df.shape)
# Encoding categorical variables
df['stress_level'] = df['stress_level'].map({
    'Low': 0,
    'Medium': 1,
    'High': 2
})
df['burnout_level'] = df['burnout_level'].map({
    'Low': 0,
    'Medium': 1,
    'High': 2
})
df = pd.get_dummies(df, columns=['course'], drop_first=True)

# Features and target
X = df.drop('burnout_level', axis=1)
y = df['burnout_level']

# Train-Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y   
)

#SCALING
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVM": SVC(kernel='rbf'),
    "KNN": KNeighborsClassifier(n_neighbors=7),  # slightly better than 5
    "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=42)
}

#TRAIN & EVALUATE
results = {}
for name, model in models.items():
    print("\n==============================")
    print(f"{name}")
    print("==============================")
    if name in ["SVM", "KNN"]:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    print("Accuracy:", round(acc, 4))
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred, target_names=['Low','Medium','High']))
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Low','Medium','High'],
                yticklabels=['Low','Medium','High'])
    plt.title(f"{name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()
#MODEL COMPARISON
print("\n====== MODEL COMPARISON ======")
for model_name, acc in results.items():
    print(f"{model_name}: {round(acc,4)}")
best_model = max(results, key=results.get)
print("\n🔥 Best Model:", best_model)