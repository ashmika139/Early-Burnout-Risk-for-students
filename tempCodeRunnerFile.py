# ================================
# 1. IMPORT LIBRARIES
# ================================
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("finalized_dataset.csv")
print("Shape:", df.shape)


# FEATURE ENGINEERING
stress_map = {'Low': 1, 'Medium': 2, 'High': 3}

df['burnout_score'] = (
    0.10 * df['anxiety_score'] +
    0.10 * df['depression_score'] +
    0.10 * df['stress_level'].map(stress_map) +
    0.10 * df['academic_pressure_score']
)

df['burnout_level'] = pd.qcut(
    df['burnout_score'],
    q=3,
    labels=['Low', 'Medium', 'High']
)
df.drop('burnout_score', axis=1, inplace=True)
# ENCODING
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

categorical_cols = df.select_dtypes(include='object').columns
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
#FEATURES & TARGET
X = df.drop('burnout_level', axis=1)
y = df['burnout_level']
#TRAIN-TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
#SCALING 
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
#MODELS
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=5,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        min_samples_leaf=5,
        class_weight='balanced',
        random_state=42
    ),

    "AdaBoost": AdaBoostClassifier(n_estimators=100),

    "SVM": SVC(kernel='linear'),

    "KNN": KNeighborsClassifier(n_neighbors=9, weights='distance')
}
results = []

for name, model in models.items():

    print("\n==============================")
    print(f"{name}")
    print("==============================")
    if name in ["SVM", "KNN"]:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)

    acc = accuracy_score(y_test, y_pred)

    results.append({
        "Model": name,
        "Accuracy": round(acc, 4),
        "CV Accuracy": round(cv_scores.mean(), 4)
    })

    print("Test Accuracy:", round(acc, 4))

    print("\nClassification Report:\n")
    print(classification_report(
        y_test, y_pred,
        target_names=['Low', 'Medium', 'High']
    ))

#FINAL COMPARISON
results_df = pd.DataFrame(results)
print("\n====== FINAL MODEL COMPARISON ======")
print(results_df.sort_values(by="CV Accuracy", ascending=False))