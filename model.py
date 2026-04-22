import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, label_binarize

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Metrics
from sklearn.metrics import (
    accuracy_score, classification_report,
    f1_score, roc_auc_score,
    precision_score, recall_score,
    confusion_matrix, roc_curve, auc
)
#LOAD DATA
df = pd.read_csv("finalized_dataset.csv")
print("Shape:", df.shape)
#FEATURE ENGINEERING
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
#ENCODING
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

    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        random_state=42
    ),

    "SVM": SVC(kernel='linear', probability=True),

    "KNN": KNeighborsClassifier(n_neighbors=9, weights='distance')
}
#TRAIN & EVALUATE
results = []
y_test_bin = label_binarize(y_test, classes=[0, 1, 2])

for name, model in models.items():

    print("\n==============================")
    print(f"{name}")
    print("==============================")

    if name in ["SVM", "KNN", "Logistic Regression"]:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    roc = roc_auc_score(y_test_bin, y_prob, multi_class='ovr')

    results.append({
        "Model": name,
        "Accuracy": round(acc, 2),
        "Precision": round(precision, 2),
        "Recall": round(recall, 2),
        "F1 Score": round(f1, 2),
        "ROC-AUC": round(roc, 2)
    })

    print("Accuracy:", round(acc, 2))
    print("F1 Score:", round(f1, 2))
    print("ROC-AUC:", round(roc, 2))

    print("\nClassification Report:\n")
    print(classification_report(
        y_test, y_pred,
        target_names=['Low', 'Medium', 'High']
    ))
    # CONFUSION MATRIX
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Low','Medium','High'],
                yticklabels=['Low','Medium','High'])
    plt.title(f"{name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()
# 10. ROC CURVE 
plt.figure(figsize=(8,6))
plt.style.use('seaborn-v0_8')

for name, model in models.items():

    if name in ["SVM", "KNN", "Logistic Regression"]:
        model.fit(X_train_scaled, y_train)
        y_prob = model.predict_proba(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)

    fpr, tpr, _ = roc_curve(y_test_bin.ravel(), y_prob.ravel())
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, linewidth=2,
             label=f"{name} (AUC = {roc_auc:.2f})")

# Diagonal line
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison (All Models)")
plt.legend()
plt.grid(True)
plt.show()
#FINAL TABLE
results_df = pd.DataFrame(results)

print("\n====== RESULTS & EVALUATION ======")
print(results_df.sort_values(by="ROC-AUC", ascending=False))