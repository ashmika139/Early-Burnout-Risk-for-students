import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")
df = pd.read_csv("finalized_dataset.csv")

print("Shape of dataset:", df.shape)
print("\n Columns:")
print(df.columns)

print("\n Data Types:")
print(df.dtypes)

print("\n Missing Values:")
print(df.isnull().sum())
print("\n Duplicate Rows:", df.duplicated().sum())

print("\n Statistical Summary:")
print(df.describe())

categorical_cols = df.select_dtypes(include='object').columns
print("\n Categorical Columns:", categorical_cols)
for col in categorical_cols:
    print(f"\n Value counts for {col}:")
    print(df[col].value_counts())

#dataset balance
target_col = "burnout_level"
print("\n Burnout Level Distribution:")
print(df[target_col].value_counts())
print("\n Percentage Distribution:")
print(df[target_col].value_counts(normalize=True) * 100)

#Visualization
#Target Distribution
plt.figure(figsize=(6,4))
sns.countplot(x=target_col, data=df)
plt.title("Burnout Level Distribution")
plt.show()

#Categorical vs Target 
for col in categorical_cols:
    if col != target_col:
        plt.figure(figsize=(6,4))
        sns.countplot(x=col, hue=target_col, data=df)
        plt.title(f"{col} vs Burnout Level")
        plt.xticks(rotation=45)
        plt.show()

#histogram 
df.hist(figsize=(12,10))
plt.suptitle("Feature Distributions")
plt.show()
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

#grouped analysis
print("\n Mean values grouped by burnout level:")

#outliers
numeric_cols = df.select_dtypes(include=np.number).columns
for col in numeric_cols:
    plt.figure(figsize=(6,4))
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot of {col}")
    plt.show()

# Final insight 
print("\nQUICK INSIGHTS OF THE DATASET:")
balance = df[target_col].value_counts(normalize=True) * 100
if max(balance) - min(balance) < 20:
    print(" Dataset is BALANCED")
else:
    print(" Dataset is IMBALANCED")
print("\nTop Correlations with Burnout (approx idea):")
print(df.corr(numeric_only=True)[numeric_cols].mean().sort_values(ascending=False).head())