# 1. Import Libraries
import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
# ============================================
# 2. Load Dataset
# ============================================

df = pd.read_csv(r"C:\Users\nabee\Downloads\diabetes.csv")

print("Dataset Preview:\n", df.head())

# ============================================
# 3. Data Preprocessing
# ============================================

# Check missing values
print("\nMissing Values:\n", df.isnull().sum())

# Define features and target
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ============================================
# 4. Train Models
# ============================================

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Logistic Regression (baseline)
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)

# AdaBoost (Boosting)
ab = AdaBoostClassifier()
ab.fit(X_train, y_train)

# ============================================
# 5. Predictions
# ============================================

rf_pred = rf.predict(X_test)
lr_pred = lr.predict(X_test)
ab_pred = ab.predict(X_test)

# ============================================
# 6. Evaluation
# ============================================

print("\n--- Model Accuracy ---")
print("Random Forest:", accuracy_score(y_test, rf_pred))
print("Logistic Regression:", accuracy_score(y_test, lr_pred))
print("AdaBoost:", accuracy_score(y_test, ab_pred))

print("\n--- Random Forest Report ---")
print(classification_report(y_test, rf_pred))

# ============================================
# 7. Save Model
# ============================================

with open("diabetes_model.pkl", "wb") as f:
    pickle.dump(rf, f)

print("\nModel saved successfully!")# 1. Import Libraries
import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# ============================================
# 2. Load Dataset
# ============================================

df = pd.read_csv(r"C:\Users\nabee\Downloads\diabetes.csv")

print("Dataset Preview:\n", df.head())

# ============================================
# 3. Data Preprocessing
# ============================================

# Check missing values
print("\nMissing Values:\n", df.isnull().sum())

# Define features and target
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ============================================
# 4. Train Models
# ============================================

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Logistic Regression (baseline)
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)

# AdaBoost (Boosting)
ab = AdaBoostClassifier()
ab.fit(X_train, y_train)

# ============================================
# 5. Predictions
# ============================================

rf_pred = rf.predict(X_test)
lr_pred = lr.predict(X_test)
ab_pred = ab.predict(X_test)

# ============================================
# 6. Evaluation
# ============================================

print("\n--- Model Accuracy ---")
print("Random Forest:", accuracy_score(y_test, rf_pred))
print("Logistic Regression:", accuracy_score(y_test, lr_pred))
print("AdaBoost:", accuracy_score(y_test, ab_pred))

print("\n--- Random Forest Report ---")
print(classification_report(y_test, rf_pred))

# ============================================
# 7. Save Model
# ============================================

with open("diabetes_model.pkl", "wb") as f:
    pickle.dump(rf, f)

print("\nModel saved successfully!")
