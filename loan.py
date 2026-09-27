import os
import pandas as pd
import numpy as np

# Logistic regression
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Model evaluation tools that may be needed later
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score
)

# Optional visualization packages
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load CSV file (checks local workspace first, then fallback)
csv_path = "SBAcase.11.13.17.csv"
if not os.path.exists(csv_path):
    csv_path = "/data/workspace_files/SBAcase.11.13.17.csv"

df = pd.read_csv(csv_path)

# View the first rows
print("=== Head (2 rows) ===")
print(df.head(2))

# 2. Dataset counts
print("\nDataset shape:", df.shape)
print("\nSelected counts:")
print(df["Selected"].value_counts().sort_index())

print("\nDefault counts:")
print(df["Default"].value_counts().sort_index())

# 3. Numeric conversion & missing values check
model_columns = [
    "Selected",
    "Default",
    "New",
    "RealEstate",
    "DisbursementGross",
    "Portion",
    "Recession"
]

for column in model_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

print("\nMissing values:")
print(df[model_columns].isna().sum())

# 4. Train / Test split
train = df.loc[df["Selected"] == 1].copy()
test = df.loc[df["Selected"] == 0].copy()

print("\nTraining observations:", len(train))
print("Testing observations:", len(test))

# 5. Model 7a
predictors_7a = [
    "New",
    "RealEstate",
    "DisbursementGross",
    "Portion",
    "Recession"
]

X_train_7a = sm.add_constant(
    train[predictors_7a],
    has_constant="add"
)

y_train = train["Default"]

model_7a = sm.GLM(
    y_train,
    X_train_7a,
    family=sm.families.Binomial()
)

result_7a = model_7a.fit()

print("\n=== Model 7a Summary ===")
print(result_7a.summary())

table_7a = pd.DataFrame({
    "Estimate": result_7a.params,
    "Standard Error": result_7a.bse,
    "Wald Chi-Square": (result_7a.params / result_7a.bse) ** 2,
    "p-value": result_7a.pvalues
})

print("\n=== Table 7a ===")
print(table_7a.round(4))

# 6. Model 8
predictors_8 = [
    "RealEstate",
    "Portion",
    "Recession"
]

X_train_8 = sm.add_constant(
    train[predictors_8],
    has_constant="add"
)

model_8 = sm.GLM(
    y_train,
    X_train_8,
    family=sm.families.Binomial()
)

result_8 = model_8.fit()

print("\n=== Model 8 Summary ===")
print(result_8.summary())

table_8 = pd.DataFrame({
    "Estimate": result_8.params,
    "Standard Error": result_8.bse,
    "Wald Chi-Square": (result_8.params / result_8.bse) ** 2,
    "p-value": result_8.pvalues
})

print("\n=== Table 8 ===")
print(table_8.round(4))

# 7. Model 8 Predictions on Test Data
X_test_8 = sm.add_constant(
    test[predictors_8],
    has_constant="add"
)

test["PredictedProbability"] = result_8.predict(X_test_8)

print("\n=== Test Predictions (First 5) ===")
print(test[
    ["Default", "RealEstate", "Portion",
     "Recession", "PredictedProbability"]
].head())

test["PredictedDefault"] = (
    test["PredictedProbability"] >= 0.50
).astype(int)

test["Classification"] = np.where(
    test["PredictedDefault"] == 1,
    "Higher risk",
    "Lower risk"
)

test["ActualStatus"] = np.where(
    test["Default"] == 1,
    "Charged off",
    "Paid in full"
)

# 8. Confusion Table (Table 9)
table_9 = pd.crosstab(
    test["Classification"],
    test["ActualStatus"]
)

table_9 = table_9.reindex(
    index=["Higher risk", "Lower risk"],
    columns=["Charged off", "Paid in full"]
)

print("\n=== Table 9 ===")
print(table_9)

# 9. Misclassification Metrics
misclassified = (
    test["PredictedDefault"] != test["Default"]
).sum()

misclassification_rate = misclassified / len(test)

print("\n=== Misclassification Performance ===")
print("Number misclassified:", misclassified)
print(f"Misclassification rate: {misclassification_rate:.4f}")
print(f"Misclassification percentage: {misclassification_rate:.2%}")
