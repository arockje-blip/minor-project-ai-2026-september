# ============================================================
# Project 1 : Cardiovascular Disease Prediction
# Run:  python cardio_prediction.py        (cardio_train.csv in same folder)
# Colab: upload cardio_train.csv, paste this code, run.
# Plots are shown AND saved in the "plots" folder.
# ============================================================
import os, time, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, classification_report)

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
os.makedirs("plots", exist_ok=True)

def save(name):
    plt.tight_layout()
    plt.savefig(f"plots/{name}.png", dpi=110)
    plt.show()

# ------------------------------------------------------------
# STEP 1 : LOAD DATA
# ------------------------------------------------------------
df = pd.read_csv("cardio_train.csv", sep=None, engine="python")
print("Shape:", df.shape)
print(df.head())
print(df.info())
print("Missing values:\n", df.isnull().sum())
print("Duplicate rows:", df.duplicated().sum())

# ------------------------------------------------------------
# STEP 2 : DATA PRE-PROCESSING
# ------------------------------------------------------------
df = df.drop(columns="id")                       # id is not useful
df["age"] = (df["age"] / 365.25).round(1)        # age is in days -> convert to years

# Remove impossible blood pressure values (e.g. 16020/11000, negatives)
before = len(df)
df = df[(df["ap_hi"].between(70, 250)) & (df["ap_lo"].between(40, 200)) & (df["ap_hi"] > df["ap_lo"])]
print("Rows removed (wrong blood pressure):", before - len(df))

# Remove height / weight outliers (outside 0.5% - 99.5%)
before = len(df)
for col in ["height", "weight"]:
    low, high = df[col].quantile([0.005, 0.995])
    df = df[df[col].between(low, high)]
print("Rows removed (height/weight outliers):", before - len(df))

# New features
df["bmi"] = df["weight"] / (df["height"] / 100) ** 2
df["pulse_pressure"] = df["ap_hi"] - df["ap_lo"]
df = df.reset_index(drop=True)
print("Final shape:", df.shape)
print(df.describe().T)

# ------------------------------------------------------------
# STEP 3 : DATA ANALYSIS & VISUALIZATION
# ------------------------------------------------------------
# 3.1 Histograms of numeric columns
df[["age", "height", "weight", "ap_hi", "ap_lo", "bmi"]].hist(bins=40, figsize=(14, 7), color="#d6336c")
plt.suptitle("Numeric feature distributions")
save("01_histograms")

# 3.2 Target, gender, cholesterol
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
sns.countplot(data=df, x="cardio", ax=ax[0], palette="Set2");                 ax[0].set_title("Cardio (0 = no, 1 = yes)")
sns.countplot(data=df, x="gender", hue="cardio", ax=ax[1], palette="Set2");   ax[1].set_title("Gender (1 = female, 2 = male)")
sns.countplot(data=df, x="cholesterol", hue="cardio", ax=ax[2], palette="Set2"); ax[2].set_title("Cholesterol (1 normal -> 3 high)")
save("02_target_gender_cholesterol")

# 3.3 Lifestyle + vitals
fig, ax = plt.subplots(2, 3, figsize=(15, 8))
for a, col in zip(ax.ravel()[:4], ["gluc", "smoke", "alco", "active"]):
    sns.countplot(data=df, x=col, hue="cardio", ax=a, palette="Set2"); a.set_title(col + " vs cardio")
sns.boxplot(data=df, x="cardio", y="ap_hi", ax=ax[1, 1], palette="Set2"); ax[1, 1].set_title("Systolic BP vs cardio")
sns.boxplot(data=df, x="cardio", y="bmi", ax=ax[1, 2], palette="Set2");   ax[1, 2].set_title("BMI vs cardio")
save("03_lifestyle_vitals")

# 3.4 Age effect
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
sns.kdeplot(data=df, x="age", hue="cardio", fill=True, ax=ax[0], palette="Set2"); ax[0].set_title("Age distribution")
age_band = pd.cut(df["age"], [29, 40, 45, 50, 55, 60, 65])
df.groupby(age_band)["cardio"].mean().plot(kind="bar", ax=ax[1], color="#d6336c")
ax[1].set_title("Disease rate by age band"); ax[1].set_ylabel("rate")
save("04_age")

# 3.5 Disease rate by BP group and BMI class
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
bp_group = pd.cut(df["ap_hi"], [0, 119, 129, 139, 300], labels=["<120", "120-129", "130-139", "140+"])
df.groupby(bp_group)["cardio"].mean().plot(kind="bar", ax=ax[0], color="#d6336c"); ax[0].set_title("Disease rate by systolic BP")
bmi_class = pd.cut(df["bmi"], [0, 18.5, 25, 30, 35, 100], labels=["under", "normal", "over", "obese", "severe"])
df.groupby(bmi_class)["cardio"].mean().plot(kind="bar", ax=ax[1], color="#d6336c"); ax[1].set_title("Disease rate by BMI class")
save("05_bp_bmi_rate")

# ------------------------------------------------------------
# STEP 4 : CORRELATION MATRIX
# ------------------------------------------------------------
corr = df.corr()
plt.figure(figsize=(11, 9))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation matrix")
save("06_correlation_matrix")
print("\nCorrelation with cardio:\n", corr["cardio"].drop("cardio").sort_values(ascending=False).round(3))

# ------------------------------------------------------------
# STEP 5 : TRAIN / TEST SPLIT + SCALING
# ------------------------------------------------------------
X = df.drop(columns="cardio")
y = df["cardio"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# ------------------------------------------------------------
# STEP 6 : MACHINE LEARNING MODELS
# ------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "KNN":                 KNeighborsClassifier(n_neighbors=25),
    "Decision Tree":       DecisionTreeClassifier(max_depth=6, random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=150, max_depth=10, min_samples_leaf=20,
                                                  n_jobs=-1, random_state=42),
    "SVM":                 SVC(kernel="rbf", C=1, random_state=42),
    "Naive Bayes":         GaussianNB(),          # extra, taught in class 11
}

results, preds = [], {}
for name, model in models.items():
    start = time.time()
    if name == "SVM":
        # SVM is slow on 55k rows, so train on a 10,000 row sample
        idx = np.random.RandomState(42).choice(len(X_train_s), 10000, replace=False)
        model.fit(X_train_s[idx], y_train.iloc[idx])
    else:
        model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)
    preds[name] = y_pred
    results.append({
        "Model": name,
        "Accuracy":  accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall":    recall_score(y_test, y_pred),
        "F1":        f1_score(y_test, y_pred),
        "Seconds":   round(time.time() - start, 1),
    })
    print(f"{name:20s} accuracy = {results[-1]['Accuracy']:.4f}")

results = pd.DataFrame(results).sort_values("Accuracy", ascending=False).reset_index(drop=True)
print("\n", results.round(4).to_string(index=False))

# Accuracy comparison bar chart
plt.figure(figsize=(9, 4))
sns.barplot(data=results, x="Model", y="Accuracy", palette="viridis")
plt.ylim(0.6, 0.8); plt.xticks(rotation=20); plt.title("Model accuracy comparison")
for i, v in enumerate(results["Accuracy"]):
    plt.text(i, v + 0.002, f"{v:.3f}", ha="center")
save("07_accuracy_comparison")

# Confusion matrices
fig, ax = plt.subplots(2, 3, figsize=(15, 8))
for a, (name, y_pred) in zip(ax.ravel(), preds.items()):
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues", cbar=False, ax=a)
    a.set_title(name); a.set_xlabel("Predicted"); a.set_ylabel("Actual")
save("08_confusion_matrices")

# ------------------------------------------------------------
# STEP 7 : FINAL MODEL (best accuracy)
# ------------------------------------------------------------
best_name = results.loc[0, "Model"]
best_model = models[best_name]
print("\nBest model:", best_name)
print(classification_report(y_test, preds[best_name], target_names=["No disease", "Disease"]))

# Feature importance from Random Forest
rf = models["Random Forest"]
imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values()
imp.plot(kind="barh", figsize=(7, 5), color="#d6336c"); plt.title("Feature importance (Random Forest)")
save("09_feature_importance")

joblib.dump({"model": best_model, "scaler": scaler, "features": list(X.columns)}, "cardio_model.joblib")
print("Saved final model -> cardio_model.joblib")

# ------------------------------------------------------------
# STEP 8 : PREDICT FOR A NEW PATIENT
# ------------------------------------------------------------
def predict_patient(age, gender, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active):
    row = pd.DataFrame([{
        "age": age, "gender": gender, "height": height, "weight": weight,
        "ap_hi": ap_hi, "ap_lo": ap_lo, "cholesterol": cholesterol, "gluc": gluc,
        "smoke": smoke, "alco": alco, "active": active,
        "bmi": weight / (height / 100) ** 2, "pulse_pressure": ap_hi - ap_lo}])[X.columns]
    result = best_model.predict(scaler.transform(row))[0]
    return "Heart disease likely" if result == 1 else "No heart disease predicted"

# gender: 1 = female, 2 = male | cholesterol/gluc: 1 normal, 2 above normal, 3 high
print("\nPatient 1:", predict_patient(52, 2, 175, 90, 150, 95, 3, 1, 1, 0, 0))
print("Patient 2:", predict_patient(35, 1, 165, 58, 110, 70, 1, 1, 0, 0, 1))
