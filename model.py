import json
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import (
    GradientBoostingClassifier, BaggingClassifier,
    AdaBoostClassifier, RandomForestClassifier, StackingClassifier
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)
from sklearn.utils import resample
import warnings
warnings.filterwarnings("ignore")

RANDSEED = 42
TARGET_COL = "Tingkat_kecanduan_media_sosial_Anda"
CLASS_NAMES = {0: "Rendah", 1: "Sedang", 2: "Tinggi"}

data = pd.read_csv("data_gadget.csv")
X = data.drop(columns=[TARGET_COL])
y = data[TARGET_COL]
feature_names = list(X.columns)

X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.3, random_state=RANDSEED, stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.2, random_state=RANDSEED, stratify=y_train_full
)

print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
print(f"Train dist:\n{y_train.value_counts().sort_index().to_string()}")
print(f"Test dist:\n{y_test.value_counts().sort_index().to_string()}")


def upsample_training(X_tr, y_tr):
    train_data = pd.concat([X_tr, y_tr], axis=1)
    max_size = y_tr.value_counts().max()
    upsampled = []
    for cls in y_tr.unique():
        subset = train_data[train_data[TARGET_COL] == cls]
        if len(subset) < max_size:
            subset = resample(subset, replace=True, n_samples=max_size, random_state=RANDSEED)
        upsampled.append(subset)
    result = pd.concat(upsampled).sample(frac=1, random_state=RANDSEED)
    return result.drop(columns=[TARGET_COL]), result[TARGET_COL]


X_train_res, y_train_res = upsample_training(X_train, y_train)
print(f"\nUpsampled train: {len(X_train_res)}")
print(f"Upsampled dist:\n{y_train_res.value_counts().sort_index().to_string()}")

classifiers = {
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=RANDSEED),
    "GaussianNB": GaussianNB(),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "DecisionTree": DecisionTreeClassifier(random_state=RANDSEED, max_depth=5),
    "SVM": SVC(kernel="rbf", probability=True, random_state=RANDSEED),
    "SGD": SGDClassifier(loss="modified_huber", max_iter=1000, random_state=RANDSEED),
    "MLP": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=RANDSEED),
    "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=RANDSEED),
    "GradientBoosting": GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=RANDSEED),
    "Bagging": BaggingClassifier(n_estimators=50, random_state=RANDSEED),
    "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=RANDSEED),
    "Stacking": StackingClassifier(
        estimators=[
            ("rf", RandomForestClassifier(n_estimators=50, max_depth=4, random_state=RANDSEED)),
            ("gb", GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=RANDSEED)),
            ("svm", SVC(kernel="rbf", probability=True, random_state=RANDSEED)),
        ],
        final_estimator=LogisticRegression(max_iter=500, random_state=RANDSEED),
        cv=3
    ),
}

scalers = {
    "MinMaxScaler": MinMaxScaler(),
    "StandardScaler": StandardScaler(),
}

results = []

print("\n" + "=" * 90)
print(f"{'Model':<25} {'Scaler':<16} {'CV-Mean':>8} {'CV-Std':>8} {'Train':>8} {'Val':>8} {'Test':>8} {'Gap':>8}")
print("=" * 90)

for scaler_name, scaler in scalers.items():
    for clf_name, clf in classifiers.items():
        from sklearn.base import clone
        pipe = Pipeline([
            ("scaler", clone(scaler)),
            ("classifier", clone(clf))
        ])

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDSEED)

        try:
            cv_scores = cross_val_score(pipe, X_train_res, y_train_res, cv=cv, scoring="f1_macro")
        except Exception:
            cv_scores = np.array([0.0])

        pipe.fit(X_train_res, y_train_res)

        train_acc = accuracy_score(y_train_res, pipe.predict(X_train_res))
        val_acc = accuracy_score(y_val, pipe.predict(X_val))
        test_acc = accuracy_score(y_test, pipe.predict(X_test))
        test_f1 = f1_score(y_test, pipe.predict(X_test), average="macro")
        gap = abs(train_acc - test_acc)

        results.append({
            "model": clf_name,
            "scaler": scaler_name,
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
            "train_acc": train_acc,
            "val_acc": val_acc,
            "test_acc": test_acc,
            "test_f1": test_f1,
            "gap": gap,
            "pipeline": pipe,
        })

        print(f"{clf_name:<25} {scaler_name:<16} {cv_scores.mean():>8.4f} {cv_scores.std():>8.4f} "
              f"{train_acc:>8.4f} {val_acc:>8.4f} {test_acc:>8.4f} {gap:>8.4f}")

print("=" * 90)

results_df = pd.DataFrame([{k: v for k, v in r.items() if k != "pipeline"} for r in results])
results_df["score"] = results_df["test_f1"] * 0.6 + results_df["cv_mean"] * 0.3 - results_df["gap"] * 0.1
results_df = results_df.sort_values("score", ascending=False)

print("\nTop 5 Models:")
print(results_df.head().to_string(index=False))

best_idx = results_df.index[0]
best = results[best_idx]
best_pipe = best["pipeline"]

print(f"\n{'=' * 60}")
print(f"BEST MODEL: {best['model']} + {best['scaler']}")
print(f"Test Accuracy: {best['test_acc']:.4f} | Test F1: {best['test_f1']:.4f} | Gap: {best['gap']:.4f}")
print(f"{'=' * 60}")

y_pred = best_pipe.predict(X_test)
print("\nClassification Report (Test):")
print(classification_report(y_test, y_pred, target_names=list(CLASS_NAMES.values())))
print("Confusion Matrix (Test):")
print(confusion_matrix(y_test, y_pred))

joblib.dump(best_pipe, "best_model.joblib")
print("\nModel saved: best_model.joblib")

feature_info = {}
for col in feature_names:
    unique = sorted(data[col].unique().tolist())
    feature_info[col] = {
        "min": float(data[col].min()),
        "max": float(data[col].max()),
        "unique_values": [float(v) for v in unique],
        "dtype": str(data[col].dtype),
    }

metadata = {
    "model_name": best["model"],
    "scaler_name": best["scaler"],
    "test_accuracy": round(best["test_acc"], 4),
    "test_f1_macro": round(best["test_f1"], 4),
    "train_accuracy": round(best["train_acc"], 4),
    "cv_mean_f1": round(best["cv_mean"], 4),
    "gap": round(best["gap"], 4),
    "class_names": CLASS_NAMES,
    "feature_names": feature_names,
    "feature_info": feature_info,
    "target_column": TARGET_COL,
}

with open("model_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("Metadata saved: model_metadata.json")
