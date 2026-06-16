import json
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
)
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

RANDSEED = 42
TARGET_COL = "Tingkat_kecanduan_media_sosial_Anda"
CLASS_NAMES = {0: "Rendah", 1: "Sedang", 2: "Tinggi"}
TEST_SIZE = 0.2
CV_SPLITS = 5


def save_split(X_train, X_test, y_train, y_test):
    train_data = pd.concat([X_train, y_train], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)
    train_data.to_csv("train.csv", index=False)
    test_data.to_csv("test.csv", index=False)


def build_pipeline(scaler, classifier, k_features):
    steps = [("feature_selection", SelectKBest(score_func=f_classif, k=k_features))]
    if scaler is not None:
        steps.append(("scaler", clone(scaler)))
    steps.append(("classifier", clone(classifier)))
    return Pipeline(steps)


def selected_features(pipeline, feature_names):
    selector = pipeline.named_steps["feature_selection"]
    return [name for name, selected in zip(feature_names, selector.get_support()) if selected]


def feature_score_table(pipeline, feature_names):
    selector = pipeline.named_steps["feature_selection"]
    scores = np.nan_to_num(selector.scores_, nan=0.0, posinf=0.0, neginf=0.0)
    ranked = sorted(
        [{"feature": name, "score": round(float(score), 6)} for name, score in zip(feature_names, scores)],
        key=lambda item: item["score"],
        reverse=True,
    )
    return ranked


def model_selection_score(cv_mean, cv_std, train_f1):
    overfit_penalty = max(0.0, train_f1 - cv_mean)
    return cv_mean - (0.35 * cv_std) - (0.25 * overfit_penalty)


data = pd.read_csv("data_gadget.csv")
X = data.drop(columns=[TARGET_COL])
y = data[TARGET_COL]
feature_names = list(X.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDSEED,
    stratify=y,
)
save_split(X_train, X_test, y_train, y_test)

print(f"Total data: {len(data)} | Features: {len(feature_names)}")
print(f"Train: {len(X_train)} ({1 - TEST_SIZE:.0%}) | Test: {len(X_test)} ({TEST_SIZE:.0%})")
print(f"Train dist:\n{y_train.value_counts().sort_index().to_string()}")
print(f"Test dist:\n{y_test.value_counts().sort_index().to_string()}")

classifiers = {
    "LogisticRegression": LogisticRegression(
        max_iter=2000,
        C=0.5,
        class_weight="balanced",
        random_state=RANDSEED,
    ),
    "GaussianNB": GaussianNB(),
    "KNN": KNeighborsClassifier(n_neighbors=7, weights="distance"),
    "DecisionTree": DecisionTreeClassifier(
        random_state=RANDSEED,
        max_depth=3,
        min_samples_leaf=4,
        class_weight="balanced",
    ),
    "SVM": SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale",
        probability=True,
        class_weight="balanced",
        random_state=RANDSEED,
    ),
    "SGD": SGDClassifier(
        loss="modified_huber",
        alpha=0.001,
        max_iter=2000,
        tol=1e-3,
        class_weight="balanced",
        random_state=RANDSEED,
    ),
    "MLP": MLPClassifier(
        hidden_layer_sizes=(24,),
        alpha=0.01,
        early_stopping=True,
        validation_fraction=0.2,
        max_iter=1500,
        random_state=RANDSEED,
    ),
    "RandomForest": RandomForestClassifier(
        n_estimators=200,
        max_depth=4,
        min_samples_leaf=3,
        class_weight="balanced_subsample",
        random_state=RANDSEED,
    ),
    "GradientBoosting": GradientBoostingClassifier(
        n_estimators=60,
        learning_rate=0.05,
        max_depth=2,
        min_samples_leaf=3,
        subsample=0.8,
        random_state=RANDSEED,
    ),
    "Bagging": BaggingClassifier(
        estimator=DecisionTreeClassifier(
            max_depth=3,
            min_samples_leaf=4,
            class_weight="balanced",
            random_state=RANDSEED,
        ),
        n_estimators=80,
        max_samples=0.8,
        random_state=RANDSEED,
    ),
    "AdaBoost": AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1, min_samples_leaf=3, random_state=RANDSEED),
        n_estimators=60,
        learning_rate=0.5,
        random_state=RANDSEED,
    ),
    "Stacking": StackingClassifier(
        estimators=[
            (
                "lr",
                LogisticRegression(
                    max_iter=1000,
                    C=0.5,
                    class_weight="balanced",
                    random_state=RANDSEED,
                ),
            ),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=100,
                    max_depth=3,
                    min_samples_leaf=4,
                    class_weight="balanced_subsample",
                    random_state=RANDSEED,
                ),
            ),
            (
                "svm",
                SVC(
                    kernel="rbf",
                    C=1.0,
                    probability=True,
                    class_weight="balanced",
                    random_state=RANDSEED,
                ),
            ),
        ],
        final_estimator=LogisticRegression(
            max_iter=1000,
            C=0.5,
            class_weight="balanced",
            random_state=RANDSEED,
        ),
        cv=3,
    ),
}

scalers = {
    "NoScaler": None,
    "MinMaxScaler": MinMaxScaler(),
    "StandardScaler": StandardScaler(),
}

k_options = [5, 8, 10, 12, 15, "all"]
cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDSEED)
results = []

print("\n" + "=" * 118)
print(
    f"{'Model':<20} {'Scaler':<16} {'K':>5} {'CV-F1':>8} {'CV-Std':>8} "
    f"{'Train-F1':>9} {'Test-F1':>8} {'Test-Acc':>8} {'Gap':>8} {'Score':>8}"
)
print("=" * 118)

for scaler_name, scaler in scalers.items():
    for clf_name, clf in classifiers.items():
        for k_features in k_options:
            pipe = build_pipeline(scaler, clf, k_features)

            try:
                cv_scores = cross_val_score(
                    pipe,
                    X_train,
                    y_train,
                    cv=cv,
                    scoring="f1_macro",
                    error_score=np.nan,
                )
                if np.isnan(cv_scores).any():
                    continue

                pipe.fit(X_train, y_train)
                y_train_pred = pipe.predict(X_train)
                y_test_pred = pipe.predict(X_test)

                train_f1 = f1_score(y_train, y_train_pred, average="macro")
                test_f1 = f1_score(y_test, y_test_pred, average="macro")
                test_acc = accuracy_score(y_test, y_test_pred)
                test_precision = precision_score(y_test, y_test_pred, average="macro", zero_division=0)
                test_recall = recall_score(y_test, y_test_pred, average="macro", zero_division=0)
                test_balanced_acc = balanced_accuracy_score(y_test, y_test_pred)
                cv_mean = float(cv_scores.mean())
                cv_std = float(cv_scores.std())
                gap = abs(train_f1 - test_f1)
                score = model_selection_score(cv_mean, cv_std, train_f1)

                results.append(
                    {
                        "model": clf_name,
                        "scaler": scaler_name,
                        "k_features": k_features,
                        "cv_mean_f1": cv_mean,
                        "cv_std_f1": cv_std,
                        "train_f1_macro": train_f1,
                        "test_f1_macro": test_f1,
                        "test_accuracy": test_acc,
                        "test_precision_macro": test_precision,
                        "test_recall_macro": test_recall,
                        "test_balanced_accuracy": test_balanced_acc,
                        "gap_f1": gap,
                        "selection_score": score,
                        "pipeline": pipe,
                    }
                )

                print(
                    f"{clf_name:<20} {scaler_name:<16} {str(k_features):>5} "
                    f"{cv_mean:>8.4f} {cv_std:>8.4f} {train_f1:>9.4f} "
                    f"{test_f1:>8.4f} {test_acc:>8.4f} {gap:>8.4f} {score:>8.4f}"
                )
            except Exception as exc:
                print(f"{clf_name:<20} {scaler_name:<16} {str(k_features):>5} skipped: {exc}")

print("=" * 118)

if not results:
    raise RuntimeError("Tidak ada model yang berhasil dilatih.")

results_df = pd.DataFrame([{k: v for k, v in r.items() if k != "pipeline"} for r in results])
results_df = results_df.sort_values(["selection_score", "test_f1_macro"], ascending=False)

print("\nTop 10 Models (ranked by CV stability and overfit penalty):")
print(results_df.head(10).to_string(index=False))

best = max(results, key=lambda item: (item["selection_score"], item["test_f1_macro"]))
best_pipe = best["pipeline"]
best_selected_features = selected_features(best_pipe, feature_names)
best_feature_scores = feature_score_table(best_pipe, feature_names)

print(f"\n{'=' * 70}")
print(f"BEST MODEL: {best['model']} + {best['scaler']} + SelectKBest(k={best['k_features']})")
print(f"Selected features ({len(best_selected_features)} of {len(feature_names)}):")
for idx, feature in enumerate(best_selected_features, start=1):
    print(f"{idx}. {feature}")
print(
    f"CV F1: {best['cv_mean_f1']:.4f} +/- {best['cv_std_f1']:.4f} | "
    f"Train F1: {best['train_f1_macro']:.4f} | Test F1: {best['test_f1_macro']:.4f} | "
    f"Test Accuracy: {best['test_accuracy']:.4f} | Gap F1: {best['gap_f1']:.4f}"
)
print(f"{'=' * 70}")

y_pred = best_pipe.predict(X_test)
print("\nClassification Report (Test):")
print(
    classification_report(
        y_test,
        y_pred,
        labels=list(CLASS_NAMES.keys()),
        target_names=list(CLASS_NAMES.values()),
        zero_division=0,
    )
)
print("Confusion Matrix (Test):")
print(confusion_matrix(y_test, y_pred, labels=list(CLASS_NAMES.keys())))

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
    "feature_selection": "SelectKBest(f_classif)",
    "k_features": best["k_features"],
    "selected_feature_count": len(best_selected_features),
    "selected_feature_names": best_selected_features,
    "feature_scores": best_feature_scores,
    "test_size": TEST_SIZE,
    "train_size": len(X_train),
    "test_rows": len(X_test),
    "cv_splits": CV_SPLITS,
    "test_accuracy": round(best["test_accuracy"], 4),
    "test_f1_macro": round(best["test_f1_macro"], 4),
    "test_precision_macro": round(best["test_precision_macro"], 4),
    "test_recall_macro": round(best["test_recall_macro"], 4),
    "test_balanced_accuracy": round(best["test_balanced_accuracy"], 4),
    "train_f1_macro": round(best["train_f1_macro"], 4),
    "cv_mean_f1": round(best["cv_mean_f1"], 4),
    "cv_std_f1": round(best["cv_std_f1"], 4),
    "gap_f1": round(best["gap_f1"], 4),
    "selection_score": round(best["selection_score"], 4),
    "class_names": CLASS_NAMES,
    "feature_names": feature_names,
    "feature_info": feature_info,
    "target_column": TARGET_COL,
    "top_models": results_df.head(10).round(4).to_dict(orient="records"),
}

with open("model_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("Metadata saved: model_metadata.json")
