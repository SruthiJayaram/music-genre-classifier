from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier

try:
    import seaborn as sns
except Exception:
    sns = None

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

BASE_DIR = Path(__file__).resolve().parents[1]
FEATURES_PATH = BASE_DIR / "features" / "features.csv"
MODEL_PATH = BASE_DIR / "models" / "genre_model.pkl"
CONFUSION_MATRIX_PATH = BASE_DIR / "models" / "confusion_matrix.png"
MODEL_SCORES_PATH = BASE_DIR / "models" / "model_scores.csv"


def build_models() -> dict[str, Pipeline]:
    models: dict[str, Pipeline] = {
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)),
        ]),
        "SVM (RBF)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", C=10.0, gamma="scale")),
        ]),
        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=7, weights="distance")),
        ]),
        "Gradient Boosting": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(random_state=42)),
        ]),
    }

    if XGBClassifier is not None:
        models["XGBoost"] = Pipeline([
            ("scaler", StandardScaler()),
            (
                "clf",
                XGBClassifier(
                    n_estimators=400,
                    max_depth=6,
                    learning_rate=0.05,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    objective="multi:softprob",
                    eval_metric="mlogloss",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ])

    return models


def main() -> None:
    df = pd.read_csv(FEATURES_PATH)

    X = df.drop("label", axis=1)
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = build_models()
    results: list[dict] = []
    labels_all = sorted(y.unique())

    print("=== Model Comparison ===")
    for name, model in models.items():
        try:
            current = clone(model)
            if name == "XGBoost":
                encoder = LabelEncoder()
                y_train_encoded = encoder.fit_transform(y_train)
                y_test_encoded = encoder.transform(y_test)
                current.fit(X_train, y_train_encoded)
                y_pred_encoded = current.predict(X_test)
                score = accuracy_score(y_test_encoded, y_pred_encoded)
                y_pred_labels = encoder.inverse_transform(y_pred_encoded.astype(int))
                results.append(
                    {
                        "model": name,
                        "accuracy": score,
                        "pipeline": current,
                        "label_encoder": encoder,
                        "y_pred_labels": y_pred_labels,
                    }
                )
            else:
                current.fit(X_train, y_train)
                score = current.score(X_test, y_test)
                y_pred_labels = current.predict(X_test)
                results.append(
                    {
                        "model": name,
                        "accuracy": score,
                        "pipeline": current,
                        "label_encoder": None,
                        "y_pred_labels": y_pred_labels,
                    }
                )
            print(f"{name:<18} Accuracy: {score:.4f}")
        except Exception as exc:
            print(f"{name:<18} Skipped: {exc}")

    if not results:
        raise RuntimeError("No models were successfully trained.")

    results.sort(key=lambda r: r["accuracy"], reverse=True)
    best = results[0]
    best_name = best["model"]
    best_score = best["accuracy"]
    pipeline = best["pipeline"]
    label_encoder = best["label_encoder"]
    y_pred_labels = best["y_pred_labels"]
    print(f"\nBest Model: {best_name} (Accuracy: {best_score:.4f})")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"model": r["model"], "accuracy": r["accuracy"]} for r in results]).to_csv(
        MODEL_SCORES_PATH, index=False
    )
    print(f"Model score table saved to: {MODEL_SCORES_PATH}")

    cm = confusion_matrix(y_test, y_pred_labels, labels=labels_all)

    plt.figure(figsize=(10, 8))
    if sns is not None:
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels_all,
            yticklabels=labels_all,
        )
    else:
        plt.imshow(cm, cmap="Blues")
        plt.colorbar()
        plt.xticks(range(len(labels_all)), labels_all, rotation=45, ha="right")
        plt.yticks(range(len(labels_all)), labels_all)
        for i in range(len(labels_all)):
            for j in range(len(labels_all)):
                plt.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Genre Classification Confusion Matrix")
    plt.tight_layout()

    CONFUSION_MATRIX_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(CONFUSION_MATRIX_PATH, dpi=200)
    plt.close()
    print(f"Confusion matrix saved to: {CONFUSION_MATRIX_PATH}")

    model_bundle = {
        "model_name": best_name,
        "model": pipeline,
        "label_encoder": label_encoder,
        "feature_columns": X.columns.tolist(),
    }
    joblib.dump(model_bundle, MODEL_PATH)
    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
