from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parents[1]
FEATURES_PATH = BASE_DIR / "features" / "features.csv"
MODEL_PATH = BASE_DIR / "models" / "genre_model.pkl"
CONFUSION_MATRIX_PATH = BASE_DIR / "models" / "confusion_matrix.png"


def main() -> None:
    df = pd.read_csv(FEATURES_PATH)

    X = df.drop("label", axis=1)
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)),
    ])
    pipeline.fit(X_train, y_train)

    accuracy = pipeline.score(X_test, y_test)
    print(f"Accuracy: {accuracy:.4f}")

    y_pred = pipeline.predict(X_test)
    labels = sorted(y.unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Genre Classification Confusion Matrix")
    plt.tight_layout()

    CONFUSION_MATRIX_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(CONFUSION_MATRIX_PATH, dpi=200)
    plt.close()
    print(f"Confusion matrix saved to: {CONFUSION_MATRIX_PATH}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
