from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from extract_features import extract_file_features

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "genre_model.pkl"
FEATURES_PATH = BASE_DIR / "features" / "features.csv"


def _build_feature_array(file_path: str) -> pd.DataFrame:
	features = extract_file_features(Path(file_path))
	feature_columns = pd.read_csv(FEATURES_PATH, nrows=1).drop(columns=["label"]).columns.tolist()
	return pd.DataFrame(np.array([features]), columns=feature_columns)


def predict_genre(file_path: str) -> str:
	model = joblib.load(MODEL_PATH)
	feature_array = _build_feature_array(file_path)
	return str(model.predict(feature_array)[0])


def predict_genre_proba(file_path: str) -> dict[str, float]:
	"""Return a dict mapping each genre label to its predicted probability."""
	model = joblib.load(MODEL_PATH)
	feature_array = _build_feature_array(file_path)
	probas = model.predict_proba(feature_array)[0]
	classes = model.classes_
	return {str(cls): float(prob) for cls, prob in zip(classes, probas)}


if __name__ == "__main__":
	sample_file = BASE_DIR / "dataset" / "genres_original" / "blues" / "blues.00000.wav"
	if sample_file.exists():
		print(f"Prediction for {sample_file.name}: {predict_genre(str(sample_file))}")
	else:
		print("Model loaded. Call predict_genre(file_path) with a valid .wav file path.")
