from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from extract_features import extract_file_features

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "genre_model.pkl"
FEATURES_PATH = BASE_DIR / "features" / "features.csv"


def _load_model_bundle() -> dict:
	artifact = joblib.load(MODEL_PATH)

	# Backward compatibility: older artifacts may store only the model object.
	if not isinstance(artifact, dict) or "model" not in artifact:
		model = artifact
		feature_columns = pd.read_csv(FEATURES_PATH, nrows=1).drop(columns=["label"]).columns.tolist()
		return {
			"model_name": type(model).__name__,
			"model": model,
			"label_encoder": None,
			"feature_columns": feature_columns,
		}

	return artifact



def _build_feature_array(file_path: str) -> pd.DataFrame:
	features = extract_file_features(Path(file_path))
	bundle = _load_model_bundle()
	feature_columns = bundle.get("feature_columns")
	if not feature_columns:
		feature_columns = pd.read_csv(FEATURES_PATH, nrows=1).drop(columns=["label"]).columns.tolist()
	
	# Handle feature mismatch: slice to expected number of columns
	if len(features) > len(feature_columns):
		features = features[:len(feature_columns)]
	
	return pd.DataFrame(np.array([features]), columns=feature_columns)

def predict_genre(file_path: str) -> str:
	bundle = _load_model_bundle()
	model = bundle["model"]
	label_encoder = bundle.get("label_encoder")
	feature_array = _build_feature_array(file_path)
	prediction = model.predict(feature_array)[0]

	if label_encoder is not None:
		# For XGBoost or other models with encoded labels
		try:
			return str(label_encoder.inverse_transform([int(prediction)])[0])
		except (ValueError, TypeError):
			# If prediction is already a string, return as-is
			return str(prediction)

	return str(prediction)


def predict_genre_proba(file_path: str) -> dict[str, float]:
	"""Return a dict mapping each genre label to its predicted probability."""
	bundle = _load_model_bundle()
	model = bundle["model"]
	label_encoder = bundle.get("label_encoder")
	feature_array = _build_feature_array(file_path)
	
	# Pipeline.predict_proba() always returns shape (n_samples, n_classes)
	probas = model.predict_proba(feature_array)[0]
	
	if label_encoder is not None:
		classes = label_encoder.classes_
	else:
		classes = model.named_steps['clf'].classes_ if hasattr(model, 'named_steps') else model.classes_

	return {str(cls): float(prob) for cls, prob in zip(classes, probas)}


if __name__ == "__main__":
	sample_file = BASE_DIR / "dataset" / "genres_original" / "blues" / "blues.00000.wav"
	if sample_file.exists():
		print(f"Prediction for {sample_file.name}: {predict_genre(str(sample_file))}")
	else:
		print("Model loaded. Call predict_genre(file_path) with a valid .wav file path.")
