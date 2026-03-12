import os
from pathlib import Path

import librosa
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
DATASET_PATH = BASE_DIR / "dataset" / "genres_original"
OUTPUT_PATH = BASE_DIR / "features" / "features.csv"


def extract_file_features(file_path: Path) -> list[float]:
	# Load fixed duration for consistent feature vectors across files.
	y, sr = librosa.load(file_path, duration=30)

	# 40 per-coefficient MFCC means + stds for richer timbral encoding.
	mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
	mfcc_means = np.mean(mfcc, axis=1).tolist()
	mfcc_stds = np.std(mfcc, axis=1).tolist()

	spectral_centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
	spectral_rolloff = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))
	zcr = float(np.mean(librosa.feature.zero_crossing_rate(y=y)))

	# 12 chroma means — captures pitch-class distribution.
	chroma = librosa.feature.chroma_stft(y=y, sr=sr)
	chroma_means = np.mean(chroma, axis=1).tolist()

	# 7 spectral contrast bands — separates harmonic and percussive content.
	contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
	contrast_means = np.mean(contrast, axis=1).tolist()

	tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
	tempo_value = float(np.asarray(tempo).reshape(-1)[0])

	return mfcc_means + mfcc_stds + [spectral_centroid, spectral_rolloff, zcr] + chroma_means + contrast_means + [tempo_value]


def main() -> None:
	if not DATASET_PATH.exists():
		raise FileNotFoundError(f"Dataset path not found: {DATASET_PATH}")

	features: list[list[float]] = []
	labels: list[str] = []

	for genre in sorted(os.listdir(DATASET_PATH)):
		genre_folder = DATASET_PATH / genre
		if not genre_folder.is_dir():
			continue

		for file_name in sorted(os.listdir(genre_folder)):
			if not file_name.lower().endswith(".wav"):
				continue

			file_path = genre_folder / file_name

			try:
				row = extract_file_features(file_path)
			except Exception as exc:
				print(f"Skipping {file_path.name}: {exc}")
				continue

			features.append(row)
			labels.append(genre)

	mfcc_mean_cols = [f"mfcc_mean_{i+1}" for i in range(40)]
	mfcc_std_cols = [f"mfcc_std_{i+1}" for i in range(40)]
	chroma_cols = [f"chroma_{i+1}" for i in range(12)]
	contrast_cols = [f"contrast_{i+1}" for i in range(7)]
	columns = (mfcc_mean_cols + mfcc_std_cols
			   + ["spectral_centroid", "spectral_rolloff", "zero_crossing_rate"]
			   + chroma_cols + contrast_cols + ["tempo"])

	df = pd.DataFrame(features, columns=columns)
	df["label"] = labels

	OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
	df.to_csv(OUTPUT_PATH, index=False)

	print(f"Feature extraction complete! Rows: {len(df)}")
	print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
	main()
