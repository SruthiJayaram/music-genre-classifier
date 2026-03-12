import os
import sys
from pathlib import Path

import librosa
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
DATASET_PATH = BASE_DIR / "dataset" / "genres_original"
OUTPUT_PATH = BASE_DIR / "features" / "features.csv"
BAD_FILES_LOG = BASE_DIR / "features" / "bad_files.txt"


def scan_dataset(dataset_path: Path = DATASET_PATH) -> list[tuple[str, str, str]]:
	"""Scan every wav file and return a list of (genre, filename, reason) for bad files."""
	bad: list[tuple[str, str, str]] = []
	for genre in sorted(os.listdir(dataset_path)):
		genre_dir = dataset_path / genre
		if not genre_dir.is_dir():
			continue
		for fname in sorted(os.listdir(genre_dir)):
			if not fname.lower().endswith(".wav"):
				continue
			fpath = genre_dir / fname
			try:
				y, sr = librosa.load(fpath, duration=5)
				if len(y) == 0:
					bad.append((genre, fname, "empty signal"))
			except Exception as exc:
				bad.append((genre, fname, str(exc)[:120]))
	return bad


def check_balance(dataset_path: Path = DATASET_PATH) -> dict[str, int]:
	"""Return file count per genre folder."""
	counts: dict[str, int] = {}
	for genre in sorted(os.listdir(dataset_path)):
		genre_dir = dataset_path / genre
		if not genre_dir.is_dir():
			continue
		counts[genre] = sum(1 for f in os.listdir(genre_dir) if f.lower().endswith(".wav"))
	return counts


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

	# ── Dataset quality checks ────────────────────────────────────────────────
	print("=== Dataset Balance Check ===")
	counts = check_balance()
	for genre, n in counts.items():
		status = "OK" if n == 100 else f"WARNING: {n} files"
		print(f"  {genre:<12} {n:>4}  [{status}]")
	min_c, max_c = min(counts.values()), max(counts.values())
	print(f"\nBalanced: {'YES' if min_c == max_c else 'NO — min=' + str(min_c) + ', max=' + str(max_c)}")

	print("\n=== Corruption Scan (loading first 5 s of each file) ===")
	bad_files = scan_dataset()
	if bad_files:
		BAD_FILES_LOG.parent.mkdir(parents=True, exist_ok=True)
		with open(BAD_FILES_LOG, "w") as f:
			for genre, fname, reason in bad_files:
				line = f"[{genre}] {fname} -> {reason}"
				print(f"  BAD: {line}")
				f.write(line + "\n")
		print(f"  Logged {len(bad_files)} bad file(s) to: {BAD_FILES_LOG}")
	else:
		print("  All files OK.")

	# ── Skip extraction if --check flag was passed ────────────────────────────
	if "--check" in sys.argv:
		print("\n--check flag set: skipping feature extraction.")
		return

	# ── Feature extraction ────────────────────────────────────────────────────
	print("\n=== Extracting Features ===")
	features: list[list[float]] = []
	labels: list[str] = []
	bad_set = {fname for _, fname, _ in bad_files}

	for genre in sorted(os.listdir(DATASET_PATH)):
		genre_folder = DATASET_PATH / genre
		if not genre_folder.is_dir():
			continue

		for file_name in sorted(os.listdir(genre_folder)):
			if not file_name.lower().endswith(".wav"):
				continue

			# Skip files identified as corrupted in the scan
			if file_name in bad_set:
				print(f"  Skipping known bad file: {file_name}")
				continue

			file_path = genre_folder / file_name

			try:
				row = extract_file_features(file_path)
			except Exception as exc:
				print(f"  Skipping {file_name}: {exc}")
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
