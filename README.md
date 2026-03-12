# 🎵 Music Genre Classifier

A machine learning project that classifies music into one of **10 genres** from a 30-second audio clip using the [GTZAN dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification). Features a Streamlit web app for live predictions with spectrogram visualization and artist recommendations.

---

## 📁 Project Structure

```
music-genre-classifier/
│
├── dataset/
│   └── genres_original/          # GTZAN audio files (100 × 10 genres)
│       ├── blues/
│       ├── classical/
│       ├── country/
│       ├── disco/
│       ├── hiphop/
│       ├── jazz/
│       ├── metal/
│       ├── pop/
│       ├── reggae/
│       └── rock/
│
├── features/
│   └── features.csv              # Extracted audio features (generated)
│
├── models/
│   ├── genre_model.pkl           # Trained pipeline (generated)
│   └── confusion_matrix.png      # Evaluation heatmap (generated)
│
├── notebooks/
│   └── analysis.ipynb            # Exploratory analysis notebook
│
├── app/
│   └── app.py                    # Streamlit web application
│
├── src/
│   ├── extract_features.py       # Audio → numerical feature extraction
│   ├── train_model.py            # Model training + evaluation
│   └── predict.py                # Prediction + probability functions
│
├── requirements.txt
└── README.md
```

---

## 🎼 Genres Supported

| Genre | Emoji | Example Artists |
|---|---|---|
| Blues | 🎷 | B.B. King, Muddy Waters |
| Classical | 🎻 | Beethoven, Mozart |
| Country | 🤠 | Johnny Cash, Dolly Parton |
| Disco | 🪩 | Donna Summer, Bee Gees |
| Hip-Hop | 🎤 | Kendrick Lamar, Jay-Z |
| Jazz | 🎺 | Miles Davis, John Coltrane |
| Metal | 🤘 | Metallica, Black Sabbath |
| Pop | 🎵 | Michael Jackson, Taylor Swift |
| Reggae | 🌴 | Bob Marley, Peter Tosh |
| Rock | 🎸 | AC/DC, Nirvana, Led Zeppelin |

---

## ⚙️ Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/SruthiJayaram/music-genre-classifier.git
cd music-genre-classifier
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the dataset

Download the [GTZAN dataset from Kaggle](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification) and place the contents so the structure matches:

```
dataset/genres_original/blues/blues.00000.wav
dataset/genres_original/rock/rock.00000.wav
...
```

Or use the Kaggle CLI:

```bash
pip install kaggle
kaggle datasets download -d andradaolteanu/gtzan-dataset-music-genre-classification -p dataset/
```

---

## 🚀 Usage

### Step 1 — Extract Features

Converts all 1000 audio files into a 102-column CSV of numerical features.

```bash
python src/extract_features.py
```

Output: `features/features.csv`

### Step 2 — Train the Model

Trains a `StandardScaler → RandomForestClassifier` pipeline and saves the model.

```bash
python src/train_model.py
```

Output:
- `models/genre_model.pkl`
- `models/confusion_matrix.png`

### Step 3 — Predict a Single File

```bash
python src/predict.py
```

Or use it as a library:

```python
from src.predict import predict_genre, predict_genre_proba

print(predict_genre("path/to/song.wav"))
# → 'rock'

print(predict_genre_proba("path/to/song.wav"))
# → {'blues': 0.02, 'rock': 0.72, 'metal': 0.15, ...}
```

### Step 4 — Launch the Web App

```bash
python -m streamlit run app/app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🌐 Web App Features

| Feature | Description |
|---|---|
| 📤 Upload | Accepts `.wav`, `.mp3`, `.ogg`, `.flac` |
| 🏷️ Prediction | Displays predicted genre with emoji |
| 📊 Top-3 Confidence | Bar chart of top 3 genre probabilities |
| 🔊 Mel Spectrogram | Time-frequency heatmap of the audio |
| 🎧 Artist Recommendations | 3 iconic artists for the predicted genre |

---

## 🧠 Model Details

### Feature Extraction (102 features per file)

| Feature | Count | Description |
|---|---|---|
| MFCC means | 40 | Timbre characteristics per coefficient |
| MFCC std devs | 40 | Variance per MFCC coefficient |
| Spectral centroid | 1 | Brightness of the sound |
| Spectral rolloff | 1 | Frequency below which 85% of energy lies |
| Zero crossing rate | 1 | Noisiness / percussion level |
| Chroma | 12 | Pitch class distribution (C–B) |
| Spectral contrast | 7 | Harmonic vs percussive separation |
| Tempo | 1 | Estimated BPM |

### Training Pipeline

```
StandardScaler → RandomForestClassifier(n_estimators=200)
```

- Train/test split: 80% / 20%
- Test accuracy: **~72.5%**
- Known common confusions: Jazz ↔ Classical, Reggae ↔ Hip-Hop, Rock ↔ Disco

---

## 📊 Dataset

**GTZAN Music Genre Dataset**

- 1000 audio files total
- 10 genres × 100 files each
- Each file: 30 seconds, `.wav` format, 22050 Hz
- Source: [Kaggle — GTZAN Dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)

> **Note:** The dataset is not included in this repository due to its size (~1.2 GB). Download it separately using the instructions above.

---

## 📦 Dependencies

```
librosa       # Audio loading and feature extraction
numpy         # Numerical computing
pandas        # Data handling
scikit-learn  # Machine learning pipeline
matplotlib    # Plotting
seaborn       # Confusion matrix heatmap
streamlit     # Web application UI
joblib        # Model serialization
```

Install all with:
```bash
pip install -r requirements.txt
```

---

## 📄 License

This project is open source. Dataset is subject to its own [Kaggle license](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification).
