import sys
import tempfile
from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

BASE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = BASE_DIR / "src"

if str(SRC_DIR) not in sys.path:
	sys.path.insert(0, str(SRC_DIR))

import pandas as pd

from predict import predict_genre, predict_genre_proba  # noqa: E402

GENRE_EMOJI = {
    "blues": "🎷", "classical": "🎻", "country": "🤠",
    "disco": "🪩", "hiphop": "🎤", "jazz": "🎺",
    "metal": "🤘", "pop": "🎵", "reggae": "🌴", "rock": "🎸",
}

GENRE_RECOMMENDATIONS = {
    "blues":     ["B.B. King", "Muddy Waters", "Robert Johnson"],
    "classical": ["Ludwig van Beethoven", "Wolfgang Amadeus Mozart", "Johann Sebastian Bach"],
    "country":   ["Johnny Cash", "Dolly Parton", "Willie Nelson"],
    "disco":     ["Donna Summer", "Bee Gees", "Gloria Gaynor"],
    "hiphop":    ["Kendrick Lamar", "Jay-Z", "Tupac Shakur"],
    "jazz":      ["Miles Davis", "John Coltrane", "Louis Armstrong"],
    "metal":     ["Metallica", "AC/DC", "Black Sabbath"],
    "pop":       ["Michael Jackson", "Madonna", "Taylor Swift"],
    "reggae":    ["Bob Marley", "Peter Tosh", "Jimmy Cliff"],
    "rock":      ["AC/DC", "Nirvana", "Led Zeppelin"],
}


def plot_spectrogram(file_path: str) -> plt.Figure:
    y, sr = librosa.load(file_path, duration=30)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    S_dB = librosa.power_to_db(S, ref=np.max)
    fig, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(S_dB, x_axis="time", y_axis="mel", sr=sr, fmax=8000, ax=ax)
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set_title("Mel Spectrogram")
    plt.tight_layout()
    return fig


st.set_page_config(page_title="Music Genre Classifier", page_icon="🎵", layout="centered")
st.title("🎵 Music Genre Classifier")
st.write("Upload an audio file (.wav, .mp3, .ogg, .flac) to predict its genre.")

uploaded_file = st.file_uploader("Upload audio file", type=["wav", "mp3", "ogg", "flac"])

if uploaded_file is not None:
    suffix = Path(uploaded_file.name).suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(uploaded_file.read())
        temp_path = temp_file.name

    try:
        with st.spinner("Analysing audio..."):
            genre = predict_genre(temp_path)
            proba = predict_genre_proba(temp_path)
            fig = plot_spectrogram(temp_path)

        emoji = GENRE_EMOJI.get(genre.lower(), "🎶")
        st.success(f"Predicted Genre: **{genre.capitalize()}** {emoji}")

        top3 = sorted(proba.items(), key=lambda x: x[1], reverse=True)[:3]
        labels = [f"{GENRE_EMOJI.get(g, '🎶')} {g.capitalize()}" for g, _ in top3]
        values = [round(v * 100, 1) for _, v in top3]

        st.subheader("Top-3 Confidence")
        chart_df = pd.DataFrame({"Genre": labels, "Confidence (%)": values}).set_index("Genre")
        st.bar_chart(chart_df)

        st.subheader("🔊 Mel Spectrogram")
        st.pyplot(fig)
        plt.close(fig)

        recs = GENRE_RECOMMENDATIONS.get(genre.lower(), [])
        if recs:
            st.subheader("🎧 Recommended Artists")
            for artist in recs:
                st.markdown(f"- {artist}")

    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
    finally:
        Path(temp_path).unlink(missing_ok=True)
