import os
import glob
import random

import numpy as np
import pandas as pd
import librosa
import tensorflow as tf


# =========================
# Paths
# =========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "model.keras")
AUDIO_DIR = os.path.join(BASE_DIR, "fma-small")

# Change this if your tracks.csv is somewhere else
TRACKS_CSV = os.path.join(BASE_DIR, "tracks.csv")

NUM_SAMPLES = 200

SR = 22050
DURATION = 30
SEGMENT_DURATION = 3

N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512

SEGMENT_SAMPLES = SR * SEGMENT_DURATION
MAX_SAMPLES = SR * DURATION


CLASSES = [
    "Electronic",
    "Experimental",
    "Folk",
    "Hip-Hop",
    "Instrumental",
    "International",
    "Pop",
    "Rock"
]


# =========================
# SpecAugment
# =========================

class SpecAugment(tf.keras.layers.Layer):

    def __init__(self, freq_mask=16, time_mask=16, **kwargs):
        super().__init__(**kwargs)

        self.freq_mask = freq_mask
        self.time_mask = time_mask

    def call(self, inputs, training=None):

        if not training:
            return inputs

        return inputs

    def get_config(self):

        config = super().get_config()

        config.update({
            "freq_mask": self.freq_mask,
            "time_mask": self.time_mask
        })

        return config


# =========================
# Load model
# =========================

print("Loading model...")

model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={
        "SpecAugment": SpecAugment
    },
    compile=False
)

print("Model loaded successfully.\n")


# =========================
# Find tracks.csv
# =========================

if not os.path.exists(TRACKS_CSV):

    possible = glob.glob(
        os.path.join(BASE_DIR, "**", "tracks.csv"),
        recursive=True
    )

    if possible:
        TRACKS_CSV = possible[0]

    else:
        raise FileNotFoundError(
            "tracks.csv not found.\n"
            "Put tracks.csv in the project root or update TRACKS_CSV."
        )

print("Metadata:", TRACKS_CSV)


# =========================
# Load metadata
# =========================

print("Loading metadata...")

tracks = pd.read_csv(
    TRACKS_CSV,
    header=[0, 1, 2],
    index_col=0
)

# Find genre_top column
genre_col = None

for col in tracks.columns:

    text = " ".join(
        str(x).lower()
        for x in col
    )

    if "genre_top" in text:
        genre_col = col
        break

if genre_col is None:

    raise ValueError(
        "Could not find genre_top column in tracks.csv"
    )


genre_map = {}

for track_id, row in tracks.iterrows():

    genre = row[genre_col]

    if pd.isna(genre):
        continue

    genre = str(genre)

    if genre in CLASSES:
        genre_map[int(track_id)] = genre


print(
    f"Genres available for {len(genre_map)} tracks.\n"
)


# =========================
# Find audio files
# =========================

audio_files = glob.glob(
    os.path.join(
        AUDIO_DIR,
        "**",
        "*.mp3"
    ),
    recursive=True
)

valid_audio = []

for path in audio_files:

    filename = os.path.basename(path)

    try:
        track_id = int(
            os.path.splitext(filename)[0]
        )
    except ValueError:
        continue

    if track_id in genre_map:
        valid_audio.append(path)


print(
    f"Audio files with known genres: "
    f"{len(valid_audio)}"
)


if len(valid_audio) < NUM_SAMPLES:

    raise RuntimeError(
        f"Only {len(valid_audio)} usable audio files found."
    )


# =========================
# Random selection
# =========================

random.seed()

selected = random.sample(
    valid_audio,
    NUM_SAMPLES
)


# =========================
# Audio → Log-Mel
# =========================

def audio_to_segments(path):

    audio, _ = librosa.load(
        path,
        sr=SR,
        mono=True
    )

    # Limit / pad to 30 seconds
    if len(audio) > MAX_SAMPLES:

        audio = audio[:MAX_SAMPLES]

    elif len(audio) < MAX_SAMPLES:

        audio = np.pad(
            audio,
            (0, MAX_SAMPLES - len(audio))
        )

    segments = []

    for start in range(
        0,
        MAX_SAMPLES,
        SEGMENT_SAMPLES
    ):

        segment = audio[
            start:start + SEGMENT_SAMPLES
        ]

        if len(segment) < SEGMENT_SAMPLES:

            segment = np.pad(
                segment,
                (0, SEGMENT_SAMPLES - len(segment))
            )

        mel = librosa.feature.melspectrogram(
            y=segment,
            sr=SR,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
            power=2.0
        )

        mel_db = librosa.power_to_db(
            mel,
            ref=np.max
        )

        # Same normalization used during CNN testing
        mel_db = (mel_db + 80.0) / 80.0

        segments.append(mel_db.astype(np.float32))

    X = np.array(segments)

    # Add channel dimension
    X = X[..., np.newaxis]

    return X


# =========================
# Test
# =========================

correct = 0

print("\n")
print("=" * 85)
print("RANDOM 20 AUDIO TEST")
print("=" * 85)


for i, path in enumerate(selected, 1):

    filename = os.path.basename(path)

    track_id = int(
        os.path.splitext(filename)[0]
    )

    actual = genre_map[track_id]

    try:

        X = audio_to_segments(path)

        predictions = model.predict(
            X,
            verbose=0
        )

        # Average all 3-sec segment predictions
        mean_prediction = np.mean(
            predictions,
            axis=0
        )

        predicted_index = np.argmax(
            mean_prediction
        )

        predicted = CLASSES[predicted_index]

        confidence = (
            mean_prediction[predicted_index] * 100
        )

        is_correct = predicted == actual

        if is_correct:
            correct += 1
            result = "✅ CORRECT"
        else:
            result = "❌ WRONG"

        print(
            f"\n{i:02d}. {filename}"
        )

        print(
            f"    Actual    : {actual}"
        )

        print(
            f"    Predicted : {predicted}"
        )

        print(
            f"    Confidence: {confidence:.2f}%"
        )

        print(
            f"    Result    : {result}"
        )

    except Exception as e:

        print(
            f"\n{i:02d}. {filename}"
        )

        print(
            f"    ERROR: {e}"
        )


# =========================
# Final result
# =========================

accuracy = (
    correct / NUM_SAMPLES
) * 100


print("\n")
print("=" * 85)
print("FINAL RESULT")
print("=" * 85)

print(
    f"Correct : {correct}/{NUM_SAMPLES}"
)

print(
    f"Wrong   : {NUM_SAMPLES - correct}/{NUM_SAMPLES}"
)

print(
    f"Accuracy: {accuracy:.2f}%"
)

print("=" * 85)