import os
import glob

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# SETTINGS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_PATHS = [
    os.path.join(BASE_DIR, "model.keras"),
    os.path.join(BASE_DIR, "modelnew.keras"),
]

TEST_DIR = os.path.join(
    BASE_DIR,
    "cnn_data",
    "test"
)

TRACKS_CSV = os.path.join(
    BASE_DIR,
    "tracks.csv"
)

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


# ============================================================
# SPEC AUGMENT
# ============================================================

class SpecAugment(tf.keras.layers.Layer):

    def __init__(
        self,
        freq_mask=16,
        time_mask=16,
        n_freq_masks=2,
        n_time_masks=2,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.freq_mask = freq_mask
        self.time_mask = time_mask
        self.n_freq_masks = n_freq_masks
        self.n_time_masks = n_time_masks

    def call(self, inputs, training=None):

        if not training:
            return inputs

        return inputs

    def get_config(self):

        config = super().get_config()

        config.update({
            "freq_mask": self.freq_mask,
            "time_mask": self.time_mask,
            "n_freq_masks": self.n_freq_masks,
            "n_time_masks": self.n_time_masks
        })

        return config


# ============================================================
# FIND TRACKS.CSV
# ============================================================

if not os.path.exists(TRACKS_CSV):

    possible = glob.glob(
        os.path.join(
            BASE_DIR,
            "**",
            "tracks.csv"
        ),
        recursive=True
    )

    if not possible:
        raise FileNotFoundError(
            "tracks.csv not found."
        )

    TRACKS_CSV = possible[0]


print("Metadata:", TRACKS_CSV)


# ============================================================
# LOAD METADATA
# ============================================================

tracks = pd.read_csv(
    TRACKS_CSV,
    header=[0, 1, 2],
    index_col=0
)

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
        "genre_top column not found."
    )


genre_map = {}

for track_id, row in tracks.iterrows():

    genre = row[genre_col]

    if pd.isna(genre):
        continue

    genre = str(genre)

    if genre in CLASSES:

        genre_map[int(track_id)] = (
            CLASSES.index(genre)
        )


# ============================================================
# FIND TEST TRACKS
# ============================================================

test_files = sorted(
    glob.glob(
        os.path.join(
            TEST_DIR,
            "*.npy"
        )
    )
)

if not test_files:

    raise FileNotFoundError(
        f"No test .npy files found in {TEST_DIR}"
    )


test_tracks = []

for path in test_files:

    try:

        track_id = int(
            os.path.splitext(
                os.path.basename(path)
            )[0]
        )

    except ValueError:

        continue

    if track_id in genre_map:

        test_tracks.append(
            track_id
        )


test_tracks = np.array(
    sorted(test_tracks),
    dtype=np.int64
)

y_true = np.array([
    genre_map[int(track_id)]
    for track_id in test_tracks
])


print()
print("=" * 70)
print("PURE HELD-OUT TEST SET")
print("=" * 70)

print(
    "Test tracks:",
    len(test_tracks)
)

print(
    "Expected: approximately 2397"
)


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate_model(model_path):

    print()
    print("=" * 70)
    print("MODEL")
    print("=" * 70)

    print(
        "Loading:",
        os.path.basename(model_path)
    )

    model = tf.keras.models.load_model(
        model_path,
        custom_objects={
            "SpecAugment": SpecAugment
        },
        compile=False
    )

    print("Loaded.")

    predictions = []

    for index, track_id in enumerate(
        test_tracks,
        1
    ):

        path = os.path.join(
            TEST_DIR,
            f"{int(track_id):06d}.npy"
        )

        X = np.load(
            path
        ).astype(np.float32)

        # Same normalization as training
        X = (X + 80.0) / 80.0

        # Channel dimension
        X = X[..., np.newaxis]

        # Direct model call
        segment_probs = model(
            X,
            training=False
        ).numpy()

        # Track-level prediction
        mean_probs = np.mean(
            segment_probs,
            axis=0
        )

        prediction = int(
            np.argmax(mean_probs)
        )

        predictions.append(
            prediction
        )

        if index % 250 == 0:

            print(
                f"Processed "
                f"{index}/{len(test_tracks)}"
            )


    y_pred = np.array(
        predictions
    )

    # --------------------------------------------------------
    # Overall accuracy
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    correct = int(
        np.sum(y_true == y_pred)
    )

    print()
    print("=" * 70)
    print("RESULT")
    print("=" * 70)

    print(
        f"Correct  : {correct}/{len(y_true)}"
    )

    print(
        f"Wrong    : "
        f"{len(y_true) - correct}/{len(y_true)}"
    )

    print(
        f"Accuracy : {accuracy * 100:.2f}%"
    )


    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("CLASSIFICATION REPORT")
    print("=" * 70)

    print(
        classification_report(
            y_true,
            y_pred,
            labels=np.arange(
                len(CLASSES)
            ),
            target_names=CLASSES,
            digits=4,
            zero_division=0
        )
    )


    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    print("=" * 70)
    print("CONFUSION MATRIX")
    print("=" * 70)

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=np.arange(
            len(CLASSES)
        )
    )

    print(
        "\n"
        + "Actual \\ Pred".ljust(18)
        + "".join(
            f"{c[:10]:>12}"
            for c in CLASSES
        )
    )

    for i, row in enumerate(cm):

        print(
            CLASSES[i].ljust(18)
            + "".join(
                f"{value:>12}"
                for value in row
            )
        )

    return accuracy, y_pred


# ============================================================
# TEST BOTH MODELS
# ============================================================

results = {}

for model_path in MODEL_PATHS:

    if not os.path.exists(model_path):

        print(
            f"\nSkipping missing model: "
            f"{model_path}"
        )

        continue

    accuracy, predictions = evaluate_model(
        model_path
    )

    results[
        os.path.basename(model_path)
    ] = accuracy


# ============================================================
# FINAL COMPARISON
# ============================================================

print()
print("=" * 70)
print("FINAL MODEL COMPARISON")
print("=" * 70)

for model_name, accuracy in results.items():

    print(
        f"{model_name:<30} "
        f"{accuracy * 100:.2f}%"
    )

if len(results) == 2:

    names = list(results.keys())

    old_acc = results[names[0]]
    new_acc = results[names[1]]

    print()
    print(
        f"Change: "
        f"{(new_acc - old_acc) * 100:+.2f} percentage points"
    )

print("=" * 70)