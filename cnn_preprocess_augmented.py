import os
import random

import librosa
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm


# =========================
# Paths
# =========================

METADATA_PATH = (
    "/content/drive/MyDrive/fma-small/"
    "fma_metadata/fma_metadata/tracks.csv"
)

AUDIO_DIR = "/content/fma_small"

OUTPUT_DIR = (
    "/content/drive/MyDrive/"
    "cnn_augmented_data"
)


# =========================
# Settings
# =========================

SAMPLE_RATE = 22050
SEGMENT_SECONDS = 3

N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512

RANDOM_STATE = 49

SHARD_SIZE = 256

MAX_AUDIO_SECONDS = 30

TARGET_GENRES = [
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
# Augmentation settings
# =========================

AUGMENT_PROBABILITY = 0.80

TIME_SHIFT_SECONDS = 0.5

TIME_STRETCH_MIN = 0.90
TIME_STRETCH_MAX = 1.10

PITCH_SHIFT_MIN = -2
PITCH_SHIFT_MAX = 2

GAIN_MIN_DB = -20
GAIN_MAX_DB = 20


random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


# =========================
# Metadata
# =========================

def load_metadata():

    tracks = pd.read_csv(
        METADATA_PATH,
        index_col=0,
        header=[0, 1]
    )

    tracks = tracks[
        tracks[("set", "subset")] == "small"
    ]

    tracks = tracks[
        tracks[("track", "genre_top")].isin(
            TARGET_GENRES
        )
    ]

    tracks = tracks[
        [("track", "genre_top")]
    ].dropna()

    return tracks


# =========================
# Audio path
# =========================

def get_audio_path(track_id):

    tid = f"{int(track_id):06d}"

    return os.path.join(
        AUDIO_DIR,
        tid[:3],
        tid + ".mp3"
    )


# =========================
# Audio augmentation
# =========================

def fix_length(y, target_length):

    if len(y) > target_length:

        return y[:target_length]

    if len(y) < target_length:

        return np.pad(
            y,
            (0, target_length - len(y))
        )

    return y


def time_shift(y):

    max_shift = int(
        TIME_SHIFT_SECONDS * SAMPLE_RATE
    )

    shift = random.randint(
        -max_shift,
        max_shift
    )

    if shift == 0:
        return y

    return np.roll(y, shift)


def time_stretch(y):

    rate = random.uniform(
        TIME_STRETCH_MIN,
        TIME_STRETCH_MAX
    )

    stretched = librosa.effects.time_stretch(
        y,
        rate=rate
    )

    return fix_length(
        stretched,
        len(y)
    )


def pitch_shift(y, sr):

    steps = random.uniform(
        PITCH_SHIFT_MIN,
        PITCH_SHIFT_MAX
    )

    return librosa.effects.pitch_shift(
        y,
        sr=sr,
        n_steps=steps
    )


def gain(y):

    gain_db = random.uniform(
        GAIN_MIN_DB,
        GAIN_MAX_DB
    )

    factor = 10 ** (gain_db / 20.0)

    return y * factor


def augment_audio(y, sr):

    if random.random() > AUGMENT_PROBABILITY:
        return y

    operations = []

    if random.random() < 0.50:
        operations.append("shift")

    if random.random() < 0.50:
        operations.append("stretch")

    if random.random() < 0.50:
        operations.append("pitch")

    if random.random() < 0.50:
        operations.append("gain")

    if not operations:

        operations.append(
            random.choice([
                "shift",
                "stretch",
                "pitch",
                "gain"
            ])
        )

    for operation in operations:

        if operation == "shift":
            y = time_shift(y)

        elif operation == "stretch":
            y = time_stretch(y)

        elif operation == "pitch":
            y = pitch_shift(y, sr)

        elif operation == "gain":
            y = gain(y)

    return np.clip(y, -1.0, 1.0)


# =========================
# Mel conversion
# =========================

def audio_to_segments(
    file_path,
    augment=False
):

    y, sr = librosa.load(
        file_path,
        sr=SAMPLE_RATE,
        mono=True,
        duration=MAX_AUDIO_SECONDS
    )

    segment_length = (
        SAMPLE_RATE * SEGMENT_SECONDS
    )

    segments = []

    for start in range(
        0,
        len(y) - segment_length + 1,
        segment_length
    ):

        segment = y[
            start:start + segment_length
        ]

        if augment:

            segment = augment_audio(
                segment,
                sr
            )

        mel = librosa.feature.melspectrogram(
            y=segment,
            sr=sr,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
            power=2.0
        )

        mel_db = librosa.power_to_db(
            mel,
            ref=np.max
        )

        segments.append(
            mel_db.astype(np.float32)
        )

    return np.asarray(
        segments,
        dtype=np.float32
    )


# =========================
# Track split
# =========================

def create_splits(tracks):

    track_ids = tracks.index.to_numpy()

    labels = tracks[
        ("track", "genre_top")
    ].to_dict()

    trainval_ids, test_ids = train_test_split(
        track_ids,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=[
            labels[x]
            for x in track_ids
        ]
    )

    train_ids, val_ids = train_test_split(
        trainval_ids,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=[
            labels[x]
            for x in trainval_ids
        ]
    )

    return (
        train_ids,
        val_ids,
        test_ids,
        labels
    )


# =========================
# Shard writer
# =========================

class ShardWriter:

    def __init__(
        self,
        split_name,
        shard_size
    ):

        self.split_name = split_name
        self.shard_size = shard_size

        self.X = []
        self.y = []
        self.track_ids = []

        self.shard_index = 0

        self.output_dir = os.path.join(
            OUTPUT_DIR,
            split_name
        )

        os.makedirs(
            self.output_dir,
            exist_ok=True
        )

    def add(
        self,
        segments,
        label,
        track_id
    ):

        for segment in segments:

            self.X.append(segment)
            self.y.append(label)
            self.track_ids.append(track_id)

            if len(self.X) >= self.shard_size:

                self.flush()

    def flush(self):

        if not self.X:
            return

        X = np.asarray(
            self.X,
            dtype=np.float32
        )

        y = np.asarray(
            self.y,
            dtype=np.int32
        )

        track_ids = np.asarray(
            self.track_ids,
            dtype=np.int32
        )

        output_file = os.path.join(
            self.output_dir,
            f"shard_{self.shard_index:05d}.npz"
        )

        np.savez(
            output_file,
            X=X,
            y=y,
            track_ids=track_ids
        )

        print(
            f"Saved {output_file} "
            f"({len(X)} segments)"
        )

        self.X.clear()
        self.y.clear()
        self.track_ids.clear()

        self.shard_index += 1

    def close(self):

        self.flush()


# =========================
# Process split
# =========================

def process_split(
    track_ids,
    labels,
    split_name,
    encoder
):

    augment = split_name == "train"

    print(
        f"\nProcessing {split_name.upper()}"
    )

    writer = ShardWriter(
        split_name,
        SHARD_SIZE
    )

    valid_tracks = []
    valid_labels = []

    for track_id in tqdm(
        track_ids,
        desc=split_name
    ):

        file_path = get_audio_path(
            track_id
        )

        if not os.path.exists(file_path):

            print(
                f"\nMissing: {track_id}"
            )

            continue

        try:

            segments = audio_to_segments(
                file_path,
                augment=augment
            )

            if len(segments) == 0:

                print(
                    f"\nNo segments: {track_id}"
                )

                continue

            label = encoder.transform(
                [labels[track_id]]
            )[0]

            writer.add(
                segments,
                label,
                track_id
            )

            valid_tracks.append(
                track_id
            )

            valid_labels.append(
                label
            )

        except Exception as e:

            print(
                f"\nSkipping {track_id}: {e}"
            )

    writer.close()

    return (
        np.asarray(
            valid_tracks,
            dtype=np.int32
        ),
        np.asarray(
            valid_labels,
            dtype=np.int32
        )
    )


# =========================
# Main
# =========================

def main():

    print("Loading metadata...")

    tracks = load_metadata()

    print(
        f"Target tracks: {len(tracks)}"
    )

    print(
        "\nGenre distribution:"
    )

    print(
        tracks[
            ("track", "genre_top")
        ].value_counts()
    )

    (
        train_ids,
        val_ids,
        test_ids,
        labels
    ) = create_splits(tracks)

    print(
        f"\nInitial split:"
    )

    print(
        f"Train: {len(train_ids)}"
    )

    print(
        f"Validation: {len(val_ids)}"
    )

    print(
        f"Test: {len(test_ids)}"
    )

    encoder = LabelEncoder()

    encoder.fit(
        [
            labels[x]
            for x in train_ids
        ]
    )

    os.makedirs(
        os.path.join(
            OUTPUT_DIR,
            "metadata"
        ),
        exist_ok=True
    )

    print(
        "\nClasses:"
    )

    print(
        encoder.classes_
    )

    print(
        "\nProcessing training audio..."
    )

    valid_train_ids, train_y = process_split(
        train_ids,
        labels,
        "train",
        encoder
    )

    print(
        "\nProcessing validation audio..."
    )

    valid_val_ids, val_y = process_split(
        val_ids,
        labels,
        "val",
        encoder
    )

    print(
        "\nProcessing test audio..."
    )

    valid_test_ids, test_y = process_split(
        test_ids,
        labels,
        "test",
        encoder
    )

    metadata_dir = os.path.join(
        OUTPUT_DIR,
        "metadata"
    )

    np.save(
        os.path.join(
            metadata_dir,
            "train_tracks.npy"
        ),
        valid_train_ids
    )

    np.save(
        os.path.join(
            metadata_dir,
            "train_labels.npy"
        ),
        train_y
    )

    np.save(
        os.path.join(
            metadata_dir,
            "val_tracks.npy"
        ),
        valid_val_ids
    )

    np.save(
        os.path.join(
            metadata_dir,
            "val_labels.npy"
        ),
        val_y
    )

    np.save(
        os.path.join(
            metadata_dir,
            "test_tracks.npy"
        ),
        valid_test_ids
    )

    np.save(
        os.path.join(
            metadata_dir,
            "test_labels.npy"
        ),
        test_y
    )

    np.save(
        os.path.join(
            metadata_dir,
            "classes.npy"
        ),
        encoder.classes_
    )

    print(
        "\n=========================="
    )

    print(
        "PREPROCESSING COMPLETE"
    )

    print(
        "=========================="
    )

    print(
        f"Valid train tracks: {len(valid_train_ids)}"
    )

    print(
        f"Valid validation tracks: {len(valid_val_ids)}"
    )

    print(
        f"Valid test tracks: {len(valid_test_ids)}"
    )

    print(
        f"\nSaved to:"
    )

    print(
        OUTPUT_DIR
    )


if __name__ == "__main__":
    main()