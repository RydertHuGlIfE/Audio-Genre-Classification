import os

import numpy as np
import pandas as pd
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


METADATA_DIR = "fma-small/fma_metadata/fma_metadata"
AUDIO_DIR = "fma-small/fma_small/fma_small"

OUTPUT_DIR = "cnn_data"

SAMPLE_RATE = 22050
SEGMENT_SECONDS = 3
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
RANDOM_STATE = 49


def get_metadata():

    tracks_path = os.path.join(
        METADATA_DIR,
        "tracks.csv"
    )

    tracks = pd.read_csv(
        tracks_path,
        index_col=0,
        header=[0, 1]
    )

    tracks = tracks[
        tracks[("set", "subset")] == "small"
    ]

    tracks = tracks[
        [("track", "genre_top")]
    ].dropna()

    return tracks


def get_audio_path(track_id):

    tid = f"{track_id:06d}"

    return os.path.join(
        AUDIO_DIR,
        tid[:3],
        tid + ".mp3"
    )


def audio_to_segments(file_path):

    y, sr = librosa.load(
        file_path,
        sr=SAMPLE_RATE,
        mono=True,
        duration=30
    )

    segment_length = SAMPLE_RATE * SEGMENT_SECONDS

    segments = []

    for start in range(
        0,
        len(y) - segment_length + 1,
        segment_length
    ):

        segment = y[
            start:start + segment_length
        ]

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

    return np.array(
        segments,
        dtype=np.float32
    )


def process_tracks(
    track_ids,
    labels,
    split_name
):

    output_dir = os.path.join(
        OUTPUT_DIR,
        split_name
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    valid_ids = []
    label_list = []

    total = len(track_ids)

    for i, track_id in enumerate(track_ids):

        file_path = get_audio_path(track_id)

        if not os.path.exists(file_path):
            print(f"Missing: {track_id}")
            continue

        try:

            segments = audio_to_segments(
                file_path
            )

            if len(segments) == 0:
                print(
                    f"No segments: {track_id}"
                )
                continue

            output_file = os.path.join(
                output_dir,
                f"{track_id:06d}.npy"
            )

            np.save(
                output_file,
                segments
            )

            valid_ids.append(track_id)
            label_list.append(labels[track_id])

        except Exception as e:

            print(
                f"Skipping {track_id}: {e}"
            )

        if (
            (i + 1) % 100 == 0
            or i + 1 == total
        ):

            print(
                f"{split_name}: "
                f"{i + 1}/{total} tracks"
            )

    return (
        np.array(valid_ids),
        np.array(label_list)
    )


def main():

    print("Loading metadata...")

    tracks = get_metadata()

    track_ids = tracks.index.to_numpy()

    labels = tracks[
        ("track", "genre_top")
    ].to_dict()

    print(
        f"Total tracks: {len(track_ids)}"
    )

    # Split tracks before segmentation

    train_ids, test_ids = train_test_split(
        track_ids,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=[
            labels[x]
            for x in track_ids
        ]
    )

    print(
        f"Train tracks: {len(train_ids)}"
    )

    print(
        f"Test tracks:  {len(test_ids)}"
    )

    # Encode labels

    encoder = LabelEncoder()

    encoder.fit([
        labels[x]
        for x in train_ids
    ])

    train_labels = {
        track_id: encoder.transform(
            [labels[track_id]]
        )[0]
        for track_id in train_ids
    }

    test_labels = {
        track_id: encoder.transform(
            [labels[track_id]]
        )[0]
        for track_id in test_ids
    }

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    print("\nProcessing training audio...")

    valid_train_ids, train_y = process_tracks(
        train_ids,
        train_labels,
        "train"
    )

    print("\nProcessing test audio...")

    valid_test_ids, test_y = process_tracks(
        test_ids,
        test_labels,
        "test"
    )

    # Save metadata

    np.save(
        os.path.join(
            OUTPUT_DIR,
            "train_tracks.npy"
        ),
        valid_train_ids
    )

    np.save(
        os.path.join(
            OUTPUT_DIR,
            "train_labels.npy"
        ),
        train_y
    )

    np.save(
        os.path.join(
            OUTPUT_DIR,
            "test_tracks.npy"
        ),
        valid_test_ids
    )

    np.save(
        os.path.join(
            OUTPUT_DIR,
            "test_labels.npy"
        ),
        test_y
    )

    np.save(
        os.path.join(
            OUTPUT_DIR,
            "classes.npy"
        ),
        encoder.classes_
    )

    print("\nDone.")

    print(
        f"Train files: {len(valid_train_ids)}"
    )

    print(
        f"Test files: {len(valid_test_ids)}"
    )

    print(
        f"Saved in: {OUTPUT_DIR}/"
    )


if __name__ == "__main__":
    main()