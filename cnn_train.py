import os
import random

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


DATA_DIR = "cnn_data"

TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")

BATCH_SIZE = 32
EPOCHS = 20
RANDOM_STATE = 49


random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)


classes = np.load(
    os.path.join(DATA_DIR, "classes.npy"),
    allow_pickle=True
)

train_tracks = np.load(
    os.path.join(DATA_DIR, "train_tracks.npy")
)

train_labels = np.load(
    os.path.join(DATA_DIR, "train_labels.npy")
)

test_tracks = np.load(
    os.path.join(DATA_DIR, "test_tracks.npy")
)

test_labels = np.load(
    os.path.join(DATA_DIR, "test_labels.npy")
)


def create_samples(tracks, labels, directory):
    samples = []
    for track_id, label in zip(tracks, labels):
        path = os.path.join(
            directory,
            f"{track_id:06d}.npy"
        )
        if not os.path.exists(path):
            continue
        data = np.load(path)
        for i in range(len(data)):
            samples.append(
                (
                    path,
                    i,
                    int(label)
                )
            )

    return samples


print("Preparing training samples...")

train_samples = create_samples(
    train_tracks,
    train_labels,
    TRAIN_DIR
)

print(f"Train segments: {len(train_samples)}")

print("Preparing test samples...")

test_samples = create_samples(
    test_tracks,
    test_labels,
    TEST_DIR
)

print(f"Test segments: {len(test_samples)}")


class AudioGenerator(
    keras.utils.Sequence
):

    def __init__(
        self,
        samples,
        batch_size=32,
        shuffle=True
    ):

        self.samples = samples
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(samples))

        self.on_epoch_end()

    def __len__(self):

        return int(
            np.ceil(
                len(self.samples)
                / self.batch_size
            )
        )

    def __getitem__(self, index):

        batch_indexes = self.indexes[
            index * self.batch_size:
            (index + 1) * self.batch_size
        ]

        batch_samples = [
            self.samples[i]
            for i in batch_indexes
        ]

        X = []
        y = []

        cache = {}

        for path, segment_index, label in batch_samples:

            if path not in cache:
                cache[path] = np.load(path)

            segment = cache[path][segment_index]

            X.append(segment)
            y.append(label)

        X = np.array(
            X,
            dtype=np.float32
        )

        X = X[..., np.newaxis]

        y = np.array(
            y,
            dtype=np.int32
        )

        # Normalize using the dB range
        X = (X + 80.0) / 80.0

        return X, y

    def on_epoch_end(self):

        if self.shuffle:
            np.random.shuffle(self.indexes)


train_generator = AudioGenerator(
    train_samples,
    BATCH_SIZE,
    shuffle=True
)

test_generator = AudioGenerator(
    test_samples,
    BATCH_SIZE,
    shuffle=False
)


model = keras.Sequential([

    layers.Input(
        shape=(128, 130, 1)
    ),

    layers.Conv2D(
        32,
        (3, 3),
        padding="same",
        activation="relu"
    ),

    layers.BatchNormalization(),

    layers.MaxPooling2D(
        (2, 2)
    ),

    layers.Conv2D(
        64,
        (3, 3),
        padding="same",
        activation="relu"
    ),

    layers.BatchNormalization(),

    layers.MaxPooling2D(
        (2, 2)
    ),

    layers.Conv2D(
        128,
        (3, 3),
        padding="same",
        activation="relu"
    ),

    layers.BatchNormalization(),

    layers.MaxPooling2D(
        (2, 2)
    ),

    layers.Conv2D(
        256,
        (3, 3),
        padding="same",
        activation="relu"
    ),

    layers.BatchNormalization(),

    layers.GlobalAveragePooling2D(),

    layers.Dense(
        128,
        activation="relu"
    ),

    layers.Dropout(0.4),

    layers.Dense(
        len(classes),
        activation="softmax"
    )
])


model.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


model.summary()


os.makedirs(
    "cnn_model",
    exist_ok=True
)


callbacks = [

    keras.callbacks.ModelCheckpoint(
        "cnn_model/best_cnn.keras",
        monitor="val_accuracy",
        save_best_only=True,
        mode="max"
    ),

    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-6
    ),

    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True
    )
]


print("\nStarting training...\n")

history = model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=EPOCHS,
    callbacks=callbacks
)


print("\nFinal evaluation:")

loss, accuracy = model.evaluate(
    test_generator
)

print(
    f"Test Accuracy: {accuracy * 100:.2f}%"
)

model.save(
    "cnn_model/final_cnn.keras"
)

np.save(
    "cnn_model/history.npy",
    history.history,
    allow_pickle=True
)

print("\nModel saved.")