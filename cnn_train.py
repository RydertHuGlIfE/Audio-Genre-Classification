import os
import random

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from tqdm import tqdm


DATA_DIR = "/content/drive/MyDrive/cnn_data"

TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")

MODEL_DIR = "/content/drive/MyDrive/cnn_model"

BATCH_SIZE = 96
EPOCHS = 30
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


print("Classes:")
print(classes)

print(
    f"\nTotal training tracks: {len(train_tracks)}"
)

print(
    f"Total test tracks: {len(test_tracks)}"
)


# Split training tracks into train and validation

train_ids, val_ids, train_y_tracks, val_y_tracks = train_test_split(
    train_tracks,
    train_labels,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=train_labels
)


print(
    f"\nTraining tracks: {len(train_ids)}"
)

print(
    f"Validation tracks: {len(val_ids)}"
)

print(
    f"Test tracks: {len(test_tracks)}"
)


def create_samples(tracks, labels, directory, name):

    samples = []

    for track_id, label in tqdm(
        zip(tracks, labels),
        total=len(tracks),
        desc=name
    ):

        path = os.path.join(
            directory,
            f"{track_id:06d}.npy"
        )

        if not os.path.exists(path):
            continue

        try:

            data = np.load(path)

            for i in range(len(data)):

                samples.append(
                    (
                        path,
                        i,
                        int(label)
                    )
                )

        except Exception as e:

            print(
                f"\nSkipping {track_id}: {e}"
            )

    return samples


print("\nPreparing training samples...")

train_samples = create_samples(
    train_ids,
    train_y_tracks,
    TRAIN_DIR,
    "Train"
)

print(
    f"Train segments: {len(train_samples)}"
)


print("\nPreparing validation samples...")

val_samples = create_samples(
    val_ids,
    val_y_tracks,
    TRAIN_DIR,
    "Validation"
)

print(
    f"Validation segments: {len(val_samples)}"
)


print("\nPreparing test samples...")

test_samples = create_samples(
    test_tracks,
    test_labels,
    TEST_DIR,
    "Test"
)

print(
    f"Test segments: {len(test_samples)}"
)


class AudioGenerator(keras.utils.Sequence):

    def __init__(
        self,
        samples,
        batch_size=96,
        shuffle=True,
        **kwargs
    ):

        super().__init__(**kwargs)

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

        X = []
        y = []

        cache = {}

        for sample_index in batch_indexes:

            path, segment_index, label = (
                self.samples[sample_index]
            )

            if path not in cache:
                cache[path] = np.load(path)

            segment = cache[path][segment_index]

            X.append(segment)
            y.append(label)

        X = np.asarray(
            X,
            dtype=np.float32
        )

        X = X[..., np.newaxis]

        y = np.asarray(
            y,
            dtype=np.int32
        )

        X = (X + 80.0) / 80.0

        return X, y

    def on_epoch_end(self):

        if self.shuffle:
            np.random.shuffle(self.indexes)


train_generator = AudioGenerator(
    train_samples,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_generator = AudioGenerator(
    val_samples,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_generator = AudioGenerator(
    test_samples,
    batch_size=BATCH_SIZE,
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
    metrics=["accuracy"],
    steps_per_execution=10
)


model.summary()


os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


callbacks = [

    keras.callbacks.ModelCheckpoint(
        os.path.join(
            MODEL_DIR,
            "best_cnn.keras"
        ),
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1
    ),

    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1
    ),

    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
]


print("\nStarting training...\n")


history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=callbacks
)


print("\nFinal test evaluation...\n")


loss, accuracy = model.evaluate(
    test_generator,
    verbose=1
)


print(
    f"\nTest Accuracy: {accuracy * 100:.2f}%"
)


model.save(
    os.path.join(
        MODEL_DIR,
        "final_cnn.keras"
    )
)


np.save(
    os.path.join(
        MODEL_DIR,
        "history.npy"
    ),
    history.history,
    allow_pickle=True
)


np.save(
    os.path.join(
        MODEL_DIR,
        "classes.npy"
    ),
    classes
)


print("\nModel saved to:")
print(MODEL_DIR)