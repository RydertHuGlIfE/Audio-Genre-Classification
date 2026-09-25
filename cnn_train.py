import os
import random
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from tqdm import tqdm


DATA_DIR = "/content/cnn_data"

TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")

MODEL_DIR = "/content/drive/MyDrive/cnn_model"

BATCH_SIZE = 96
EPOCHS = 30
RANDOM_STATE = 49
MAX_WORKERS = 16


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


def get_track_info(args):
    track_id, label, directory = args

    path = os.path.join(
        directory,
        f"{track_id:06d}.npy"
    )

    if not os.path.exists(path):
        return path, int(label), 0

    try:
        data = np.load(
            path,
            mmap_mode="r"
        )

        return path, int(label), data.shape[0]

    except Exception as e:
        print(
            f"\nSkipping {track_id}: {e}"
        )
        return path, int(label), 0


def get_track_counts(tracks, labels, directory, name):

    tasks = [
        (
            track_id,
            label,
            directory
        )
        for track_id, label in zip(
            tracks,
            labels
        )
    ]

    results = []

    with ThreadPoolExecutor(
        max_workers=MAX_WORKERS
    ) as executor:

        mapped = executor.map(
            get_track_info,
            tasks
        )

        for result in tqdm(
            mapped,
            total=len(tasks),
            desc=name
        ):
            results.append(result)

    return results


def load_tracks_to_ram(
    tracks,
    labels,
    directory,
    name
):

    print(f"\nCounting {name.lower()} segments...")

    info = get_track_counts(
        tracks,
        labels,
        directory,
        name
    )

    total_segments = sum(
        item[2]
        for item in info
    )

    print(
        f"{name} segments: {total_segments}"
    )

    X = np.empty(
        (
            total_segments,
            128,
            130,
            1
        ),
        dtype=np.float32
    )

    y = np.empty(
        total_segments,
        dtype=np.int32
    )

    position = 0

    print(
        f"Loading {name.lower()} into RAM..."
    )

    for path, label, count in tqdm(
        info,
        desc=name
    ):

        if count == 0:
            continue

        try:
            data = np.load(path)

            end = position + count

            X[position:end, :, :, 0] = (
                data + 80.0
            ) / 80.0

            y[position:end] = label

            position = end

        except Exception as e:
            print(
                f"\nSkipping {path}: {e}"
            )

    X = X[:position]
    y = y[:position]

    return X, y


print("\nPreparing training data...")

X_train, y_train = load_tracks_to_ram(
    train_ids,
    train_y_tracks,
    TRAIN_DIR,
    "Train"
)


print("\nPreparing validation data...")

X_val, y_val = load_tracks_to_ram(
    val_ids,
    val_y_tracks,
    TRAIN_DIR,
    "Validation"
)


print(
    f"\nX_train shape: {X_train.shape}"
)

print(
    f"X_val shape: {X_val.shape}"
)

print("\nCreating TensorFlow datasets...")

train_dataset = tf.data.Dataset.from_tensor_slices(
    (X_train, y_train)
)

train_dataset = train_dataset.shuffle(
    buffer_size=min(10000, len(y_train)),
    seed=RANDOM_STATE,
    reshuffle_each_iteration=True
)

train_dataset = train_dataset.batch(
    BATCH_SIZE
)

train_dataset = train_dataset.prefetch(
    tf.data.AUTOTUNE
)


val_dataset = tf.data.Dataset.from_tensor_slices(
    (X_val, y_val)
)

val_dataset = val_dataset.batch(
    BATCH_SIZE
)

val_dataset = val_dataset.prefetch(
    tf.data.AUTOTUNE
)


print(
    f"Training batches: {len(train_dataset)}"
)

print(
    f"Validation batches: {len(val_dataset)}"
)


del X_train
del y_train


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
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=callbacks
)


del X_val
del y_val



print("\nLoading test data...\n")


X_test, y_test = load_tracks_to_ram(
    test_tracks,
    test_labels,
    TEST_DIR,
    "Test"
)


test_dataset = tf.data.Dataset.from_tensor_slices(
    (
        X_test,
        y_test
    )
)

test_dataset = test_dataset.batch(
    BATCH_SIZE
)

test_dataset = test_dataset.prefetch(
    tf.data.AUTOTUNE
)


print("\nFinal test evaluation...\n")


loss, accuracy = model.evaluate(
    test_dataset,
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