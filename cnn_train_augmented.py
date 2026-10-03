import os
import random
import gc

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# =========================
# Paths
# =========================

DATA_DIR = "/content/drive/MyDrive/cnn_augmented_data"
MODEL_DIR = "/content/drive/MyDrive/cnn_augmented_model"

TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
TEST_DIR = os.path.join(DATA_DIR, "test")

METADATA_DIR = os.path.join(
    DATA_DIR,
    "metadata"
)


# =========================
# Settings
# =========================

BATCH_SIZE = 192
EPOCHS = 20
RANDOM_STATE = 49


random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)


# =========================
# Load metadata
# =========================

classes = np.load(
    os.path.join(
        METADATA_DIR,
        "classes.npy"
    ),
    allow_pickle=True
)

train_tracks = np.load(
    os.path.join(
        METADATA_DIR,
        "train_tracks.npy"
    )
)

val_tracks = np.load(
    os.path.join(
        METADATA_DIR,
        "val_tracks.npy"
    )
)

test_tracks = np.load(
    os.path.join(
        METADATA_DIR,
        "test_tracks.npy"
    )
)


print("Classes:")
print(classes)

print(
    f"\nTraining tracks: {len(train_tracks)}"
)

print(
    f"Validation tracks: {len(val_tracks)}"
)

print(
    f"Test tracks: {len(test_tracks)}"
)


# =========================
# Shard utilities
# =========================

def get_shards(directory):

    return sorted(
        [
            os.path.join(
                directory,
                filename
            )
            for filename in os.listdir(directory)
            if filename.endswith(".npz")
        ]
    )


train_shards = get_shards(TRAIN_DIR)
val_shards = get_shards(VAL_DIR)
test_shards = get_shards(TEST_DIR)


print(
    f"\nTrain shards: {len(train_shards)}"
)

print(
    f"Validation shards: {len(val_shards)}"
)

print(
    f"Test shards: {len(test_shards)}"
)


# =========================
# Count segments
# =========================

def count_segments(shards):

    total = 0

    for shard in shards:

        with np.load(
            shard,
            allow_pickle=False
        ) as data:

            total += len(
                data["y"]
            )

    return total


train_segments = count_segments(
    train_shards
)

val_segments = count_segments(
    val_shards
)

test_segments = count_segments(
    test_shards
)


print(
    f"\nTrain segments: {train_segments}"
)

print(
    f"Validation segments: {val_segments}"
)

print(
    f"Test segments: {test_segments}"
)


# =========================
# Batch generator
# =========================

def batch_generator(
    shards,
    batch_size,
    shuffle=False
):

    while True:

        shard_order = np.arange(
            len(shards)
        )

        if shuffle:
            np.random.shuffle(
                shard_order
            )

        X_buffer = []
        y_buffer = []

        for shard_index in shard_order:

            shard_path = shards[
                shard_index
            ]

            with np.load(
                shard_path,
                allow_pickle=False
            ) as data:

                X = data["X"]
                y = data["y"]

                if shuffle:

                    order = np.random.permutation(
                        len(y)
                    )

                    X = X[order]
                    y = y[order]

                for i in range(
                    len(y)
                ):

                    X_buffer.append(
                        X[i]
                    )

                    y_buffer.append(
                        y[i]
                    )

                    if len(
                        y_buffer
                    ) >= batch_size:

                        batch_X = np.asarray(
                            X_buffer,
                            dtype=np.float32
                        )

                        batch_y = np.asarray(
                            y_buffer,
                            dtype=np.int32
                        )

                        batch_X = (
                            batch_X + 80.0
                        ) / 80.0

                        batch_X = batch_X[
                            ..., np.newaxis
                        ]

                        yield (
                            batch_X,
                            batch_y
                        )

                        X_buffer.clear()
                        y_buffer.clear()

        if y_buffer:

            batch_X = np.asarray(
                X_buffer,
                dtype=np.float32
            )

            batch_y = np.asarray(
                y_buffer,
                dtype=np.int32
            )

            batch_X = (
                batch_X + 80.0
            ) / 80.0

            batch_X = batch_X[
                ..., np.newaxis
            ]

            yield (
                batch_X,
                batch_y
            )

        if not shuffle:
            break


# =========================
# TensorFlow datasets
# =========================

output_signature = (
    tf.TensorSpec(
        shape=(
            None,
            128,
            130,
            1
        ),
        dtype=tf.float32
    ),
    tf.TensorSpec(
        shape=(None,),
        dtype=tf.int32
    )
)


train_dataset = tf.data.Dataset.from_generator(
    lambda: batch_generator(
        train_shards,
        BATCH_SIZE,
        shuffle=True
    ),
    output_signature=output_signature
)

val_dataset = tf.data.Dataset.from_generator(
    lambda: batch_generator(
        val_shards,
        BATCH_SIZE,
        shuffle=False
    ),
    output_signature=output_signature
)

test_dataset = tf.data.Dataset.from_generator(
    lambda: batch_generator(
        test_shards,
        BATCH_SIZE,
        shuffle=False
    ),
    output_signature=output_signature
)


train_dataset = train_dataset.prefetch(
    tf.data.AUTOTUNE
)

val_dataset = val_dataset.prefetch(
    tf.data.AUTOTUNE
)

test_dataset = test_dataset.prefetch(
    tf.data.AUTOTUNE
)


train_steps = int(
    np.ceil(
        train_segments / BATCH_SIZE
    )
)

val_steps = int(
    np.ceil(
        val_segments / BATCH_SIZE
    )
)

test_steps = int(
    np.ceil(
        test_segments / BATCH_SIZE
    )
)


print(
    f"\nTraining steps: {train_steps}"
)

print(
    f"Validation steps: {val_steps}"
)

print(
    f"Test steps: {test_steps}"
)


# =========================
# CNN model
# =========================

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

    layers.Dropout(0.5),

    layers.Dense(
        len(classes),
        activation="softmax"
    )
])


model.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=0.00045
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


model.summary()


# =========================
# Callbacks
# =========================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


callbacks = [

    keras.callbacks.ModelCheckpoint(
        os.path.join(
            MODEL_DIR,
            "best_cnn_augmented.keras"
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
        monitor="val_accuracy",
        patience=5,
        mode="max",
        restore_best_weights=True,
        verbose=1
    )
]


# =========================
# Training
# =========================

print(
    "\nStarting augmented CNN training...\n"
)


history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    steps_per_epoch=train_steps,
    validation_steps=val_steps,
    epochs=EPOCHS,
    callbacks=callbacks
)


# =========================
# Test
# =========================

print(
    "\nFinal test evaluation...\n"
)


loss, accuracy = model.evaluate(
    test_dataset,
    steps=test_steps,
    verbose=1
)


print(
    f"\nAugmented CNN Test Accuracy: "
    f"{accuracy * 100:.2f}%"
)


# =========================
# Save
# =========================

model.save(
    os.path.join(
        MODEL_DIR,
        "final_cnn_augmented.keras"
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


print(
    "\nModel saved to:"
)

print(
    MODEL_DIR
)


del train_dataset
del val_dataset
del test_dataset

gc.collect()