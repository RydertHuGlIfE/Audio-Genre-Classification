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

DRIVE_DATA_DIR = (
    "/content/drive/MyDrive/"
    "cnn_augmented_data"
)

LOCAL_DATA_DIR = (
    "/content/cnn_augmented_data"
)

MODEL_DIR = (
    "/content/drive/MyDrive/"
    "cnn_augmented_model_v2"
)


# =========================
# Settings
# =========================

BATCH_SIZE = 128
EPOCHS = 60
RANDOM_STATE = 49

N_MELS = 128
TIME_FRAMES = 130
NUM_CLASSES = 8


random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)


# =========================
# Copy processed data locally
# =========================

if not os.path.exists(
    os.path.join(
        LOCAL_DATA_DIR,
        "metadata"
    )
):

    print(
        "Copying processed dataset "
        "from Drive to local storage..."
    )

    os.system(
        f'rsync -a '
        f'"{DRIVE_DATA_DIR}/" '
        f'"{LOCAL_DATA_DIR}/"'
    )


DATA_DIR = LOCAL_DATA_DIR

TRAIN_DIR = os.path.join(
    DATA_DIR,
    "train"
)

VAL_DIR = os.path.join(
    DATA_DIR,
    "val"
)

TEST_DIR = os.path.join(
    DATA_DIR,
    "test"
)

METADATA_DIR = os.path.join(
    DATA_DIR,
    "metadata"
)


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
    f"\nTraining tracks: "
    f"{len(train_tracks)}"
)

print(
    f"Validation tracks: "
    f"{len(val_tracks)}"
)

print(
    f"Test tracks: "
    f"{len(test_tracks)}"
)


# =========================
# Shards
# =========================

def get_shards(directory):

    return sorted(
        [
            os.path.join(
                directory,
                filename
            )
            for filename in os.listdir(
                directory
            )
            if filename.endswith(".npz")
        ]
    )


train_shards = get_shards(
    TRAIN_DIR
)

val_shards = get_shards(
    VAL_DIR
)

test_shards = get_shards(
    TEST_DIR
)


print(
    f"\nTrain shards: "
    f"{len(train_shards)}"
)

print(
    f"Validation shards: "
    f"{len(val_shards)}"
)

print(
    f"Test shards: "
    f"{len(test_shards)}"
)


# =========================
# Segment counts
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
    f"\nTrain segments: "
    f"{train_segments}"
)

print(
    f"Validation segments: "
    f"{val_segments}"
)

print(
    f"Test segments: "
    f"{test_segments}"
)


# =========================
# Generator
# =========================

def batch_generator(
    shards,
    batch_size,
    shuffle=False
):

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
                ) == batch_size:

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

                    batch_X = (
                        batch_X[..., np.newaxis]
                    )

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

        batch_X = (
            batch_X[..., np.newaxis]
        )

        yield (
            batch_X,
            batch_y
        )


# =========================
# TensorFlow datasets
# =========================

output_signature = (

    tf.TensorSpec(
        shape=(
            None,
            N_MELS,
            TIME_FRAMES,
            1
        ),
        dtype=tf.float32
    ),

    tf.TensorSpec(
        shape=(None,),
        dtype=tf.int32
    )
)


def make_dataset(
    shards,
    batch_size,
    shuffle=False
):

    dataset = tf.data.Dataset.from_generator(

        lambda: batch_generator(
            shards,
            batch_size,
            shuffle
        ),

        output_signature=output_signature
    )

    return dataset.prefetch(
        tf.data.AUTOTUNE
    )


train_dataset = make_dataset(
    train_shards,
    BATCH_SIZE,
    shuffle=True
)

val_dataset = make_dataset(
    val_shards,
    BATCH_SIZE,
    shuffle=False
)

test_dataset = make_dataset(
    test_shards,
    BATCH_SIZE,
    shuffle=False
)


train_steps = int(
    np.ceil(
        train_segments /
        BATCH_SIZE
    )
)

val_steps = int(
    np.ceil(
        val_segments /
        BATCH_SIZE
    )
)

test_steps = int(
    np.ceil(
        test_segments /
        BATCH_SIZE
    )
)


print(
    f"\nTraining steps: "
    f"{train_steps}"
)

print(
    f"Validation steps: "
    f"{val_steps}"
)

print(
    f"Test steps: "
    f"{test_steps}"
)


# =========================
# SpecAugment
# =========================

class SpecAugment(
    layers.Layer
):

    def __init__(
        self,
        freq_mask=12,
        time_mask=12,
        **kwargs
    ):

        super().__init__(
            **kwargs
        )

        self.freq_mask = freq_mask
        self.time_mask = time_mask

    def call(
        self,
        inputs,
        training=False
    ):

        if not training:

            return inputs

        shape = tf.shape(
            inputs
        )

        batch = shape[0]
        freq = shape[1]
        time = shape[2]

        # Frequency masking

        f = tf.random.uniform(
            [],
            minval=0,
            maxval=self.freq_mask + 1,
            dtype=tf.int32
        )

        f0 = tf.random.uniform(
            [],
            minval=0,
            maxval=tf.maximum(
                1,
                freq - f + 1
            ),
            dtype=tf.int32
        )

        freq_mask = (
            tf.range(freq)[None, :, None]
            >= f0
        ) & (
            tf.range(freq)[None, :, None]
            < f0 + f
        )

        freq_mask = tf.cast(
            freq_mask,
            inputs.dtype
        )

        freq_mask = tf.broadcast_to(
            freq_mask,
            [batch, freq, time]
        )

        freq_mask = (
            1.0 - freq_mask
        )

        # Time masking

        t = tf.random.uniform(
            [],
            minval=0,
            maxval=self.time_mask + 1,
            dtype=tf.int32
        )

        t0 = tf.random.uniform(
            [],
            minval=0,
            maxval=tf.maximum(
                1,
                time - t + 1
            ),
            dtype=tf.int32
        )

        time_mask = (
            tf.range(time)[None, None, :]
            >= t0
        ) & (
            tf.range(time)[None, None, :]
            < t0 + t
        )

        time_mask = tf.cast(
            time_mask,
            inputs.dtype
        )

        time_mask = tf.broadcast_to(
            time_mask,
            [batch, freq, time]
        )

        time_mask = (
            1.0 - time_mask
        )

        mask = (
            freq_mask *
            time_mask
        )

        mask = mask[..., None]

        return inputs * mask


# =========================
# Residual block
# =========================

def residual_block(
    x,
    filters,
    stride=1
):

    shortcut = x

    x = layers.Conv2D(
        filters,
        3,
        strides=stride,
        padding="same",
        use_bias=False
    )(x)

    x = layers.BatchNormalization()(x)

    x = layers.ReLU()(x)

    x = layers.Conv2D(
        filters,
        3,
        padding="same",
        use_bias=False
    )(x)

    x = layers.BatchNormalization()(x)

    if (
        stride != 1
        or shortcut.shape[-1] != filters
    ):

        shortcut = layers.Conv2D(
            filters,
            1,
            strides=stride,
            padding="same",
            use_bias=False
        )(shortcut)

        shortcut = layers.BatchNormalization()(
            shortcut
        )

    x = layers.Add()([
        x,
        shortcut
    ])

    x = layers.ReLU()(x)

    return x


# =========================
# Model
# =========================

inputs = keras.Input(
    shape=(
        N_MELS,
        TIME_FRAMES,
        1
    )
)


x = SpecAugment(
    freq_mask=12,
    time_mask=12
)(inputs)


x = layers.Conv2D(
    32,
    3,
    padding="same",
    use_bias=False
)(x)

x = layers.BatchNormalization()(x)

x = layers.ReLU()(x)


x = residual_block(
    x,
    32
)

x = residual_block(
    x,
    64,
    stride=2
)

x = residual_block(
    x,
    64
)

x = residual_block(
    x,
    128,
    stride=2
)

x = residual_block(
    x,
    128
)

x = residual_block(
    x,
    256,
    stride=2
)

x = residual_block(
    x,
    256
)


x = layers.GlobalAveragePooling2D()(x)


x = layers.Dense(
    256,
    activation="relu"
)(x)

x = layers.BatchNormalization()(x)

x = layers.Dropout(
    0.45
)(x)


outputs = layers.Dense(
    len(classes),
    activation="softmax"
)(x)


model = keras.Model(
    inputs,
    outputs
)


# =========================
# Learning rate
# =========================

initial_lr = 3e-4

lr_schedule = keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=initial_lr,
    decay_steps=EPOCHS * train_steps,
    alpha=0.05
)


optimizer = keras.optimizers.Adam(
    learning_rate=lr_schedule
)


model.compile(
    optimizer=optimizer,
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
            "best_cnn_augmented_v2.keras"
        ),
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1
    ),

    keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=8,
        mode="max",
        restore_best_weights=True,
        verbose=1
    )
]


# =========================
# Training
# =========================

print(
    "\nStarting V2 augmented "
    "ResNet CNN training...\n"
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
    f"\nV2 Test Accuracy: "
    f"{accuracy * 100:.2f}%"
)


# =========================
# Save
# =========================

model.save(
    os.path.join(
        MODEL_DIR,
        "final_cnn_augmented_v2.keras"
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