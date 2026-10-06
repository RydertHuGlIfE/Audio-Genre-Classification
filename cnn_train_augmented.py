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
EPOCHS = 15
RANDOM_STATE = 49

N_MELS = 128
TIME_FRAMES = 130
NUM_CLASSES = 8
MIXUP_ALPHA = 0.2


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
        f'rsync -aL '
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
    shuffle=False,
    buffer_shards=16
):

    shard_order = np.arange(
        len(shards)
    )

    if shuffle:

        np.random.shuffle(
            shard_order
        )

    for chunk_start in range(
        0,
        len(shard_order),
        buffer_shards
    ):

        chunk_indices = shard_order[
            chunk_start:chunk_start + buffer_shards
        ]

        X_list = []
        y_list = []

        for idx in chunk_indices:

            with np.load(
                shards[idx],
                allow_pickle=False
            ) as data:

                X_list.append(data["X"])
                y_list.append(data["y"])

        X_buf = np.concatenate(
            X_list,
            axis=0
        )

        y_buf = np.concatenate(
            y_list,
            axis=0
        )

        if shuffle:

            perm = np.random.permutation(
                len(y_buf)
            )

            X_buf = X_buf[perm]
            y_buf = y_buf[perm]

        for i in range(
            0,
            len(y_buf),
            batch_size
        ):

            batch_X = X_buf[
                i:i + batch_size
            ].astype(np.float32)

            batch_y = y_buf[
                i:i + batch_size
            ].astype(np.int32)

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
    shuffle=False,
    repeat=False
):

    dataset = tf.data.Dataset.from_generator(

        lambda: batch_generator(
            shards,
            batch_size,
            shuffle
        ),

        output_signature=output_signature
    )

    if repeat:
        dataset = dataset.repeat()

    return dataset


def to_one_hot(images, labels):

    labels = tf.one_hot(
        labels,
        depth=NUM_CLASSES,
        dtype=tf.float32
    )

    return images, labels


def mixup_batch(images, labels, alpha=MIXUP_ALPHA):

    batch_size = tf.shape(images)[0]

    perm = tf.random.shuffle(
        tf.range(batch_size)
    )

    images_shuffled = tf.gather(
        images,
        perm
    )

    labels_shuffled = tf.gather(
        labels,
        perm
    )

    # Sample lambda from Beta(alpha, alpha) via two Gamma distributions
    gamma1 = tf.random.gamma(
        shape=[batch_size, 1, 1, 1],
        alpha=alpha
    )

    gamma2 = tf.random.gamma(
        shape=[batch_size, 1, 1, 1],
        alpha=alpha
    )

    lam_x = gamma1 / (gamma1 + gamma2)

    lam_y = tf.reshape(
        lam_x,
        [batch_size, 1]
    )

    mixed_images = (
        lam_x * images +
        (1.0 - lam_x) * images_shuffled
    )

    mixed_labels = (
        lam_y * labels +
        (1.0 - lam_y) * labels_shuffled
    )

    return mixed_images, mixed_labels


def load_shards_to_ram(
    shards,
    name="Validation"
):
    print(
        f"\nLoading {name} shards into RAM "
        f"({len(shards)} shards)..."
    )

    X_list = []
    y_list = []
    track_ids_list = []

    for shard in shards:
        with np.load(
            shard,
            allow_pickle=False
        ) as data:
            X_list.append(data["X"])
            y_list.append(data["y"])
            if "track_ids" in data:
                track_ids_list.append(
                    data["track_ids"]
                )

    X = np.concatenate(
        X_list,
        axis=0
    ).astype(np.float32)

    y = np.concatenate(
        y_list,
        axis=0
    ).astype(np.int32)

    # Normalize to [0, 1]
    X = (X + 80.0) / 80.0
    X = X[..., np.newaxis]

    track_ids = (
        np.concatenate(
            track_ids_list,
            axis=0
        )
        if track_ids_list
        else None
    )

    print(
        f"{name} loaded: "
        f"X={X.shape} ({X.nbytes / (1024**2):.1f} MB), "
        f"y={y.shape}"
    )

    return X, y, track_ids


# Train continues to stream from generator to conserve RAM
train_dataset = make_dataset(
    train_shards,
    BATCH_SIZE,
    shuffle=True,
    repeat=True
)

train_dataset = (
    train_dataset
    .map(
        to_one_hot,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    .map(
        lambda x, y: mixup_batch(x, y, alpha=MIXUP_ALPHA),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    .prefetch(
        tf.data.AUTOTUNE
    )
)

# Load validation directly into RAM (fixed size, ~670MB)
X_val, y_val, _ = load_shards_to_ram(
    val_shards,
    name="Validation"
)

y_val_one_hot = tf.one_hot(
    y_val,
    depth=NUM_CLASSES,
    dtype=tf.float32
)

val_dataset = tf.data.Dataset.from_tensor_slices(
    (X_val, y_val_one_hot)
).batch(
    BATCH_SIZE
).prefetch(
    tf.data.AUTOTUNE
)

train_steps = int(
    np.ceil(
        train_segments /
        BATCH_SIZE
    )
)

val_steps = int(
    np.ceil(
        len(y_val) /
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


# =========================
# SpecAugment
# =========================

class SpecAugment(
    layers.Layer
):

    def __init__(
        self,
        freq_mask=16,
        time_mask=16,
        n_freq_masks=2,
        n_time_masks=2,
        **kwargs
    ):

        super().__init__(
            **kwargs
        )

        self.freq_mask = freq_mask
        self.time_mask = time_mask
        self.n_freq_masks = n_freq_masks
        self.n_time_masks = n_time_masks

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

        x = inputs

        # Frequency masking: per-sample
        for _ in range(self.n_freq_masks):

            f = tf.random.uniform(
                [batch],
                minval=0,
                maxval=self.freq_mask + 1,
                dtype=tf.int32
            )

            max_f0 = tf.maximum(
                1,
                freq - f + 1
            )

            f0 = tf.random.uniform(
                [batch],
                minval=0,
                maxval=tf.int32.max,
                dtype=tf.int32
            ) % max_f0

            freq_indices = tf.range(freq)[None, :, None]
            f0_exp = f0[:, None, None]
            f_exp = f[:, None, None]

            mask = (
                (freq_indices >= f0_exp) &
                (freq_indices < (f0_exp + f_exp))
            )

            mask = 1.0 - tf.cast(
                mask,
                inputs.dtype
            )

            x = x * mask[..., None]

        # Time masking: per-sample
        for _ in range(self.n_time_masks):

            t = tf.random.uniform(
                [batch],
                minval=0,
                maxval=self.time_mask + 1,
                dtype=tf.int32
            )

            max_t0 = tf.maximum(
                1,
                time - t + 1
            )

            t0 = tf.random.uniform(
                [batch],
                minval=0,
                maxval=tf.int32.max,
                dtype=tf.int32
            ) % max_t0

            time_indices = tf.range(time)[None, None, :]
            t0_exp = t0[:, None, None]
            t_exp = t[:, None, None]

            mask = (
                (time_indices >= t0_exp) &
                (time_indices < (t0_exp + t_exp))
            )

            mask = 1.0 - tf.cast(
                mask,
                inputs.dtype
            )

            x = x * mask[..., None]

        return x

    def get_config(self):

        config = super().get_config()

        config.update({
            "freq_mask": self.freq_mask,
            "time_mask": self.time_mask,
            "n_freq_masks": self.n_freq_masks,
            "n_time_masks": self.n_time_masks
        })

        return config


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
    freq_mask=16,
    time_mask=16,
    n_freq_masks=2,
    n_time_masks=2
)(inputs)


x = layers.Conv2D(
    16,
    3,
    padding="same",
    use_bias=False
)(x)

x = layers.BatchNormalization()(x)

x = layers.ReLU()(x)


x = residual_block(
    x,
    16
)

x = residual_block(
    x,
    32,
    stride=2
)

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


x = layers.GlobalAveragePooling2D()(x)


x = layers.Dense(
    128,
    activation="relu"
)(x)

x = layers.BatchNormalization()(x)

x = layers.Dropout(
    0.50
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
# Learning rate & Optimizer
# =========================

initial_lr = 3e-4

lr_schedule = keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=initial_lr,
    decay_steps=EPOCHS * train_steps,
    alpha=0.05
)


optimizer = keras.optimizers.AdamW(
    learning_rate=lr_schedule,
    weight_decay=1e-2
)


model.compile(
    optimizer=optimizer,
    loss=keras.losses.CategoricalCrossentropy(
        label_smoothing=0.1
    ),
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
    "\nStarting V2 augmented "
    "ResNet CNN training...\n"
)


history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    steps_per_epoch=train_steps,
    epochs=EPOCHS,
    callbacks=callbacks
)


# =========================
# Test
# =========================

print(
    "\nLoading test data for evaluation...\n"
)

X_test, y_test, test_track_ids = load_shards_to_ram(
    test_shards,
    name="Test"
)

y_test_one_hot = tf.one_hot(
    y_test,
    depth=NUM_CLASSES,
    dtype=tf.float32
)

test_dataset = tf.data.Dataset.from_tensor_slices(
    (X_test, y_test_one_hot)
).batch(
    BATCH_SIZE
).prefetch(
    tf.data.AUTOTUNE
)

print(
    "\nEvaluating segment-level test accuracy...\n"
)

loss, segment_accuracy = model.evaluate(
    test_dataset,
    verbose=1
)

print(
    f"\nV2 Segment-Level Test Accuracy: "
    f"{segment_accuracy * 100:.2f}%\n"
)

# Track-level evaluation (Softmax Probability Averaging per track)
if test_track_ids is not None:
    print(
        "Computing Track-Level (Song-Level) "
        "Accuracy via Softmax Averaging..."
    )

    test_preds = model.predict(
        test_dataset,
        verbose=1
    )

    unique_tracks = np.unique(test_track_ids)
    track_correct = 0

    for tid in unique_tracks:
        idx = np.where(test_track_ids == tid)[0]
        avg_prob = np.mean(test_preds[idx], axis=0)
        track_pred = np.argmax(avg_prob)
        track_true = y_test[idx[0]]
        if track_pred == track_true:
            track_correct += 1

    track_accuracy = (
        track_correct / len(unique_tracks)
    ) * 100.0

    print(
        "\n" + "=" * 50
    )
    print(
        f"Segment-Level Test Accuracy: "
        f"{segment_accuracy * 100:.2f}%"
    )
    print(
        f"Track-Level Test Accuracy:   "
        f"{track_accuracy:.2f}% "
        f"({track_correct}/{len(unique_tracks)} tracks)"
    )
    print(
        "=" * 50 + "\n"
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
del X_val
del y_val
del X_test
del y_test

gc.collect()