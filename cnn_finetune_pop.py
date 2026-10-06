import argparse
import math
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score
from tensorflow import keras


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.keras"
DATA_DIR_CANDIDATES = (
    BASE_DIR / "cnn_augmented_data",
    Path("/content/cnn_augmented_data"),
    Path("/content/drive/MyDrive/cnn_augmented_data"),
)
BATCH_SIZE = 128
EPOCHS = 8
POP_CLASS_WEIGHT = 2.0
LEARNING_RATE = 1e-5
MAX_MACRO_F1_DROP = 0.02
PATIENCE = 3
RANDOM_STATE = 49
N_MELS = 128
TIME_FRAMES = 130


class SpecAugment(keras.layers.Layer):
    def __init__(
        self,
        freq_mask=16,
        time_mask=16,
        n_freq_masks=2,
        n_time_masks=2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.freq_mask = freq_mask
        self.time_mask = time_mask
        self.n_freq_masks = n_freq_masks
        self.n_time_masks = n_time_masks

    def call(self, inputs, training=None):
        return inputs

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "freq_mask": self.freq_mask,
                "time_mask": self.time_mask,
                "n_freq_masks": self.n_freq_masks,
                "n_time_masks": self.n_time_masks,
            }
        )
        return config


def resolve_data_dir(requested):
    candidates = [Path(requested).expanduser()] if requested else DATA_DIR_CANDIDATES
    for candidate in candidates:
        if (candidate / "metadata" / "classes.npy").is_file():
            return candidate.resolve()

    searched = "\n".join(f"  {path}" for path in candidates)
    raise FileNotFoundError(
        "Augmented dataset metadata was not found. The existing trainer expects "
        "sharded .npz data with a metadata directory. Searched:\n" + searched
    )


def load_split_metadata(data_dir):
    metadata_dir = data_dir / "metadata"
    classes = np.load(metadata_dir / "classes.npy", allow_pickle=False)
    split_tracks = {
        split: np.load(metadata_dir / f"{split}_tracks.npy", allow_pickle=False)
        for split in ("train", "val", "test")
    }
    split_labels = {
        split: np.load(metadata_dir / f"{split}_labels.npy", allow_pickle=False)
        for split in ("train", "val")
    }

    reference_classes_path = BASE_DIR / "cnn_data" / "classes.npy"
    if reference_classes_path.is_file():
        reference_classes = np.load(reference_classes_path, allow_pickle=True)
        if classes.tolist() != reference_classes.tolist():
            raise ValueError(
                "Augmented dataset class order does not match cnn_data/classes.npy."
            )

    if len(classes) != 8 or "Pop" not in classes.tolist():
        raise ValueError(
            f"Expected the model's 8-class mapping including Pop; got {classes.tolist()}"
        )

    split_sets = {}
    for split, tracks in split_tracks.items():
        if len(np.unique(tracks)) != len(tracks):
            raise ValueError(f"Duplicate track IDs in {split} metadata.")
        split_sets[split] = set(map(int, tracks.tolist()))

    if (
        split_sets["train"] & split_sets["val"]
        or split_sets["train"] & split_sets["test"]
        or split_sets["val"] & split_sets["test"]
    ):
        raise ValueError("Train, validation, and test track IDs overlap.")

    track_labels = {}
    for split, labels in split_labels.items():
        tracks = split_tracks[split]
        if len(tracks) != len(labels):
            raise ValueError(f"{split} track and label metadata lengths differ.")
        if np.any(labels < 0) or np.any(labels >= len(classes)):
            raise ValueError(f"Invalid class index in {split} labels.")
        track_labels[split] = dict(zip(map(int, tracks), map(int, labels)))

    return classes, split_sets, track_labels


def verify_model_split(data_dir, model_split_dir, classes, split_sets, track_labels):
    reference_dir = Path(model_split_dir).expanduser()
    if (reference_dir / "metadata").is_dir():
        reference_dir = reference_dir / "metadata"

    reference_classes = np.load(reference_dir / "classes.npy", allow_pickle=False)
    if classes.tolist() != reference_classes.tolist():
        raise ValueError("Augmented class order differs from the model-run class mapping.")

    for split in ("train", "val", "test"):
        reference_tracks = np.load(
            reference_dir / f"{split}_tracks.npy", allow_pickle=False
        )
        if set(map(int, reference_tracks.tolist())) != split_sets[split]:
            raise ValueError(
                f"Augmented {split} track IDs do not match the model-run split."
            )

        reference_labels_path = reference_dir / f"{split}_labels.npy"
        if reference_labels_path.is_file():
            reference_labels = np.load(reference_labels_path, allow_pickle=False)
            if len(reference_tracks) != len(reference_labels):
                raise ValueError(f"Model-run {split} track and label lengths differ.")
            expected = dict(zip(map(int, reference_tracks), map(int, reference_labels)))
            current = track_labels.get(split)
            if current is None:
                current_tracks = np.load(
                    data_dir / "metadata" / f"{split}_tracks.npy", allow_pickle=False
                )
                current_labels = np.load(
                    data_dir / "metadata" / f"{split}_labels.npy", allow_pickle=False
                )
                current = dict(zip(map(int, current_tracks), map(int, current_labels)))
            if expected != current:
                raise ValueError(f"Augmented {split} labels do not match the model-run labels.")


def shard_paths(data_dir, split):
    split_dir = data_dir / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Missing {split} directory: {split_dir}")
    shards = sorted(split_dir.glob("*.npz"))
    if not shards:
        raise FileNotFoundError(f"No .npz shards found in {split_dir}")
    return shards


def inspect_shards(shards, allowed_tracks, expected_labels, split, batch_size):
    total_segments = 0
    steps = 0
    for shard_path in shards:
        with np.load(shard_path, allow_pickle=False) as data:
            missing = {"X", "y", "track_ids"} - set(data.files)
            if missing:
                raise ValueError(f"{shard_path} is missing keys: {sorted(missing)}")
            features = data["X"]
            labels = data["y"]
            track_ids = data["track_ids"]

            if features.ndim != 3 or features.shape[1:] != (N_MELS, TIME_FRAMES):
                raise ValueError(f"Unexpected feature shape in {shard_path}: {features.shape}")
            if len(features) != len(labels) or len(labels) != len(track_ids):
                raise ValueError(f"Feature, label, and track counts differ in {shard_path}.")

            for track_id, label in zip(track_ids.tolist(), labels.tolist()):
                track_id = int(track_id)
                if track_id not in allowed_tracks:
                    raise ValueError(f"{split} shard contains out-of-split track {track_id}.")
                if expected_labels.get(track_id) != int(label):
                    raise ValueError(f"Label mismatch for track {track_id} in {shard_path}.")

            total_segments += len(labels)
            steps += math.ceil(len(labels) / batch_size)

    return total_segments, steps


def train_batches(shards, batch_size, pop_label, pop_weight):
    for shard_index in np.random.permutation(len(shards)):
        with np.load(shards[shard_index], allow_pickle=False) as data:
            features = data["X"].astype(np.float32)
            labels = data["y"].astype(np.int32)

        order = np.random.permutation(len(labels))
        for start in range(0, len(labels), batch_size):
            indices = order[start : start + batch_size]
            batch_x = (features[indices] + 80.0) / 80.0
            batch_x = batch_x[..., np.newaxis]
            batch_y = labels[indices]
            weights = np.where(batch_y == pop_label, pop_weight, 1.0)
            yield batch_x, batch_y, weights.astype(np.float32)


def make_train_dataset(shards, batch_size, pop_label, pop_weight):
    signature = (
        tf.TensorSpec((None, N_MELS, TIME_FRAMES, 1), tf.float32),
        tf.TensorSpec((None,), tf.int32),
        tf.TensorSpec((None,), tf.float32),
    )
    return tf.data.Dataset.from_generator(
        lambda: train_batches(shards, batch_size, pop_label, pop_weight),
        output_signature=signature,
    ).repeat().prefetch(tf.data.AUTOTUNE)


def evaluate_track_metrics(model, shards, batch_size, class_count, pop_label):
    probabilities = {}
    track_labels = {}

    for shard_path in shards:
        with np.load(shard_path, allow_pickle=False) as data:
            features = data["X"].astype(np.float32)
            labels = data["y"].astype(np.int32)
            track_ids = data["track_ids"].astype(np.int64)

        for start in range(0, len(labels), batch_size):
            end = start + batch_size
            batch_x = (features[start:end] + 80.0) / 80.0
            batch_x = batch_x[..., np.newaxis]
            batch_probs = model(batch_x, training=False).numpy()

            for track_id, label, prediction in zip(
                track_ids[start:end], labels[start:end], batch_probs
            ):
                track_id = int(track_id)
                if track_id in track_labels and track_labels[track_id] != int(label):
                    raise ValueError(f"Conflicting validation labels for track {track_id}.")
                track_labels[track_id] = int(label)
                if track_id not in probabilities:
                    probabilities[track_id] = np.zeros(class_count, dtype=np.float64)
                probabilities[track_id] += prediction

    true_labels = np.asarray([track_labels[track_id] for track_id in probabilities])
    predicted_labels = np.asarray(
        [int(np.argmax(probabilities[track_id])) for track_id in probabilities]
    )
    class_f1 = f1_score(
        true_labels,
        predicted_labels,
        labels=np.arange(class_count),
        average=None,
        zero_division=0,
    )
    return {
        "pop_f1": float(class_f1[pop_label]),
        "macro_f1": float(np.mean(class_f1)),
        "pop_tracks": int(np.sum(true_labels == pop_label)),
    }


class PopCheckpoint(keras.callbacks.Callback):
    def __init__(
        self,
        val_shards,
        output_path,
        class_count,
        pop_label,
        batch_size,
        macro_f1_drop,
        patience,
    ):
        super().__init__()
        self.val_shards = val_shards
        self.output_path = output_path
        self.class_count = class_count
        self.pop_label = pop_label
        self.batch_size = batch_size
        self.macro_f1_drop = macro_f1_drop
        self.patience = patience
        self.best_pop_f1 = -1.0
        self.baseline_macro_f1 = 0.0
        self.wait = 0

    def on_train_begin(self, logs=None):
        baseline = evaluate_track_metrics(
            self.model,
            self.val_shards,
            self.batch_size,
            self.class_count,
            self.pop_label,
        )
        self.best_pop_f1 = baseline["pop_f1"]
        self.baseline_macro_f1 = baseline["macro_f1"]
        self.model.save(self.output_path)
        print(
            "Baseline validation: "
            f"Pop F1={baseline['pop_f1']:.4f}, "
            f"macro F1={baseline['macro_f1']:.4f}, "
            f"Pop tracks={baseline['pop_tracks']}"
        )

    def on_epoch_end(self, epoch, logs=None):
        logs = logs if logs is not None else {}
        metrics = evaluate_track_metrics(
            self.model,
            self.val_shards,
            self.batch_size,
            self.class_count,
            self.pop_label,
        )
        logs["val_pop_f1"] = metrics["pop_f1"]
        logs["val_macro_f1"] = metrics["macro_f1"]
        print(
            f"Validation tracks: Pop F1={metrics['pop_f1']:.4f}, "
            f"macro F1={metrics['macro_f1']:.4f}"
        )

        macro_ok = metrics["macro_f1"] >= self.baseline_macro_f1 - self.macro_f1_drop
        if metrics["pop_f1"] > self.best_pop_f1 and macro_ok:
            self.best_pop_f1 = metrics["pop_f1"]
            self.wait = 0
            self.model.save(self.output_path)
            print(f"Saved improved model: {self.output_path}")
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.model.stop_training = True


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune model.keras to improve Pop using augmented train shards."
    )
    parser.add_argument("--data-dir", help="Augmented dataset root; defaults to detected project paths.")
    parser.add_argument(
        "--model-split-dir",
        required=True,
        help="Metadata directory (or its parent dataset directory) for the model's original split.",
    )
    parser.add_argument("--model", default=str(MODEL_PATH), help="Starting Keras model.")
    parser.add_argument(
        "--output",
        default=str(BASE_DIR / "model_pop_finetuned.keras"),
        help="Output model path; the original model is not overwritten.",
    )
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--pop-weight", type=float, default=POP_CLASS_WEIGHT)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.pop_weight <= 1:
        raise ValueError("epochs/batch-size must be positive and pop-weight must exceed 1.")

    np.random.seed(RANDOM_STATE)
    tf.random.set_seed(RANDOM_STATE)

    data_dir = resolve_data_dir(args.data_dir)
    classes, split_sets, track_labels = load_split_metadata(data_dir)
    verify_model_split(
        data_dir,
        args.model_split_dir,
        classes,
        split_sets,
        track_labels,
    )
    pop_label = classes.tolist().index("Pop")
    train_shards = shard_paths(data_dir, "train")
    val_shards = shard_paths(data_dir, "val")

    train_segments, train_steps = inspect_shards(
        train_shards,
        split_sets["train"],
        track_labels["train"],
        "train",
        args.batch_size,
    )
    inspect_shards(
        val_shards,
        split_sets["val"],
        track_labels["val"],
        "val",
        args.batch_size,
    )

    model = keras.models.load_model(
        args.model,
        custom_objects={"SpecAugment": SpecAugment},
        compile=False,
    )
    if model.input_shape[1:] != (N_MELS, TIME_FRAMES, 1):
        raise ValueError(f"Unexpected model input shape: {model.input_shape}")
    if model.output_shape[-1] != len(classes):
        raise ValueError("Model output size does not match the augmented class mapping.")

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss=keras.losses.SparseCategoricalCrossentropy(),
    )

    print(f"Dataset: {data_dir}")
    print(f"Training segments: {train_segments}; steps per epoch: {train_steps}")
    print(f"Pop class index: {pop_label}; training weight: {args.pop_weight:g}")
    print("Validation uses unaugmented validation shards; test shards are not opened.")

    model.fit(
        make_train_dataset(train_shards, args.batch_size, pop_label, args.pop_weight),
        steps_per_epoch=train_steps,
        epochs=args.epochs,
        callbacks=[
            PopCheckpoint(
                val_shards,
                str(output_path),
                len(classes),
                pop_label,
                args.batch_size,
                MAX_MACRO_F1_DROP,
                PATIENCE,
            )
        ],
        verbose=1,
    )
    np.save(output_path.with_name(f"{output_path.stem}_classes.npy"), classes)
    print(f"Selected model: {output_path}")


if __name__ == "__main__":
    main()