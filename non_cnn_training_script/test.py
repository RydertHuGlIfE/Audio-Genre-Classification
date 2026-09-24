import argparse
import os

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from tqdm import tqdm


FEATURE_FILE = "features_small.csv"
MODEL_DIRECTORIES = {
    "sigmoid": os.path.join("sigmoid_model", "best_weight"),
    "train": os.path.join("train_model", "best_weight"),
    "root": ".",
}


def load_dataset(weights_dir):
    if not os.path.exists(FEATURE_FILE):
        raise FileNotFoundError(
            f"Feature file not found: {FEATURE_FILE}. Run datapreprocess.py first."
        )

    model_path = os.path.join(weights_dir, "best_model.joblib")
    scaler_path = os.path.join(weights_dir, "scaler.joblib")
    encoder_path = os.path.join(weights_dir, "label_encoder.joblib")
    missing_files = [
        path
        for path in (model_path, scaler_path, encoder_path)
        if not os.path.exists(path)
    ]
    if missing_files:
        raise FileNotFoundError(
            "Missing best_weight file(s): " + ", ".join(missing_files)
        )

    data = pd.read_csv(FEATURE_FILE)
    target_col = "genre" if "genre" in data.columns else data.columns[-1]

    drop_cols = [target_col]
    if "track_id" in data.columns:
        drop_cols.append("track_id")
    if data.columns[0].startswith("Unnamed") or data.columns[0] == "0":
        drop_cols.append(data.columns[0])

    features = data.drop(columns=drop_cols)
    labels = joblib.load(encoder_path).transform(data[target_col])

    _, test_features, _, test_labels = train_test_split(
        features,
        labels,
        test_size=0.3,
        random_state=49,
        stratify=labels,
    )

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler, test_features, test_labels


def main():
    parser = argparse.ArgumentParser(description="Evaluate a saved genre model.")
    parser.add_argument(
        "--model",
        choices=MODEL_DIRECTORIES,
        default="sigmoid",
        help="Model artifacts to evaluate (default: sigmoid).",
    )
    args = parser.parse_args()

    weights_dir = MODEL_DIRECTORIES[args.model]
    model, scaler, test_features, test_labels = load_dataset(weights_dir)
    scaled_features = scaler.transform(test_features)

    predictions = []
    correct = 0
    for index in tqdm(range(len(test_labels)), desc="Testing test split", unit="song"):
        prediction = model.predict(scaled_features[index:index + 1])[0]
        predictions.append(prediction)
        if prediction == test_labels[index]:
            correct += 1

    predictions = pd.Series(predictions).to_numpy()
    wrong = len(test_labels) - correct

    print(f"Evaluation: {weights_dir}")
    print(f"Test split samples evaluated: {len(test_labels)}")
    print(f"Correct: {correct}")
    print(f"Wrong: {wrong}")
    print(f"accuracy: {accuracy_score(test_labels, predictions):.4f}")
    print(
        "precision: "
        f"{precision_score(test_labels, predictions, average='macro', zero_division=0):.4f}"
    )
    print(
        "recall: "
        f"{recall_score(test_labels, predictions, average='macro', zero_division=0):.4f}"
    )
    print(f"f1_macro: {f1_score(test_labels, predictions, average='macro'):.4f}")
    print(f"f1_micro: {f1_score(test_labels, predictions, average='micro'):.4f}")


if __name__ == "__main__":
    main()
