import os
import joblib
import pandas as pd
import numpy as np

def predict(feature_input):
    """
    Predict genre given a feature vector (518 elements) or a single-row Pandas DataFrame.
    """
    # Check sigmoid_model/best_weight first, then fall back to local root
    best_weight_dir = os.path.join("sigmoid_model", "best_weight")
    model_path = os.path.join(best_weight_dir, "best_model.joblib")
    scaler_path = os.path.join(best_weight_dir, "scaler.joblib")
    encoder_path = os.path.join(best_weight_dir, "label_encoder.joblib")

    if not (os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(encoder_path)):
        model_path = "best_model.joblib"
        scaler_path = "scaler.joblib"
        encoder_path = "label_encoder.joblib"

    if not (os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(encoder_path)):
        print("Model files not found. Please run 'python3 train.py' first.")
        return

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    label_encoder = joblib.load(encoder_path)

    # If passed as a DataFrame, use as-is; if array/list, wrap in DataFrame with scaler feature names
    if isinstance(feature_input, pd.DataFrame):
        feature_df = feature_input
    elif isinstance(feature_input, pd.Series):
        feature_df = pd.DataFrame([feature_input.values], columns=scaler.feature_names_in_)
    else:
        arr = np.array(feature_input).reshape(1, -1)
        feature_df = pd.DataFrame(arr, columns=scaler.feature_names_in_)

    scaled_features = scaler.transform(feature_df)
    pred_encoded = model.predict(scaled_features)
    predicted_genres = label_encoder.inverse_transform(pred_encoded)

    if len(predicted_genres) == 1:
        return predicted_genres[0]
    return predicted_genres

if __name__ == "__main__":
    feature_file = "features_small.csv"
    if os.path.exists(feature_file):
        df = pd.read_csv(feature_file)
        target_col = 'genre' if 'genre' in df.columns else df.columns[-1]
        
        drop_cols = [target_col]
        if 'track_id' in df.columns:
            drop_cols.append('track_id')
        if df.columns[0].startswith('Unnamed') or df.columns[0] == '0':
            drop_cols.append(df.columns[0])

    
        sample_count = min(100, len(df))
        sample_indices = df.sample(n=sample_count).index
        feature_df = df.drop(columns=drop_cols)
        sample_features = feature_df.loc[sample_indices]
        predictions = []

        for sample_number, (row_index, pred) in enumerate(
            zip(sample_indices, predict(sample_features)), start=1
        ):
            actual_genre = df.loc[row_index, target_col]
            predictions.append(pred)
            print(f"Sample {sample_number} Prediction: Predicted = '{pred}', Actual = '{actual_genre}'")

        actual_genres = df.loc[sample_indices, target_col].to_numpy()
        correct = sum(pred == actual for pred, actual in zip(predictions, actual_genres))
        print(f"\nPrediction summary ({sample_count} random songs)")
        print(f"Correct: {correct}")
        print(f"Wrong: {sample_count - correct}")

