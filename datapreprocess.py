import os
import pandas as pd

def extract_small_features(base_dir="fma-small/fma_metadata/fma_metadata", output_csv="features_small.csv"):
    tracks_path = os.path.join(base_dir, "tracks.csv")
    features_path = os.path.join(base_dir, "features.csv")

    if not os.path.exists(tracks_path) or not os.path.exists(features_path):
        raise FileNotFoundError(f"Metadata files not found in {base_dir}")

    print("Loading track metadata...")
    tracks = pd.read_csv(tracks_path, index_col=0, header=[0, 1])
    small_tracks = tracks[tracks[('set', 'subset')] == 'small']
    labels = small_tracks[('track', 'genre_top')].dropna()

    print("Loading pre-computed features (this may take a minute)...")
    features = pd.read_csv(features_path, index_col=0, header=[0, 1, 2])

    print("Filtering features for fma_small subset...")
    small_features = features.loc[labels.index].copy()

    # Flatten multi-level feature column headers
    small_features.columns = ['_'.join(map(str, col)).strip() for col in small_features.columns]

    # Assign genre target column
    small_features['genre'] = labels.values

    print(f"Saving extracted features to {output_csv}...")
    small_features.to_csv(output_csv)
    print(f"Successfully saved {small_features.shape[0]} samples with {small_features.shape[1]} columns to {output_csv}!")

if __name__ == '__main__':
    extract_small_features()