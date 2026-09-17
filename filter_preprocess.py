import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import librosa
import joblib
from scipy.signal import butter, lfilter
import multiprocessing as mp

# Suppress librosa and soundfile warnings
warnings.filterwarnings('ignore')

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def apply_bandpass_filter(data, lowcut=20.0, highcut=20000.0, fs=22050, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return lfilter(b, a, data)

def extract_features_from_file(args):
    track_id, file_path, genre = args
    if not os.path.exists(file_path):
        return None
    try:
        # Load audio (mono, sr=22050, up to 30 seconds)
        y, sr = librosa.load(file_path, sr=22050, mono=True, duration=30)
        if len(y) == 0:
            return None
        
        # 1. Bandpass filter
        y_filt = apply_bandpass_filter(y, lowcut=20.0, highcut=20000.0, fs=sr)  # Bandpass filter
        
        # 2. Peak normalization
        y_norm = librosa.util.normalize(y_filt)  # Normalize volume
        
        # 3. Pre-emphasis filter
        y_pre = librosa.effects.preemphasis(y_norm, coef=0.97)  # Boost highs
        
        # 4. HPSS (Harmonic-Percussive Source Separation)
        y_harmonic, y_percussive = librosa.effects.hpss(y_pre)  # Separate stems
        
        features = {'track_id': track_id, 'genre': genre}
        
        # 5. Extract features
        # Tempo
        tempo, _ = librosa.beat.beat_track(y=y_percussive, sr=sr)
        features['tempo'] = float(tempo[0]) if isinstance(tempo, (list, np.ndarray)) else float(tempo)
        
        # MFCC Harmonic (13 coefficients)
        mfcc_harm = librosa.feature.mfcc(y=y_harmonic, sr=sr, n_mfcc=13)
        for i in range(13):
            features[f'mfcc_harm_mean_{i}'] = float(np.mean(mfcc_harm[i]))
            features[f'mfcc_harm_std_{i}'] = float(np.std(mfcc_harm[i]))
            
        # MFCC Percussive (13 coefficients)
        mfcc_perc = librosa.feature.mfcc(y=y_percussive, sr=sr, n_mfcc=13)
        for i in range(13):
            features[f'mfcc_perc_mean_{i}'] = float(np.mean(mfcc_perc[i]))
            features[f'mfcc_perc_std_{i}'] = float(np.std(mfcc_perc[i]))

        # MFCC Harmonic Deltas & Delta-Deltas
        mfcc_harm_delta = librosa.feature.delta(mfcc_harm)
        mfcc_harm_delta2 = librosa.feature.delta(mfcc_harm, order=2)
        for i in range(13):
            features[f'mfcc_harm_delta_mean_{i}'] = float(np.mean(mfcc_harm_delta[i]))
            features[f'mfcc_harm_delta_std_{i}'] = float(np.std(mfcc_harm_delta[i]))
            features[f'mfcc_harm_delta2_mean_{i}'] = float(np.mean(mfcc_harm_delta2[i]))
            features[f'mfcc_harm_delta2_std_{i}'] = float(np.std(mfcc_harm_delta2[i]))
            
        # Chroma CENS (pitch profile from harmonic component)
        chroma = librosa.feature.chroma_cens(y=y_harmonic, sr=sr)
        for i in range(12):
            features[f'chroma_mean_{i}'] = float(np.mean(chroma[i]))
            features[f'chroma_std_{i}'] = float(np.std(chroma[i]))
            
        # Spectral Centroid (brightness)
        centroid = librosa.feature.spectral_centroid(y=y_pre, sr=sr)
        features['centroid_mean'] = float(np.mean(centroid))
        features['centroid_std'] = float(np.std(centroid))
        
        # Spectral Bandwidth
        bandwidth = librosa.feature.spectral_bandwidth(y=y_pre, sr=sr)
        features['bandwidth_mean'] = float(np.mean(bandwidth))
        features['bandwidth_std'] = float(np.std(bandwidth))
        
        # Spectral Contrast
        contrast = librosa.feature.spectral_contrast(y=y_pre, sr=sr)
        for i in range(contrast.shape[0]):
            features[f'contrast_mean_{i}'] = float(np.mean(contrast[i]))
            features[f'contrast_std_{i}'] = float(np.std(contrast[i]))
            
        # Spectral Rolloff
        rolloff = librosa.feature.spectral_rolloff(y=y_pre, sr=sr)
        features['rolloff_mean'] = float(np.mean(rolloff))
        features['rolloff_std'] = float(np.std(rolloff))
        
        # Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(y_pre)
        features['zcr_mean'] = float(np.mean(zcr))
        features['zcr_std'] = float(np.std(zcr))
        
        # RMS Energy
        rms = librosa.feature.rms(y=y_pre)
        features['rms_mean'] = float(np.mean(rms))
        features['rms_std'] = float(np.std(rms))
        
        return features
    except Exception as e:
        return None

def main():
    metadata_dir = "fma-small/fma_metadata/fma_metadata"
    audio_dir = "fma-small/fma_small/fma_small"
    output_csv = "filtered_features.csv"
    
    tracks_path = os.path.join(metadata_dir, "tracks.csv")
    if not os.path.exists(tracks_path):
        print(f"Error: Could not find track metadata at {tracks_path}")
        return
        
    print("Loading tracks metadata...")
    tracks = pd.read_csv(tracks_path, index_col=0, header=[0, 1])
    small_tracks = tracks[tracks[('set', 'subset')] == 'small']
    labels = small_tracks[('track', 'genre_top')].dropna()
    
    print(f"Found {len(labels)} tracks in the fma_small subset.")
    
    tasks = []
    for track_id, genre in labels.items():
        tid_str = '{:06d}'.format(track_id)
        file_path = os.path.join(audio_dir, tid_str[:3], tid_str + '.mp3')
        if os.path.exists(file_path):
            tasks.append((track_id, file_path, genre))
            
    print(f"Verified {len(tasks)} audio files exist on disk.")
    
    # Process tasks in parallel
    num_workers = max(1, mp.cpu_count() - 1)
    print(f"Starting filtered feature extraction with {num_workers} parallel workers...")
    
    results = []
    pool = mp.Pool(processes=num_workers)
    
    count = 0
    total = len(tasks)
    
    for result in pool.imap_unordered(extract_features_from_file, tasks, chunksize=4):
        if result is not None:
            results.append(result)
        count += 1
        if count % 100 == 0 or count == total:
            print(f"Processed {count}/{total} files ({(count/total)*100:.1f}%) - Extracted {len(results)} valid samples...", flush=True)
            
    pool.close()
    pool.join()
    
    print("\nExtraction complete! Creating DataFrame...")
    df = pd.DataFrame(results)
    
    # Save to CSV
    df.to_csv(output_csv, index=False)
    print(f"Saved {df.shape[0]} samples with {df.shape[1]} features to {output_csv}!")

if __name__ == '__main__':
    main()
