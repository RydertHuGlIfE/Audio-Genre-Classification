import os
import warnings
import numpy as np
import pandas as pd
import librosa
import multiprocessing as mp
from scipy.signal import butter, sosfilt

warnings.filterwarnings('ignore')


# ==============================================================================
# 1. Signal Filters
# ==============================================================================
def apply_highpass(y, sr, cutoff=1500, order=5):
    """Remove everything below 'cutoff' Hz — isolates treble / transients."""
    sos = butter(order, cutoff / (sr / 2), btype='high', output='sos')
    return sosfilt(sos, y)


def apply_lowpass(y, sr, cutoff=500, order=5):
    """Remove everything above 'cutoff' Hz — isolates bass / sub-bass."""
    sos = butter(order, cutoff / (sr / 2), btype='low', output='sos')
    return sosfilt(sos, y)


def apply_bandpass(y, sr, lowcut=300, highcut=3000, order=5):
    """Retain mid-range frequencies — vocal and instrument body."""
    sos = butter(order, [lowcut / (sr / 2), highcut / (sr / 2)], btype='band', output='sos')
    return sosfilt(sos, y)


# ==============================================================================
# 2. Feature Extraction Worker (called per audio file)
# ==============================================================================
def extract_features_from_file(args):
    track_id, file_path, genre = args
    if not os.path.exists(file_path):
        return None
    try:
        # Load raw audio (mono, 22050 Hz, 30 seconds)
        y, sr = librosa.load(file_path, sr=22050, mono=True, duration=30)
        if len(y) == 0:
            return None

        # Peak-normalize base signal
        y = librosa.util.normalize(y)

        # Frequency band decomposition
        y_low  = apply_lowpass(y, sr, cutoff=500)        # Bass (< 500 Hz)
        y_mid  = apply_bandpass(y, sr, 300, 3000)        # Mids (300–3000 Hz)
        y_high = apply_highpass(y, sr, cutoff=1500)      # Treble (> 1500 Hz)

        # HPSS: separate harmonic and percussive stems for richer feature sets
        y_pre = librosa.effects.preemphasis(y, coef=0.97)
        y_harmonic, y_percussive = librosa.effects.hpss(y_pre)

        features = {'track_id': track_id, 'genre': genre}


        # === TEMPO & BEAT FEATURES ===
        tempo, beat_frames = librosa.beat.beat_track(y=y_percussive, sr=sr)
        features['tempo'] = float(tempo[0]) if isinstance(tempo, (list, np.ndarray)) else float(tempo)

        onset_env = librosa.onset.onset_strength(y=y_percussive, sr=sr)
        if len(beat_frames) > 0:
            features['beat_strength_mean'] = float(onset_env[beat_frames].mean())
            features['beat_strength_std']  = float(onset_env[beat_frames].std())
        else:
            features['beat_strength_mean'] = 0.0
            features['beat_strength_std']  = 0.0

        # Beat interval regularity (lower std = more rhythmically stable)
        if len(beat_frames) >= 2:
            beat_times = librosa.frames_to_time(beat_frames, sr=sr)
            intervals = np.diff(beat_times)
            features['beat_interval_mean'] = float(intervals.mean())
            features['beat_interval_std']  = float(intervals.std())
        else:
            features['beat_interval_mean'] = 0.0
            features['beat_interval_std']  = 0.0


        # === PITCH & PITCH VARIABILITY (YIN algorithm) ===
        f0 = librosa.yin(y_harmonic, fmin=50, fmax=2000, sr=sr)
        voiced = f0[f0 > 0]
        features['pitch_mean']         = float(voiced.mean()) if len(voiced) > 0 else 0.0
        features['pitch_std']          = float(voiced.std())  if len(voiced) > 0 else 0.0
        features['pitch_var']          = float(voiced.var())  if len(voiced) > 0 else 0.0
        features['voiced_fraction']    = float(len(voiced) / len(f0)) if len(f0) > 0 else 0.0
        features['pitch_range']        = float(voiced.max() - voiced.min()) if len(voiced) > 0 else 0.0


        # === HIGH-PASS FEATURES (Treble > 1500 Hz) ===
        mfcc_hp = librosa.feature.mfcc(y=y_high, sr=sr, n_mfcc=13)
        for i in range(13):
            features[f'mfcc_hp_mean_{i}'] = float(np.mean(mfcc_hp[i]))
            features[f'mfcc_hp_std_{i}']  = float(np.std(mfcc_hp[i]))

        zcr_hp = librosa.feature.zero_crossing_rate(y_high)
        features['zcr_hp_mean'] = float(np.mean(zcr_hp))
        features['zcr_hp_std']  = float(np.std(zcr_hp))

        centroid_hp = librosa.feature.spectral_centroid(y=y_high, sr=sr)
        features['centroid_hp_mean'] = float(np.mean(centroid_hp))
        features['centroid_hp_std']  = float(np.std(centroid_hp))


        # === LOW-PASS FEATURES (Bass < 500 Hz) ===
        rms_lp = librosa.feature.rms(y=y_low)
        features['rms_lp_mean'] = float(np.mean(rms_lp))
        features['rms_lp_std']  = float(np.std(rms_lp))

        mfcc_lp = librosa.feature.mfcc(y=y_low, sr=sr, n_mfcc=6)
        for i in range(6):
            features[f'mfcc_lp_mean_{i}'] = float(np.mean(mfcc_lp[i]))
            features[f'mfcc_lp_std_{i}']  = float(np.std(mfcc_lp[i]))

        bandwidth_lp = librosa.feature.spectral_bandwidth(y=y_low, sr=sr)
        features['bandwidth_lp_mean'] = float(np.mean(bandwidth_lp))
        features['bandwidth_lp_std']  = float(np.std(bandwidth_lp))


        # === MID-BAND FEATURES (300–3000 Hz: vocals + instruments) ===
        mfcc_mid = librosa.feature.mfcc(y=y_mid, sr=sr, n_mfcc=13)
        for i in range(13):
            features[f'mfcc_mid_mean_{i}'] = float(np.mean(mfcc_mid[i]))
            features[f'mfcc_mid_std_{i}']  = float(np.std(mfcc_mid[i]))


        # === HARMONIC STEM FEATURES ===
        mfcc_harm = librosa.feature.mfcc(y=y_harmonic, sr=sr, n_mfcc=20)
        for i in range(20):
            features[f'mfcc_harm_mean_{i}'] = float(np.mean(mfcc_harm[i]))
            features[f'mfcc_harm_std_{i}']  = float(np.std(mfcc_harm[i]))

        mfcc_harm_delta  = librosa.feature.delta(mfcc_harm)
        mfcc_harm_delta2 = librosa.feature.delta(mfcc_harm, order=2)
        for i in range(20):
            features[f'mfcc_harm_delta_mean_{i}']  = float(np.mean(mfcc_harm_delta[i]))
            features[f'mfcc_harm_delta_std_{i}']   = float(np.std(mfcc_harm_delta[i]))
            features[f'mfcc_harm_delta2_mean_{i}'] = float(np.mean(mfcc_harm_delta2[i]))
            features[f'mfcc_harm_delta2_std_{i}']  = float(np.std(mfcc_harm_delta2[i]))

        chroma = librosa.feature.chroma_cens(y=y_harmonic, sr=sr)
        for i in range(12):
            features[f'chroma_mean_{i}'] = float(np.mean(chroma[i]))
            features[f'chroma_std_{i}']  = float(np.std(chroma[i]))

        tonnetz = librosa.feature.tonnetz(y=y_harmonic, sr=sr)
        for i in range(tonnetz.shape[0]):
            features[f'tonnetz_mean_{i}'] = float(np.mean(tonnetz[i]))
            features[f'tonnetz_std_{i}']  = float(np.std(tonnetz[i]))


        # === PERCUSSIVE STEM FEATURES ===
        mfcc_perc = librosa.feature.mfcc(y=y_percussive, sr=sr, n_mfcc=13)
        for i in range(13):
            features[f'mfcc_perc_mean_{i}'] = float(np.mean(mfcc_perc[i]))
            features[f'mfcc_perc_std_{i}']  = float(np.std(mfcc_perc[i]))


        # === FULL-SPECTRUM SPECTRAL FEATURES ===
        centroid   = librosa.feature.spectral_centroid(y=y_pre, sr=sr)
        bandwidth  = librosa.feature.spectral_bandwidth(y=y_pre, sr=sr)
        contrast   = librosa.feature.spectral_contrast(y=y_pre, sr=sr)
        rolloff    = librosa.feature.spectral_rolloff(y=y_pre, sr=sr)
        zcr        = librosa.feature.zero_crossing_rate(y_pre)
        rms        = librosa.feature.rms(y=y_pre)

        features['centroid_mean']  = float(np.mean(centroid))
        features['centroid_std']   = float(np.std(centroid))
        features['bandwidth_mean'] = float(np.mean(bandwidth))
        features['bandwidth_std']  = float(np.std(bandwidth))
        features['rolloff_mean']   = float(np.mean(rolloff))
        features['rolloff_std']    = float(np.std(rolloff))
        features['zcr_mean']       = float(np.mean(zcr))
        features['zcr_std']        = float(np.std(zcr))
        features['rms_mean']       = float(np.mean(rms))
        features['rms_std']        = float(np.std(rms))

        for i in range(contrast.shape[0]):
            features[f'contrast_mean_{i}'] = float(np.mean(contrast[i]))
            features[f'contrast_std_{i}']  = float(np.std(contrast[i]))

        return features

    except Exception:
        return None


# ==============================================================================
# 3. Parallel Extraction Entry Point
# ==============================================================================
def main():
    metadata_dir = "fma-small/fma_metadata/fma_metadata"
    audio_dir    = "fma-small/fma_small/fma_small"
    output_csv   = "filtered_features.csv"

    tracks_path = os.path.join(metadata_dir, "tracks.csv")
    if not os.path.exists(tracks_path):
        print(f"Error: Track metadata not found at '{tracks_path}'")
        return

    print("Loading track metadata...")
    tracks = pd.read_csv(tracks_path, index_col=0, header=[0, 1])
    small_tracks = tracks[tracks[('set', 'subset')] == 'small']
    labels = small_tracks[('track', 'genre_top')].dropna()
    print(f"Found {len(labels)} tracks in fma_small subset.")

    tasks = []
    for track_id, genre in labels.items():
        tid_str   = '{:06d}'.format(track_id)
        file_path = os.path.join(audio_dir, tid_str[:3], tid_str + '.mp3')
        if os.path.exists(file_path):
            tasks.append((track_id, file_path, genre))

    print(f"Verified {len(tasks)} audio files on disk.")

    num_workers = max(1, mp.cpu_count() - 2)
    print(f"\nStarting extraction with {num_workers} parallel workers...")
    print("Features extracted per track:")
    print("  - Tempo, beat strength, beat interval regularity")
    print("  - Pitch mean/std/var/range + voiced fraction (YIN)")
    print("  - High-pass features: MFCC (13), ZCR, Centroid")
    print("  - Low-pass features: RMS, MFCC (6), Bandwidth")
    print("  - Mid-band features: MFCC (13)")
    print("  - Harmonic MFCC (20) + deltas + delta-deltas")
    print("  - Chroma CENS (12), Tonnetz (6)")
    print("  - Percussive MFCC (13)")
    print("  - Full-spectrum: Centroid, Bandwidth, Contrast, Rolloff, ZCR, RMS\n")

    results = []
    pool    = mp.Pool(processes=num_workers)
    total   = len(tasks)
    count   = 0

    for result in pool.imap_unordered(extract_features_from_file, tasks, chunksize=4):
        if result is not None:
            results.append(result)
        count += 1
        if count % 100 == 0 or count == total:
            pct = (count / total) * 100
            print(f"Progress: {count}/{total} ({pct:.1f}%) — {len(results)} valid samples extracted", flush=True)

    pool.close()
    pool.join()

    print("\nExtraction complete! Saving dataset...")
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"Saved {df.shape[0]} samples with {df.shape[1]-2} features to '{output_csv}'")
    print("(Run 'python train.py' — it will auto-load filtered_features.csv)")


if __name__ == '__main__':
    main()
