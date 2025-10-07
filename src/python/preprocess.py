import os
import argparse
import numpy as np
from tqdm import tqdm
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import librosa
from utils.audio import load_list, load_audio, to_label_idx, CLASS_NAMES
from utils.io import ensure_dir, np_save_csv

def frame_signal(x, frame_length=16896, hop_length=512):
    frames = librosa.util.frame(x, frame_length=frame_length, hop_length=hop_length)
    return np.transpose(frames)

def logmel(frame, sr=16000, n_mels=30, n_fft=1024, hop_length=512):
    S = librosa.feature.melspectrogram(y=frame, sr=sr, n_mels=n_mels,
                                       n_fft=n_fft, hop_length=hop_length, center=False)
    S = S / (S.max() + 1e-9)
    return librosa.power_to_db(S, top_db=80.0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', help='JSON config path (overrides defaults)')
    ap.add_argument('--dataset_dir', required=True)
    ap.add_argument('--meta', default='TrainSet.txt')
    ap.add_argument('--out_dir', default='Output')
    args = ap.parse_args()

    import json
    cfg = None
    if args.config:
        with open(args.config, 'r', encoding='utf-8') as cf:
            cfg = json.load(cf)


    ensure_dir(args.out_dir)

    items = load_list(os.path.join(args.dataset_dir, args.meta))

    x_frames = []
    y_frames = []

    for rel_path, fine_label in tqdm(items, desc="Loading + framing"):
        y = load_audio(args.dataset_dir, rel_path, sr=16000, mono=True, duration=30)
        fl = cfg.get('framing',{}).get('frame_length',16896) if cfg else 16896
        hl = cfg.get('framing',{}).get('hop_length',512) if cfg else 512
        frames = frame_signal(y, frame_length=fl, hop_length=hl)
        lab_idx = to_label_idx(fine_label)
        x_frames.append(frames)
        y_frames.append(np.full(frames.shape[0], lab_idx))

    x_frames = np.asarray(x_frames)
    y_frames = np.asarray(y_frames)
    x_frames = x_frames.reshape(-1, x_frames.shape[-1])  # (N, 16896)
    y_frames = y_frames.reshape(-1)                      # (N,)

    # Extract Log-Mel features per frame
    feats = []
    for i in tqdm(range(len(x_frames)), desc="Log-Mel"):
        sr = cfg.get('audio',{}).get('sample_rate',16000) if cfg else 16000
        fm = cfg.get('features',{})
        n_mels = fm.get('n_mels',30) if cfg else 30
        n_fft = fm.get('n_fft',1024) if cfg else 1024
        hop = fm.get('hop_length',512) if cfg else 512
        feats.append(logmel(x_frames[i], sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop))
    feats = np.asarray(feats)  # (N, 30, 32)

    # Standardize (flatten → fit → transform)
    feats_r = feats.reshape(len(feats), 30 * 32)
    scaler = preprocessing.StandardScaler().fit(feats_r)
    feats_s = scaler.transform(feats_r)

    # Save scaler for reuse (noise eval / deployment reference)
    import numpy as _np
    _np.savez(os.path.join(args.out_dir, 'scaler.npz'), mean=scaler.mean_, scale=scaler.scale_)


    # One-hot labels
    num_classes = len(CLASS_NAMES)
    y_cat = np.eye(num_classes, dtype=np.float32)[y_frames]

    # Split: train/val/test = 0.75/0.25; then split train→train/val (0.75/0.25 of train)
    x_train, x_test, y_train, y_test = train_test_split(feats_s, y_cat, test_size=0.25, random_state=1)
    x_train, x_val,  y_train, y_val  = train_test_split(x_train, y_train, test_size=0.25, random_state=1)

    # Save CSVs
    np_save_csv(os.path.join(args.out_dir, 'x_train.csv'), x_train)
    np_save_csv(os.path.join(args.out_dir, 'y_train.csv'), y_train)
    np_save_csv(os.path.join(args.out_dir, 'x_val.csv'),   x_val)
    np_save_csv(os.path.join(args.out_dir, 'y_val.csv'),   y_val)
    np_save_csv(os.path.join(args.out_dir, 'x_test.csv'),  x_test)
    np_save_csv(os.path.join(args.out_dir, 'y_test.csv'),  y_test)

    print("Saved preprocessed CSVs into:", args.out_dir)

if __name__ == '__main__':
    main()
