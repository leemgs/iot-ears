
import os
import argparse
import numpy as np
from tqdm import tqdm
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import librosa

from utils.audio import load_list, load_audio, to_label_idx, CLASS_NAMES
from utils.io import ensure_dir, np_save_csv
from utils.augment import add_noise_for_snr, time_shift, pitch_shift, time_stretch

def frame_signal(x, frame_length=16896, hop_length=512):
    frames = librosa.util.frame(x, frame_length=frame_length, hop_length=hop_length)
    return np.transpose(frames)

def logmel(frame, sr=16000, n_mels=30, n_fft=1024, hop_length=512):
    S = librosa.feature.melspectrogram(y=frame, sr=sr, n_mels=n_mels,
                                       n_fft=n_fft, hop_length=hop_length, center=False)
    S = S / (S.max() + 1e-9)
    return librosa.power_to_db(S, top_db=80.0)

def main():
    ap = argparse.ArgumentParser(description='Preprocess with waveform-level augmentation and SNR variants')
    ap.add_argument('--dataset_dir', required=True)
    ap.add_argument('--meta', default='TrainSet.txt')
    ap.add_argument('--out_dir', default='OutputAug')
    ap.add_argument('--snr_list', default='20,10,0', help='Comma-separated SNR dB values for noise augmentation')
    ap.add_argument('--pitch_steps', default='-1,1', help='Comma-separated semitone steps for pitch shift')
    ap.add_argument('--time_stretch', default='0.9,1.1', help='Comma-separated rates for time stretch (e.g., 0.9,1.1)')
    ap.add_argument('--shift_ratio', type=float, default=0.1, help='Max ratio for random time shift')
    ap.add_argument('--config', help='JSON config path (overrides defaults)')
    args = ap.parse_args()

    ensure_dir(args.out_dir)

    # Defaults
    sr = 16000
    frame_length = 16896
    hop = 512
    n_mels = 30
    n_fft = 1024

    # Load config if provided
    if args.config:
        import json
        with open(args.config, 'r', encoding='utf-8') as f:
            cfg = json.load(f)
        sr = cfg.get('audio',{}).get('sample_rate', sr)
        frame_length = cfg.get('framing',{}).get('frame_length', frame_length)
        hop = cfg.get('framing',{}).get('hop_length', hop)
        feats = cfg.get('features', {})
        n_mels = feats.get('n_mels', n_mels)
        n_fft = feats.get('n_fft', n_fft)

    items = load_list(os.path.join(args.dataset_dir, args.meta))

    snr_values = [float(x) for x in args.snr_list.split(',') if x.strip()]
    pitch_vals = [float(x) for x in args.pitch_steps.split(',') if x.strip()]
    stretch_vals = [float(x) for x in args.time_stretch.split(',') if x.strip()]

    X, Y = [], []
    rng = np.random.RandomState(1337)

    for rel_path, fine_label in tqdm(items, desc='Augmenting'):
        y = load_audio(args.dataset_dir, rel_path, sr=sr, mono=True, duration=30.0)
        variants = [y]

        variants.append(time_shift(y, shift_max_ratio=args.shift_ratio, rng=rng))
        for snr in snr_values:
            variants.append(add_noise_for_snr(y, target_snr_db=snr, rng=rng))
        for ps in pitch_vals:
            variants.append(pitch_shift(y, sr=sr, n_steps=ps))
        for st in stretch_vals:
            if abs(st - 1.0) > 1e-6:
                try:
                    variants.append(time_stretch(y, rate=st))
                except Exception:
                    pass

        lab_idx = to_label_idx(fine_label)
        for yv in variants:
            frames = frame_signal(yv, frame_length=frame_length, hop_length=hop)
            for i in range(frames.shape[0]):
                lm = logmel(frames[i], sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop)
                X.append(lm)
                Y.append(lab_idx)

    X = np.asarray(X)  # (N, 30, 32)
    Y = np.asarray(Y)
    num_classes = len(CLASS_NAMES)
    Y = np.eye(num_classes, dtype=np.float32)[Y]

    Xr = X.reshape(len(X), 30*32)
    scaler = preprocessing.StandardScaler().fit(Xr)
    Xs = scaler.transform(Xr)

    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(Xs, Y, test_size=0.25, random_state=42)
    x_train, x_val,  y_train, y_val  = train_test_split(x_train, y_train, test_size=0.25, random_state=42)

    np.savetxt(os.path.join(args.out_dir, 'x_train.csv'), x_train, delimiter=',')
    np.savetxt(os.path.join(args.out_dir, 'y_train.csv'), y_train, delimiter=',')
    np.savetxt(os.path.join(args.out_dir, 'x_val.csv'),   x_val,   delimiter=',')
    np.savetxt(os.path.join(args.out_dir, 'y_val.csv'),   y_val,   delimiter=',')
    np.savetxt(os.path.join(args.out_dir, 'x_test.csv'),  x_test,  delimiter=',')

    # Save scaler
    np.savez(os.path.join(args.out_dir, 'scaler.npz'),
             mean=scaler.mean_, scale=scaler.scale_)

    print('Saved augmented CSVs + scaler into:', args.out_dir)

if __name__ == '__main__':
    main()
