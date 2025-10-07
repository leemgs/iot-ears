
import os
import argparse
import numpy as np
import json
import librosa

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
from keras.models import load_model
from sklearn.metrics import accuracy_score, confusion_matrix

from utils.audio import load_list, load_audio, to_label_idx
from utils.augment import add_noise_for_snr

def logmel(frame, sr=16000, n_mels=30, n_fft=1024, hop_length=512):
    S = librosa.feature.melspectrogram(y=frame, sr=sr, n_mels=n_mels,
                                       n_fft=n_fft, hop_length=hop_length, center=False)
    S = S / (S.max() + 1e-9)
    return librosa.power_to_db(S, top_db=80.0)

def frame_signal(x, frame_length=16896, hop_length=512):
    frames = librosa.util.frame(x, frame_length=frame_length, hop_length=hop_length)
    return np.transpose(frames)

def main():
    ap = argparse.ArgumentParser(description='Evaluate trained model under multiple SNR levels (dB)')
    ap.add_argument('--dataset_dir', required=True)
    ap.add_argument('--meta', default='TrainSet.txt')
    ap.add_argument('--model', default='Output/model.h5')
    ap.add_argument('--scaler', default='Output/scaler.npz')
    ap.add_argument('--snr_list', default='clean,20,10,0', help='Comma-separated: include "clean" for no noise')
    ap.add_argument('--out_dir', default='OutputNoise')
    ap.add_argument('--config', help='JSON config path for feature/framings')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Defaults
    sr = 16000
    frame_length = 16896
    hop = 512
    n_mels = 30
    n_fft = 1024

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

    # Load model and scaler
    model = load_model(args.model, compile=False)
    sc = np.load(args.scaler)
    mean, scale = sc['mean'], sc['scale']

    items = load_list(os.path.join(args.dataset_dir, args.meta))
    snr_tokens = [t.strip() for t in args.snr_list.split(',') if t.strip()]

    results = {}
    for snr_tok in snr_tokens:
        ys_true, ys_pred = [], []
        for rel_path, fine_label in items:
            y = load_audio(args.dataset_dir, rel_path, sr=sr, mono=True, duration=30.0)
            if snr_tok != 'clean':
                y = add_noise_for_snr(y, target_snr_db=float(snr_tok))
            frames = frame_signal(y, frame_length=frame_length, hop_length=hop)

            # Build features for all frames
            X = [logmel(fr, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop) for fr in frames]
            X = np.asarray(X).reshape(-1, 30*32)
            Xs = (X - mean) / (scale + 1e-12)
            Xs = Xs.reshape(-1, 30, 32, 1)

            logits = model.predict(Xs, verbose=0)
            ys_pred.extend(np.argmax(logits, axis=1).tolist())

            lab_idx = None
            try:
                from utils.audio import CLASS_MAP
                lab_idx = CLASS_MAP[fine_label]
            except Exception:
                pass
            if lab_idx is None:
                # fallback (should not happen if label is valid)
                lab_idx = 0
            ys_true.extend([lab_idx]*len(Xs))

        acc = float(accuracy_score(ys_true, ys_pred))
        cm = confusion_matrix(ys_true, ys_pred, labels=[0,1,2]).tolist()
        results[snr_tok] = {"accuracy": acc, "confusion_matrix": cm}

    with open(os.path.join(args.out_dir, 'snr_results.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    import csv
    with open(os.path.join(args.out_dir, 'snr_results.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['SNR(dB)', 'Accuracy'])
        for k, v in results.items():
            w.writerow([k, v['accuracy']])

    print('Saved:', os.path.join(args.out_dir, 'snr_results.json'))
    print('Saved:', os.path.join(args.out_dir, 'snr_results.csv'))

if __name__ == '__main__':
    main()
