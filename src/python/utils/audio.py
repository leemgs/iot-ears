import os
import numpy as np
import librosa

CLASS_MAP = {
    'bus': 2, 'car': 2, 'subway': 2, 'train': 2, 'tramway': 2,
    'cafeRestaurant': 0, 'home': 0, 'office': 0, 'shoppingCenter': 0,
    'cityCenter': 1, 'park': 1, 'residentialArea': 1,
}

CLASS_NAMES = ['indoor', 'outdoor', 'vehicle']

def load_list(meta_path):
    lines = []
    with open(meta_path, 'r', encoding='utf-8') as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith('#'):
                continue
            parts = ln.split()
            if len(parts) != 2:
                raise ValueError("Each line must be '<path> <label>'")
            lines.append((parts[0], parts[1]))
    return lines

def load_audio(dataset_dir, rel_path, sr=16000, mono=True, duration=30):
    path = os.path.join(dataset_dir, rel_path)
    y, _ = librosa.load(path, sr=sr, mono=mono, duration=duration, dtype=np.float32)
    return y

def to_label_idx(label):
    if label not in CLASS_MAP:
        raise KeyError(f"Unknown fine-grained label: {label}")
    return CLASS_MAP[label]
