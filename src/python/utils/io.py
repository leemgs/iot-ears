import os
import numpy as np
import json

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def np_save_csv(path, arr):
    np.savetxt(path, arr, delimiter=',')

def np_load_csv(path, dtype=float):
    return np.loadtxt(path, delimiter=',', dtype=dtype)

def save_json(path, obj):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
