
import numpy as np
import librosa

def add_noise_for_snr(x, target_snr_db=20.0, rng=None):
    """Add white noise to achieve target SNR (dB)."""
    if rng is None:
        rng = np.random.RandomState(1337)
    sig_power = np.mean(x**2) + 1e-12
    snr_linear = 10 ** (target_snr_db / 10.0)
    noise_power = sig_power / snr_linear
    noise = rng.randn(*x.shape) * np.sqrt(noise_power)
    return x + noise

def time_shift(x, shift_max_ratio=0.2, rng=None):
    if rng is None:
        rng = np.random.RandomState(1337)
    n = len(x)
    max_shift = int(n * shift_max_ratio)
    k = rng.randint(-max_shift, max_shift + 1)
    return np.roll(x, k)

def pitch_shift(x, sr, n_steps=1.0):
    return librosa.effects.pitch_shift(x, sr=sr, n_steps=n_steps)

def time_stretch(x, rate=1.0):
    if rate <= 0:
        return x
    return librosa.effects.time_stretch(x, rate=rate)
