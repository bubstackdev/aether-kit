"""Waveform and log-mel comparison metrics, mirrored by AetherNativeTests' KokoroMetrics.swift."""
import math

import numpy as np

SAMPLE_RATE = 24_000
N_FFT = 1024
HOP = 256
MELS = 80
F_MAX = 12_000.0
FLOOR = 1e-5

DEFINITION = {
    "waveform": "rel_l2 = ||x - r|| / ||r||; max_abs; snr_db = 20 log10(||r|| / ||x - r||)",
    "log_mel": (f"|STFT| n_fft={N_FFT} hop={HOP} periodic Hann, no centering; {MELS} HTK-mel triangles "
                f"0-{int(F_MAX)} Hz, unnormalized; ln(max(mel, {FLOOR})); mean and max over bins of |x - r|"),
}


def _mel(hz):
    return 2595.0 * math.log10(1.0 + hz / 700.0)


def mel_filterbank():
    bins = N_FFT // 2 + 1
    freqs = np.arange(bins, dtype=np.float64) * SAMPLE_RATE / N_FFT
    points = [_mel(0) + (_mel(F_MAX) - _mel(0)) * i / (MELS + 1) for i in range(MELS + 2)]
    hz = [700.0 * (10 ** (m / 2595.0) - 1.0) for m in points]
    bank = np.zeros((MELS, bins), dtype=np.float64)
    for m in range(MELS):
        lo, center, hi = hz[m], hz[m + 1], hz[m + 2]
        up = (freqs - lo) / (center - lo)
        down = (hi - freqs) / (hi - center)
        bank[m] = np.maximum(0.0, np.minimum(up, down))
    return bank


def log_mel(x):
    x = np.asarray(x, dtype=np.float64)
    frames = 1 + (x.size - N_FFT) // HOP
    window = 0.5 - 0.5 * np.cos(2 * math.pi * np.arange(N_FFT) / N_FFT)
    stack = np.stack([x[i * HOP:i * HOP + N_FFT] * window for i in range(frames)])
    magnitude = np.abs(np.fft.rfft(stack, axis=-1))
    return np.log(np.maximum(magnitude @ mel_filterbank().T, FLOOR))


def compare(x, r):
    x = np.asarray(x, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    err = x - r
    rel = float(np.linalg.norm(err) / np.linalg.norm(r))
    diff = np.abs(log_mel(x) - log_mel(r))
    return {"rel_l2": rel, "max_abs": float(np.abs(err).max()),
            "snr_db": float(20 * math.log10(1 / rel)) if rel > 0 else float("inf"),
            "log_mel_mean_abs": float(diff.mean()), "log_mel_max_abs": float(diff.max())}
