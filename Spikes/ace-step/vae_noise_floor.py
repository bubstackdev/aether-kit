"""Where the fp16 VAE's error sits (CPU only): its residual against the fp32 VAE on the same fp32 latents.

usage: vae_noise_floor.py <coreai-dir> <out.json>
For each <id>-decoded.npz from decode_latents_coreai.py: the signal's and the residual's RMS in dBFS, the residual's
peak, and, over 2048-point Hann frames at hop 512 of the mono mix, the share of time-frequency bins where the residual's
magnitude exceeds the fp32 signal's (the near-silent bins that the gate's log-spectral metrics weigh as heavily as
loud ones), with the share of bins whose fp32 magnitude is below the metrics' 1e-5 floor.
"""
import json
import sys
from pathlib import Path

import numpy as np

coreai, out = Path(sys.argv[1]), Path(sys.argv[2])


def dbfs(x):
    return float(20 * np.log10(max(x, 1e-12)))


def magnitudes(mono, n=2048, hop=512):
    return np.abs(np.fft.rfft(np.lib.stride_tricks.sliding_window_view(mono, n)[::hop] * np.hanning(n), axis=-1))


report = {}
for path in sorted(coreai.glob("*-decoded.npz")):
    decoded = np.load(path)
    fp32, fp16 = decoded["fp32"].astype(np.float64), decoded["fp32_vae16"].astype(np.float64)
    residual = fp16 - fp32
    signal_bins, residual_bins = magnitudes(fp32.mean(0)), magnitudes(residual.mean(0))
    report[path.name[:-len("-decoded.npz")]] = {
        "signalRMSdBFS": dbfs(np.sqrt((fp32 ** 2).mean())), "residualRMSdBFS": dbfs(np.sqrt((residual ** 2).mean())),
        "residualPeakdBFS": dbfs(np.abs(residual).max()),
        "binsResidualAboveSignal": float((residual_bins > signal_bins).mean()),
        "binsSignalBelowMetricFloor": float((signal_bins < 1e-5).mean())}
    print(path.name, json.dumps(report[path.name[:-len("-decoded.npz")]]))
out.write_text(json.dumps(report, indent=1) + "\n")
