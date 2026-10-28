"""ACE-Step go/no-go gate: is the Core AI fp16 trajectory as close to fp32 as the authors' own bf16 is? (CPU only.)

usage: gate_trajectory.py <references-dir> <coreai-dir> <out.json> [candidate]

candidate is the file suffix: `coreai` (default; run_trajectory_coreai.py, reference conditions) or `e2e`
(run_encoders_coreai.py, Core AI's own text and condition encoders).

For each prompt, both candidates (bf16 eager and Core AI) are measured against the fp32 eager trajectory from the same
conditions and noise:
  - latents: relative L2 and cosine of the state after every step and of the final latents;
  - audio: SNR (dB), a multi-resolution STFT distance (mean |log(|X|+ε) − log(|Y|+ε)| over 512/1024/2048-point
    Hann frames at 75% overlap) and the log-mel cosine (128 mel bands, 2048-point frames).
The gate passes when, on every prompt, Core AI is no further from fp32 than bf16 on the final-latent relative L2, the
STFT distance and the log-mel distance (1 − cosine). When decode_latents_coreai.py has decoded every trajectory with
one VAE (<id>-decoded.npz), the audio metrics use those decodes, so they judge the DiT alone; the fp16 VAE is then
measured on its own (vaeFp16OnFp32Latents) and the Core AI run's own audio is recorded (asRunAudio). Everything is recorded so a near miss can be judged; the WAVs are
there for listening.
"""
import json
import sys
from pathlib import Path

import numpy as np

SR, EPS = 48000, 1e-5


def rel_l2(x, ref):
    return float(np.linalg.norm((x - ref).ravel()) / np.linalg.norm(ref.ravel()))


def cosine(x, ref):
    x, ref = x.ravel().astype(np.float64), ref.ravel().astype(np.float64)
    return float(x @ ref / (np.linalg.norm(x) * np.linalg.norm(ref)))


def stft_mag(audio, n_fft):
    hop, window = n_fft // 4, np.hanning(n_fft)
    mono = audio.mean(0)
    frames = np.lib.stride_tricks.sliding_window_view(mono, n_fft)[::hop] * window
    return np.abs(np.fft.rfft(frames, axis=-1))


def mel_filterbank(n_fft, n_mels=128, fmin=0.0, fmax=SR / 2):
    def hz_to_mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def mel_to_hz(m):
        return 700.0 * (10 ** (m / 2595.0) - 1.0)
    edges = mel_to_hz(np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2))
    bins = np.fft.rfftfreq(n_fft, 1.0 / SR)
    lower, center, upper = edges[:-2, None], edges[1:-1, None], edges[2:, None]
    rising = (bins[None] - lower) / (center - lower)
    falling = (upper - bins[None]) / (upper - center)
    return np.maximum(0.0, np.minimum(rising, falling))


MEL = mel_filterbank(2048)


def audio_metrics(x, ref):
    n = min(x.shape[-1], ref.shape[-1])
    x, ref = x[..., :n], ref[..., :n]
    snr = float(10 * np.log10(np.sum(ref ** 2) / max(np.sum((x - ref) ** 2), 1e-20)))
    stft = float(np.mean([np.mean(np.abs(np.log(stft_mag(x, k) + EPS) - np.log(stft_mag(ref, k) + EPS)))
                          for k in (512, 1024, 2048)]))
    mel_x = np.log(stft_mag(x, 2048) @ MEL.T + EPS)
    mel_r = np.log(stft_mag(ref, 2048) @ MEL.T + EPS)
    return {"snrDB": snr, "stftDistance": stft, "logMelCosine": cosine(mel_x, mel_r)}


def candidate(xt, x0, audio, ref, ref_audio):
    per_step = [{"relL2": rel_l2(xt[i], ref["fp32_xt"][i]), "cosine": cosine(xt[i], ref["fp32_xt"][i])}
                for i in range(len(xt))]
    return {"steps": per_step, "finalRelL2": rel_l2(x0, ref["fp32_x0"]), "finalCosine": cosine(x0, ref["fp32_x0"]),
            **audio_metrics(audio, ref_audio)}


def main(references: Path, coreai: Path, out: Path, suffix: str = "coreai"):
    summary = {"criteria": "Core AI <= bf16 on final-latent relL2, STFT distance and log-mel (1 - cosine), every prompt",
               "candidate": suffix, "prompts": {}}
    passed = True
    for path in sorted(references.glob("*.npz")):
        if not (coreai / f"{path.stem}-{suffix}.npz").exists():
            print(path.stem, "skipped: no", suffix, "run")
            continue
        ref = np.load(path)
        if "bf16_x0" not in ref.files:
            print(path.stem, "skipped: no bf16 reference")
            continue
        ours = np.load(coreai / f"{path.stem}-{suffix}.npz")
        decoded_path = coreai / f"{path.stem}-decoded.npz"
        entry = {}
        if decoded_path.exists() and suffix in np.load(decoded_path).files:
            # One VAE for every trajectory (decode_latents_coreai.py): the audio metrics then judge the DiT alone. The
            # Core AI run's own audio (fp16 VAE) and the fp16 VAE on the fp32 latents are recorded beside it.
            decoded = np.load(decoded_path)
            ref_audio, bf16_audio, core_audio = decoded["fp32"], decoded["bf16"], decoded[suffix]
            entry["audioDecodedBy"] = "the Core AI fp32 VAE, for every trajectory"
            entry["vaeFp16OnFp32Latents"] = audio_metrics(decoded["fp32_vae16"], decoded["fp32"])
            entry["asRunAudio"] = audio_metrics(ours["coreai_audio"], ref["fp32_audio"])
        else:
            ref_audio, bf16_audio, core_audio = ref["fp32_audio"], ref["bf16_audio"], ours["coreai_audio"]
            entry["audioDecodedBy"] = "eager fp32 VAE for fp32 and bf16, the fp16 VAE export for Core AI"
        bf16 = candidate(ref["bf16_xt"], ref["bf16_x0"], bf16_audio, ref, ref_audio)
        core = candidate(ours["coreai_xt"], ours["coreai_x0"], core_audio, ref, ref_audio)
        checks = {"finalRelL2": core["finalRelL2"] <= bf16["finalRelL2"],
                  "stftDistance": core["stftDistance"] <= bf16["stftDistance"],
                  "logMel": (1 - core["logMelCosine"]) <= (1 - bf16["logMelCosine"])}
        ok = all(checks.values())
        passed &= ok
        summary["prompts"][path.stem] = {"bf16": bf16, "coreai": core, "checks": checks, "pass": ok, **entry}
        print(path.stem, "pass" if ok else "FAIL", json.dumps({k: {m: v[m] for m in
              ("finalRelL2", "snrDB", "stftDistance", "logMelCosine")} for k, v in (("bf16", bf16), ("coreai", core))}))
    summary["gate"] = "pass" if passed and summary["prompts"] else "fail"
    out.write_text(json.dumps(summary, indent=1))
    print("gate:", summary["gate"])
    return 0 if summary["gate"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), *(sys.argv[4:5])))
