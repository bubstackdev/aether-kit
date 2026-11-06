#!/usr/bin/env python3
"""The Kokoro gate: the exported Core AI graphs, driven by the host recipe, against the upstream CPU references.

    Forge/.venv/bin/python -I Forge/kokoro/check_kokoro.py --bundle <bundle dir> --reference <reference dir> \
        --report <new check-report.json> [--probe-compute]

For every reference case (reference_kokoro.py), with the bundle's own vocabulary, voice tables and compute placement:
- token ids and integer frames per token equal upstream exactly;
- the bucketed decoder (inputs padded to the smallest bucket that fits, seeded noise extended to the bucket, the
  valid frame count passed in, the padded tail dropped) gives waveform relative L2 <= min(0.05, 4x the case's own
  float32-versus-float64 upstream difference) and log-mel mean absolute difference <= 0.03 nats;
- a repeat synthesis is bit-identical.
These are 0.0.x's bars, stated before its first native run. `--probe-compute` also reruns, in a subprocess (an abort
must not take the gate down), the durations graph on the GPU, which 0.0.x saw abort (see export_kokoro.COMPUTE);
`--compute ROLE=cpu|gpu` runs the whole gate with another placement. Writes each case's audio as a WAV beside the
report. Exits nonzero when a bar fails.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
import time
import wave
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import speech_metrics  # noqa: E402
from kokoro_graphs import HARMONICS, SAMPLE_RATE, SAMPLES_PER_FRAME, splitmix_normal  # noqa: E402

MAX_REL_L2, FLOOR_MULTIPLE, MAX_LOG_MEL = 0.05, 4.0, 0.03


class Graph:
    def __init__(self, path: Path, function: str, compute: str) -> None:
        import coreai.runtime as rt
        self.rt = rt
        options = (rt.SpecializationOptions.cpu_only() if compute == "cpu" else
                   rt.SpecializationOptions.from_preferred_compute_unit_kind(rt.ComputeUnitKind.gpu()))
        self.model = asyncio.run(rt.AIModel.load(path, options))
        self.function = self.model.load_function(function)

    def __call__(self, **feeds: np.ndarray) -> dict[str, np.ndarray]:
        rt = self.rt

        async def go():
            out = await self.function({k: rt.NDArray(np.ascontiguousarray(v)) for k, v in feeds.items()}, {})
            return {k: v.numpy().copy() for k, v in out.items()}
        return asyncio.run(go())


def tokenize(vocabulary: dict, phonemes: str, maximum: int) -> list[int]:
    """The host rule (MANIFEST.md §5.8): Unicode scalars, no normalization, unknown symbols refused."""
    ids = []
    for ch in phonemes:
        if ch not in vocabulary:
            raise ValueError(f"phoneme {ch!r} (U+{ord(ch):04X}) is not in the vocabulary")
        ids.append(vocabulary[ch])
    if not 1 <= len(ids) <= maximum:
        raise ValueError(f"phoneme count must be 1...{maximum}")
    return [0, *ids, 0]


def frames_for(sums: np.ndarray, speed: float) -> np.ndarray:
    """max(1, round half to even (sum / speed)), in float32 as the graph's output is (torch.round's rule)."""
    return np.maximum(np.round(sums.astype(np.float32) / np.float32(speed)), 1).astype(np.int64)


def synthesize(durations: Graph, decoder: Graph, contract: dict, ids: list[int], style: np.ndarray, speed: float,
               seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, dict]:
    t0 = time.monotonic()
    out = durations(input_ids=np.array([ids], np.int32), style=style)
    t1 = time.monotonic()
    sums, d, t_en = out["duration"].reshape(-1), out["prosody"], out["text"]
    frames = frames_for(sums, speed)
    index = np.repeat(np.arange(frames.size), frames)
    count = int(index.size)
    bucket = next(b for b in contract["frames"]["buckets"] if b >= count)
    en = np.zeros((1, d.shape[-1], bucket), np.float32)
    asr = np.zeros((1, t_en.shape[1], bucket), np.float32)
    en[0, :, :count] = d[0, index, :].T
    asr[0, :, :count] = t_en[0][:, index]
    noise = splitmix_normal(seed, bucket * SAMPLES_PER_FRAME * HARMONICS).reshape(1, bucket, SAMPLES_PER_FRAME, HARMONICS)
    t2 = time.monotonic()
    audio = decoder(prosody=en, text=asr, style=style, noise=noise, frames=np.array([count], np.int32))["audio"]
    t3 = time.monotonic()
    return (audio.reshape(-1)[:count * SAMPLES_PER_FRAME], sums, frames, bucket,
            {"durationsSeconds": round(t1 - t0, 3), "decodeSeconds": round(t3 - t2, 3)})


def write_wav(path: Path, audio: np.ndarray) -> None:
    pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm.tobytes())


PROBE = """
import sys, asyncio, numpy as np
sys.path.insert(0, {here!r})
from check_kokoro import Graph
g = Graph({path!r}, {function!r}, "gpu")
feeds = {feeds}
out = g(**{{k: np.load(v) for k, v in feeds.items()}})
np.save({result!r}, out[{output!r}])
"""


def probe_gpu(bundle: Path, manifest: dict, role: str, feeds: dict[str, np.ndarray], output: str,
              scratch: Path) -> dict:
    """Runs one graph on the GPU in a fresh interpreter (an abort must not take the gate down)."""
    scratch.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, value in feeds.items():
        paths[name] = str(scratch / f"{role}-{name}.npy")
        np.save(paths[name], value)
    result = scratch / f"{role}-result.npy"
    graph = manifest["graphs"][role]
    code = PROBE.format(here=str(HERE), path=str(bundle / graph["file"].split(":", 1)[1]), function=graph["function"],
                        feeds=repr(paths), result=str(result), output=output)
    run = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, timeout=600)
    record = {"returncode": run.returncode, "stderrTail": run.stderr.strip().splitlines()[-3:]}
    if run.returncode == 0 and result.exists():
        record["output"] = np.load(result)
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--probe-compute", action="store_true")
    parser.add_argument("--compute", action="append", default=[], metavar="ROLE=cpu|gpu",
                        help="override a graph's manifest placement (a measurement, not the gate)")
    args = parser.parse_args()
    if args.report.exists():
        raise SystemExit("report already exists; choose a new path")
    manifest = json.loads((args.bundle / ".aether/aether.json").read_text())
    contract = manifest["kokoro"]
    vocabulary = contract["phonemes"]["vocabulary"]
    for override in args.compute:
        role, kind = override.split("=")
        manifest["graphs"][role]["compute"] = kind
    graphs = {role: Graph(args.bundle / g["file"].split(":", 1)[1], g["function"], g["compute"])
              for role, g in manifest["graphs"].items()}
    index = json.loads((args.reference / "references.json").read_text())
    audio_dir = args.report.with_suffix("")
    audio_dir.mkdir(parents=True)
    report = {"bundle": str(args.bundle), "reference": str(args.reference),
              "compute": {r: g["compute"] for r, g in manifest["graphs"].items()},
              "bars": {"tokens": "exact", "frames": "exact", "relL2": f"<= min({MAX_REL_L2}, {FLOOR_MULTIPLE} x fp32-vs-fp64)",
                       "logMelMean": f"<= {MAX_LOG_MEL}", "repeat": "bit-identical"},
              "metrics": speech_metrics.DEFINITION, "cases": [], "checks": 0, "passed": 0}

    def check(ok: bool, what: str) -> bool:
        report["checks"] += 1
        report["passed"] += int(ok)
        if not ok:
            print(f"FAIL {what}", file=sys.stderr, flush=True)
        return ok

    first = None
    for entry in index["cases"]:
        name = entry["name"]
        case = json.loads((args.reference / name / "case.json").read_text())
        expected = np.frombuffer((args.reference / name / "audio.f32").read_bytes(), dtype="<f4")
        voice = contract["voices"][case["voice"]]
        table = manifest["tables"][voice["table"]]
        pack = np.frombuffer((args.bundle / table["file"]).read_bytes(), dtype="<f4").reshape(table["shape"])
        style = pack[len(case["phonemes"]) - 1][None, :].copy()
        ids = tokenize(vocabulary, case["phonemes"], contract["phonemes"]["maxPhonemes"])
        audio, sums, frames, bucket, timing = synthesize(graphs["durations"], graphs["decoder"], contract, ids, style,
                                                         case["speed"], case["seed"])
        repeat, *_ = synthesize(graphs["durations"], graphs["decoder"], contract, ids, style, case["speed"], case["seed"])
        floor = case["float64_upstream"]
        bound = min(MAX_REL_L2, FLOOR_MULTIPLE * floor["rel_l2"])
        record = {"name": name, "voice": case["voice"], "speed": case["speed"], "phonemes": len(case["phonemes"]),
                  "frames": int(frames.sum()), "bucket": bucket, "seconds": round(audio.size / SAMPLE_RATE, 2),
                  "durationSumsMaxAbs": float(np.abs(sums - np.array(case["duration_sums"], np.float32)).max()),
                  "float64Floor": floor, "relL2Bound": bound, **timing}
        ok = check(ids == case["input_ids"], f"{name}: token ids")
        ok &= check(frames.tolist() == case["frames"], f"{name}: frames per token")
        if ok and audio.size == expected.size:
            value = speech_metrics.compare(audio, expected)
            record["native"] = value
            ok &= check(value["rel_l2"] <= bound, f"{name}: rel L2 {value['rel_l2']:.4f} > {bound:.4f}")
            ok &= check(value["log_mel_mean_abs"] <= MAX_LOG_MEL,
                        f"{name}: log-mel mean {value['log_mel_mean_abs']:.4f} > {MAX_LOG_MEL}")
        else:
            ok &= check(False, f"{name}: sample count {audio.size} vs {expected.size}")
        ok &= check(np.array_equal(audio, repeat), f"{name}: repeat bit-identical")
        record["repeatIdentical"] = bool(np.array_equal(audio, repeat))
        record["passed"] = bool(ok)
        write_wav(audio_dir / f"{name}.wav", audio)
        write_wav(audio_dir / f"{name}-reference.wav", expected)
        report["cases"].append(record)
        print(json.dumps({k: record[k] for k in ("name", "frames", "bucket", "passed")}
                         | {k: round(v, 5) for k, v in record.get("native", {}).items() if k in ("rel_l2", "log_mel_mean_abs")}),
              flush=True)
        if first is None:
            first = (ids, style)

    if args.probe_compute and first is not None:
        ids, style = first
        reference = graphs["durations"](input_ids=np.array([ids], np.int32), style=style)["duration"]
        gpu = probe_gpu(args.bundle, manifest, "durations", {"input_ids": np.array([ids], np.int32), "style": style},
                        "duration", audio_dir / "probe")
        if "output" in gpu:
            gpu["maxAbsVsCPU"] = float(np.abs(gpu.pop("output") - reference).max())
        report["probe"] = {"durationsOnGPU": gpu}
        print(json.dumps({"probe": report["probe"]}), flush=True)

    report["ok"] = report["checks"] == report["passed"]
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"checks": report["checks"], "passed": report["passed"], "ok": report["ok"]}))
    sys.exit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
