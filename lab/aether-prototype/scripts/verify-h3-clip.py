"""CPU-only audit of a completed Turbo8 render; visual/audio acceptance is separate.

Checks the controller's end-to-end wall time and application journal, retained
raw hashes, exact decoded-frame count and audio/video stream metadata. A green
application report or an existing MP4 alone does not satisfy this audit.
"""
import argparse
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import subprocess


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def check_run(controller, report, events):
    require(controller["passed"] and controller["child_joined"] and controller["exit_status"] == 0,
            "Native child did not complete successfully and join")
    require(math.isfinite(controller["elapsed_seconds"]) and 0 < controller["elapsed_seconds"] <= 1200,
            "Render exceeds the 20-minute end-to-end limit")
    require(report["passed"] and report["recipe"] == "turbo8" and report["joint_evaluations"] == 8,
            "Application did not complete the Turbo8 recipe")
    require(not report["diagnostic"] and not report["provided_conditioning_and_noise"],
            "Acceptance requires production geometry, native prompt encoding and seeded latents")
    require((report["width"], report["height"], report["frames"]) == (960, 544, 124),
            "Render does not use the target five-second profile")
    require([e["index"] for e in events if e["event"] == "step"] == list(range(8)),
            "Journal must contain all eight complete joint evaluations")
    for kind in ("audio_decoded", "video_decoded"):
        require(sum(e["event"] == kind for e in events) == 1, "Missing or repeated decoder completion")
    require([e["index"] for e in events if e["event"] == "video_window"] == list(range(7)),
            "Production temporal windows did not all complete")
    require(sum(e["event"] == "timing" and e["stage"] == "mux" for e in events) == 1,
            "MP4 mux completion is missing")


def check_streams(path, expected, ffprobe):
    probe = subprocess.run([str(ffprobe), "-v", "error", "-count_frames", "-show_streams", "-show_format",
                            "-of", "json", str(path)], capture_output=True, text=True, check=True, timeout=60)
    require(not probe.stderr.strip(), "FFprobe reported a media decoding error")
    media = json.loads(probe.stdout)
    video = [s for s in media["streams"] if s["codec_type"] == "video"]
    audio = [s for s in media["streams"] if s["codec_type"] == "audio"]
    require(len(video) == len(audio) == 1, "MP4 must contain exactly one video and one audio stream")
    v, a = video[0], audio[0]
    require((v["width"], v["height"]) == (expected["width"], expected["height"]), "MP4 canvas differs")
    require(int(v["nb_read_frames"]) == expected["frames"], "MP4 decoded frame count differs")
    require(Fraction(v["avg_frame_rate"]) == expected["fps"], "MP4 frame rate differs")
    require(int(a["sample_rate"]) == 32000 and a["channels"] == 2, "MP4 must retain stereo 32 kHz audio")
    target = expected["frames"] / expected["fps"]
    for stream in (v, a):
        duration = float(stream["duration"])
        require(math.isfinite(duration) and abs(duration - target) <= 1 / expected["fps"],
                "MP4 stream duration differs by more than one video frame")
    return media


def check_qualified_window(path, run, build):
    """Validate the gate and render against the shared window, not just render time."""
    record = json.loads(path.read_text())
    elapsed, budget = record["elapsed_seconds"], record["wall_budget_seconds"]
    require(record["passed"] and not record.get("controller_cleanup_pending"),
            "Qualified-window coordinator did not finish successfully")
    require(math.isfinite(elapsed) and 0 < elapsed <= budget <= 1200,
            "Numerical gate plus render exceeds the shared 20-minute window")
    require(run.resolve() == (path.parent / "render").resolve(), "Coordinator belongs to a different render")
    require([s["kind"] for s in record["stages"]] == ["phase", "render"],
            "Qualified window must contain the gate followed by the render")
    source_root = build.parent.parent
    for key, filename in [("coordinator_sha256", "run-h3-qualified-render.py"),
                          ("controller_sha256", "run-h3-check.py")]:
        require(record[key] == sha(source_root / filename), "Frozen window-controller source identity differs")
    total = 0.0
    for stage in record["stages"]:
        kind = stage["kind"]
        controller_path = path.parent / kind / "controller.json"
        require(Path(stage["controller"]).resolve() == controller_path.resolve(), "Stage controller path differs")
        require(stage["controller_sha256"] == sha(controller_path), "Stage controller hash differs")
        child = json.loads(controller_path.read_text())
        identity = json.loads((source_root / kind / "manifest.json").read_text())
        require(stage["passed"] and child["passed"] and child["child_joined"] and child["exit_status"] == 0,
                "A numerical gate/render child failed or was not joined")
        require(stage["binary_sha256"] == child["binary_sha256"] == identity["binary_sha256"] and
                stage["input_sha256"] == child["input_sha256"] == identity["bundle_sha256"],
                "A qualified stage differs from its frozen build")
        seconds = child["elapsed_seconds"]
        require(math.isfinite(seconds) and 0 < seconds <= child["wall_budget_seconds"] and
                stage["elapsed_seconds"] == seconds, "Stage wall timing differs or exceeds its budget")
        total += seconds
    require(total <= elapsed, "Stage wall times exceed the complete window's wall time")
    return dict(passed=True, wall_seconds=elapsed, coordinator_sha256=sha(path))


def audit(run, build, ffprobe, coordinator=None):
    controller = json.loads((run / "controller.json").read_text())
    report = json.loads((run / "report.json").read_text())
    events = [json.loads(line) for line in (run / "report.events.jsonl").read_text().splitlines()]
    check_run(controller, report, events)
    identity = json.loads(build.read_text())
    require(identity["binary_sha256"] == controller["binary_sha256"] and
            identity["bundle_sha256"] == controller["input_sha256"], "Controller/build identity differs")
    window = check_qualified_window(coordinator, run, build) if coordinator else None
    clip = run / "clip"
    metadata = json.loads((clip / "media.json").read_text())
    require((metadata["width"], metadata["height"], metadata["frames"], metadata["fps"]) == (960, 544, 124, 24),
            "Source media metadata differs")
    expected_names = {f"{i:08}.png" for i in range(124)}
    require({p.name for p in (clip / "frames").iterdir()} == expected_names, "Source frame set differs")
    final = [e for e in events if e["event"] == "step"][-1]
    for name, expected_hash in (("video-latents.f32", final["video_sha256"]),
                                ("audio-latents.f32", final["audio_sha256"]),
                                ("video.f32", next(e["sha256"] for e in events if e["event"] == "video_decoded")),
                                ("audio.f32", next(e["sha256"] for e in events if e["event"] == "audio_decoded"))):
        require(sha(clip / name) == expected_hash, f"Retained {name} hash differs from native completion event")
    conditioning = json.loads((clip / "conditioning.json").read_text())
    require(conditioning["tap"] == 50 and conditioning["final_norm"] is False and
            sha(clip / "conditioning.bf16") == conditioning["sha256"], "Retained conditioning identity differs")
    media = check_streams(clip / "clip.mp4", metadata, ffprobe)
    return dict(schema=1, passed=True, scope="CPU integrity/media/timing audit only; human visual and audio acceptance still required",
                quality_accepted=False, wall_seconds=window["wall_seconds"] if window else controller["elapsed_seconds"],
                render_wall_seconds=controller["elapsed_seconds"], qualified_window=window,
                mp4=str(clip / "clip.mp4"), mp4_sha256=sha(clip / "clip.mp4"), media=media,
                build_sha256=sha(build), controller_sha256=sha(run / "controller.json"),
                report_sha256=sha(run / "report.json"), journal_sha256=sha(run / "report.events.jsonl"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--ffprobe", type=Path, required=True)
    parser.add_argument("--coordinator", type=Path, help="Require the numerical gate and render to share one verified 20-minute window")
    args = parser.parse_args()
    result = audit(args.run, args.build, args.ffprobe, args.coordinator)
    (args.run / "media-audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
