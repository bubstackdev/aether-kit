"""Summarize one bounded H3 run without treating its projection as a full render."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


def stats(values):
    return {"samples": len(values), "median": statistics.median(values), "mean": statistics.mean(values), "min": min(values), "max": max(values)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--previous-journal", type=Path)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--budget-seconds", type=float, default=1200)
    args = parser.parse_args()
    report = json.loads((args.run / "report.json").read_text())
    controller = json.loads((args.run / "controller.json").read_text())
    complete = report["passed"] and controller["passed"] and controller["child_joined"] and controller["exit_status"] == 0
    if not complete and not args.allow_partial:
        raise ValueError("Only completed, joined benchmark runs can be summarized")
    if not controller["child_joined"]:
        raise ValueError("Benchmark child is still active")
    events = report.get("events") or [json.loads(line) for line in (args.run / "report.events.jsonl").read_text().splitlines()]
    timings = [e for e in events if e["event"] == "timing"]
    by_stage = lambda stage: [e for e in timings if e["stage"] == stage]
    total_steps = by_stage("step_total")
    steady = [e["seconds"] for e in total_steps if e["index"] >= 2]
    count = len(total_steps)
    if not count:
        raise ValueError("No complete denoising step was measured")
    assert [e["index"] for e in total_steps] == list(range(count))
    if complete:
        assert count == report["measured_prefix_steps"]
        assert len(by_stage("body_block")) == count * 50
        assert len([e for e in events if e["event"] == "video_window"]) == (report["frames"] - 5) // 17
        assert sum(e["event"] == "audio_decoded" for e in events) == 1
        assert sum(e["event"] == "video_decoded" for e in events) == 1
    measured = report.get("elapsed_seconds", controller["elapsed_seconds"])
    remaining = report["source_schedule_steps"] - count
    step_values = [e["seconds"] for e in total_steps]
    preparation = total_steps[0]["elapsed_seconds"] - total_steps[0]["seconds"]
    summary = {"schema": 1, "passed": bool(complete), "benchmark_complete": bool(complete), "child_joined": controller["child_joined"], "exit_status": controller["exit_status"], "corsairs_version": report["corsairs_version"],
        "scope": report["scope"], "geometry": {k: report[k] for k in ("width", "height", "frames")},
        "planned_prefix_steps": report["measured_prefix_steps"], "measured_prefix_steps": count, "observed_benchmark_seconds": measured,
        "phases_seconds": {e["stage"]: e["seconds"] for e in timings if e["index"] is None},
        "step_total_seconds": [e["seconds"] for e in total_steps],
        "step_body_seconds": [e["seconds"] for e in by_stage("step_body")],
        "post_graph_warmup_step_seconds": stats(steady) if steady else None,
        "completed_step_body_block_seconds": stats([e["seconds"] for e in by_stage("body_block") if e["index"] < count * 50]),
        "video_window_seconds": [e["seconds"] for e in by_stage("video_decoder_window")],
        "projection_50_steps": {"seconds": measured + remaining * statistics.median(steady) if complete else None,
            "observed_preparation_seconds": preparation,
            "preparation_plus_50_steps_min_seconds": preparation + report["source_schedule_steps"] * min(step_values),
            "preparation_plus_50_steps_max_seconds": preparation + report["source_schedule_steps"] * max(step_values),
            "decoders_measured": bool(complete),
            "completed_render": False, "assumption": "Unmeasured steps cost the sampled steady step median; observed startup and decoder costs retained. No PNG/WAV/mux cost. Sample range is not a confidence interval. Partial-trajectory decoder inputs differ from a completed generation."},
        "input_reports_sha256": {name: hashlib.sha256((args.run/name).read_bytes()).hexdigest() for name in ["report.json", "controller.json"]}}
    if not complete:
        summary["planned_scope"] = summary["scope"]
        summary["scope"] = f"Preparation and {count} complete source-schedule denoising steps; interrupted before decoding; no completed render or quality claim"
        summary["projection_50_steps"]["assumption"] = "Partial observation only. Range applies the fastest/slowest completed step to all 50 and adds observed preparation. Step times were still rising; no stable throughput, confidence interval or measured decoder cost. No full-render timing is claimed."
    summary["requested_wall_budget_seconds"] = args.budget_seconds
    summary["within_requested_wall_budget_demonstrated"] = False
    if args.previous_journal:
        previous = {e["index"]: e for e in (json.loads(line) for line in args.previous_journal.read_text().splitlines()) if e["event"] == "step"}
        comparison = []
        for event in [e for e in events if e["event"] == "step"]:
            old = previous[event["index"]]
            comparison.append({"index": event["index"], "video_exact": event["video_sha256"] == old["video_sha256"], "audio_exact": event["audio_sha256"] == old["audio_sha256"]})
        summary["previous_development_run_prefix"] = comparison
        summary["previous_journal_sha256"] = hashlib.sha256(args.previous_journal.read_bytes()).hexdigest()
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
