//! Bounded production workload timing; never reports a shortened run as a clip.
use aether::{
    AudioVideoRequest, ImageSize,
    h3::{H3Assets, H3Options, H3Pipeline, H3Progress},
};
use serde_json::{Value, json};
use std::{io::Write, path::PathBuf, time::Instant};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if !(2..=3).contains(&args.len()) {
        return Err("usage: h3_benchmark BUNDLE.json REPORT.json [PREFIX_STEPS=5]".into());
    }
    let steps = args.get(2).map(|s| s.parse()).transpose()?.unwrap_or(5);
    if !(3..=50).contains(&steps) {
        return Err(
            "benchmark requires 3..=50 prefix steps (both cardinalities plus steady-state)".into(),
        );
    }
    let bundle: Value = serde_json::from_slice(&std::fs::read(&args[0])?)?;
    let assets = H3Assets::open(&args[0])?;
    let schedule_steps = assets.recipe().evaluations();
    if steps > schedule_steps {
        return Err("prefix exceeds the bundle's sampling recipe".into());
    }
    let destination = PathBuf::from(&args[1]);
    if destination.exists() {
        return Err("benchmark report already exists".into());
    }
    let request = AudioVideoRequest::new(
        bundle["oracle"]["prompt"]
            .as_str()
            .ok_or("oracle prompt missing")?,
        ImageSize::new(
            bundle["width"].as_u64().ok_or("width missing")? as usize,
            bundle["height"].as_u64().ok_or("height missing")? as usize,
        ),
        bundle["frames"].as_u64().ok_or("frames missing")? as usize,
        1043,
    );
    let mut report = json!({"schema":1,"passed":false,"scope":"bounded prefix of source 50-step T2VA workload, followed by full decoder readback/stitch/hash; no PNG/WAV/mux, no completed render or quality claim","bundle":args[0],"corsairs_version":"0.3.0","source_schedule_steps":50,"measured_prefix_steps":steps,"width":request.size.width,"height":request.size.height,"frames":request.frames,"seed":request.seed,"prompt":request.prompt,"steady_state_definition":"steps 2 onward (zero-based), after both graph cardinalities have run once; one process, backend/OS caches retained, no cache purge or fixed clocks"});
    report["scope"] = "bounded prefix of bundle sampling recipe, followed by full decoder readback/stitch/hash; no PNG/WAV/mux or quality claim".into();
    report["recipe"] = serde_json::to_value(assets.recipe())?;
    report["source_schedule_steps"] = schedule_steps.into();
    std::fs::write(&destination, serde_json::to_vec_pretty(&report)?)?;
    let mut journal = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(destination.with_extension("events.jsonl"))?;
    let started = Instant::now();
    let mut events = Vec::new();
    let mut journal_error = None;
    let mut progress = |event| {
        let value = match event {
            H3Progress::PreparationCache { key, hit } => {
                json!({"event":"preparation_cache","key":key,"hit":hit})
            }
            H3Progress::Timing {
                stage,
                index,
                seconds,
            } => json!({"event":"timing","stage":stage,"index":index,"seconds":seconds}),
            H3Progress::Encoded => json!({"event":"encoded"}),
            H3Progress::ConditionerLayer { index } => {
                json!({"event":"conditioner_layer","index":index})
            }
            H3Progress::Modulation { block } => json!({"event":"modulation","block":block}),
            H3Progress::WeightsLoaded { block } => json!({"event":"weights_loaded","block":block}),
            H3Progress::Step {
                index,
                video_sha256,
                audio_sha256,
            } => {
                json!({"event":"step","index":index,"video_sha256":video_sha256,"audio_sha256":audio_sha256})
            }
            H3Progress::AudioDecoded { sha256 } => json!({"event":"audio_decoded","sha256":sha256}),
            H3Progress::VideoDecoded { sha256 } => json!({"event":"video_decoded","sha256":sha256}),
            H3Progress::VideoWindow { index } => json!({"event":"video_window","index":index}),
        };
        let mut value = value;
        value["elapsed_seconds"] = started.elapsed().as_secs_f64().into();
        if let Err(error) = writeln!(journal, "{value}") {
            journal_error = Some(error);
        }
        if value["event"] == "timing" && value["stage"] != "body_block" {
            println!(
                "{} {}: {:.3} s",
                value["stage"].as_str().unwrap_or(""),
                value["index"],
                value["seconds"].as_f64().unwrap_or(0.0)
            );
        }
        events.push(value);
    };
    let result = futures_executor::block_on(async {
        let phase = Instant::now();
        let mut pipeline = H3Pipeline::new(
            assets,
            H3Options {
                preparation_cache: std::env::var_os("AETHER_H3_PREPARATION_CACHE")
                    .map(PathBuf::from),
                ..H3Options::default()
            },
        )?;
        progress(H3Progress::Timing {
            stage: "initialization",
            index: None,
            seconds: phase.elapsed().as_secs_f64(),
        });
        pipeline
            .benchmark_with(&request, steps, &mut progress)
            .await?;
        pipeline.unload().await
    });
    drop(progress);
    report["elapsed_seconds"] = started.elapsed().as_secs_f64().into();
    report["events"] = events.into();
    if let Some(error) = journal_error {
        report["journal_error"] = error.to_string().into();
    }
    match &result {
        Ok(()) if report.get("journal_error").is_none() => report["passed"] = true.into(),
        Ok(()) => {}
        Err(error) => report["error"] = error.to_string().into(),
    }
    std::fs::write(&destination, serde_json::to_vec_pretty(&report)?)?;
    result?;
    if !report["passed"].as_bool().unwrap_or(false) {
        return Err("benchmark journal failed".into());
    }
    println!("{}", destination.display());
    Ok(())
}
