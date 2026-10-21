//! Isolated decoder measurement with synthetic latents; no render/quality claim.
use aether::{
    AudioVideoRequest, ImageSize,
    h3::{H3Assets, H3Options, H3Pipeline, H3Progress},
};
use serde_json::{Value, json};
use std::{io::Write, path::PathBuf, time::Instant};
#[path = "support/h3_video_oracle.rs"]
mod video_oracle;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 2 {
        return Err("usage: h3_decode_probe BUNDLE.json REPORT.json".into());
    }
    let destination = PathBuf::from(&args[1]);
    if destination.exists() {
        return Err("decoder report already exists".into());
    }
    let bundle: Value = serde_json::from_slice(&std::fs::read(&args[0])?)?;
    let request = AudioVideoRequest::new(
        "",
        ImageSize::new(
            bundle["width"].as_u64().ok_or("width missing")? as usize,
            bundle["height"].as_u64().ok_or("height missing")? as usize,
        ),
        bundle["frames"].as_u64().ok_or("frames missing")? as usize,
        1043,
    );
    let mut report = json!({"schema":1,"passed":false,"scope":"both original decoders at full profile geometry with seeded synthetic latents; no denoising, media or quality claim","events":[]});
    report["original_video_oracle_predefined_limit"] = 0.02.into();
    std::fs::write(&destination, serde_json::to_vec_pretty(&report)?)?;
    let mut journal = std::fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(destination.with_extension("events.jsonl"))?;
    let started = Instant::now();
    let mut events = Vec::new();
    let mut journal_error = None;
    let mut progress = |event| {
        let mut value = match event {
            H3Progress::Timing {
                stage,
                index,
                seconds,
            } => json!({"event":"timing","stage":stage,"index":index,"seconds":seconds}),
            H3Progress::AudioDecoded { sha256 } => json!({"event":"audio_decoded","sha256":sha256}),
            H3Progress::VideoDecoded { sha256 } => json!({"event":"video_decoded","sha256":sha256}),
            H3Progress::VideoWindow { index } => json!({"event":"video_window","index":index}),
            _ => json!({"event":"unexpected"}),
        };
        value["elapsed_seconds"] = started.elapsed().as_secs_f64().into();
        if let Err(e) = writeln!(journal, "{value}") {
            journal_error = Some(e);
        }
        println!("{value}");
        events.push(value);
    };
    let result: Result<(), Box<dyn std::error::Error>> = futures_executor::block_on(async {
        let runtime = corsairs::Runtime::new()?;
        if let Some(path) = bundle["video_decoder_oracle"].as_str() {
            let oracle = video_oracle::check(&runtime, std::path::Path::new(path)).await?;
            progress(H3Progress::Timing {
                stage: "video_decoder_oracle",
                index: None,
                seconds: oracle["seconds"].as_f64().ok_or("oracle timing missing")?,
            });
            report["original_video_oracle"] = oracle;
            std::fs::write(&destination, serde_json::to_vec_pretty(&report)?)?;
            runtime.drain().await?;
        }
        let assets = H3Assets::open(&args[0])?;
        let mut pipeline = H3Pipeline::with_runtime(assets, H3Options::default(), runtime)?;
        pipeline
            .benchmark_decoders_with(&request, &mut progress)
            .await?;
        pipeline.unload().await?;
        Ok(())
    });
    drop(progress);
    report["events"] = events.into();
    report["elapsed_seconds"] = started.elapsed().as_secs_f64().into();
    report["passed"] = (result.is_ok() && journal_error.is_none()).into();
    if let Some(e) = journal_error {
        report["journal_error"] = e.to_string().into();
    }
    if let Err(e) = &result {
        report["error"] = e.to_string().into();
    }
    std::fs::write(&destination, serde_json::to_vec_pretty(&report)?)?;
    result?;
    if report["passed"] != true {
        return Err("decoder journal failed".into());
    }
    Ok(())
}
