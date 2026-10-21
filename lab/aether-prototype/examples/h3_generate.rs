//! Full H3 T2VA runner; diagnostic geometry is explicit and separately labeled.
use aether::{
    AudioVideoRequest, ImageSize,
    h3::{H3Assets, H3InitialLatents, H3Options, H3Pipeline, H3Progress},
};
use corsairs::{DType, Tensor};
use serde_json::{Value, json};
use std::{path::PathBuf, time::Instant};

fn tensor(value: &Value) -> Result<Tensor, Box<dyn std::error::Error>> {
    let dtype = match value["dtype"].as_str() {
        Some("bfloat16") => DType::BFloat16,
        Some("float32") => DType::Float32,
        _ => return Err("unsupported fixture dtype".into()),
    };
    let shape: Vec<_> = value["shape"]
        .as_array()
        .ok_or("fixture shape missing")?
        .iter()
        .map(|v| {
            v.as_u64()
                .map(|n| n as usize)
                .ok_or("invalid fixture dimension")
        })
        .collect::<Result<_, _>>()?;
    let bytes = std::fs::read(value["path"].as_str().ok_or("fixture path missing")?)?;
    use sha2::{Digest, Sha256};
    let digest = Sha256::digest(&bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect::<String>();
    if value["sha256"] != digest {
        return Err("provided fixture hash mismatch".into());
    }
    Ok(Tensor::from_bytes(dtype, shape, bytes)?)
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() < 3 {
        return Err("usage: h3_generate BUNDLE.json REPORT.json OUTPUT_DIR [--provided] [--diagnostic] [PROMPT]".into());
    }
    let bundle: Value = serde_json::from_slice(&std::fs::read(&args[0])?)?;
    let report_path = PathBuf::from(&args[1]);
    let output = PathBuf::from(&args[2]);
    let provided = args[3..].iter().any(|a| a == "--provided");
    let prepare_only = args[3..].iter().any(|a| a == "--prepare-only");
    let encode_only = args[3..].iter().any(|a| a == "--encode-only");
    if encode_only && provided {
        return Err("--encode-only uses the native conditioner; omit --provided".into());
    }
    let diagnostic = args[3..].iter().any(|a| a == "--diagnostic");
    let preparation_cache = args[3..]
        .iter()
        .find_map(|a| a.strip_prefix("--cache="))
        .map(PathBuf::from);
    let ffmpeg = args[3..]
        .iter()
        .find_map(|a| a.strip_prefix("--ffmpeg="))
        .map(PathBuf::from);
    let prompt = args[3..]
        .iter()
        .find(|a| !a.starts_with("--"))
        .map(String::as_str)
        .or(bundle["oracle"]["prompt"].as_str())
        .ok_or("supply a prompt")?;
    let request = AudioVideoRequest::new(
        prompt,
        ImageSize::new(
            bundle["width"].as_u64().ok_or("width missing")? as usize,
            bundle["height"].as_u64().ok_or("height missing")? as usize,
        ),
        bundle["frames"].as_u64().ok_or("frames missing")? as usize,
        1043,
    );
    let assets = H3Assets::open(&args[0])?;
    let mut report = json!({"schema":1,"passed":false,"scope":"full H3 T2VA Rust orchestration and both decoders; bundle pins weights and sampling recipe","recipe":assets.recipe(),"joint_evaluations":assets.recipe().evaluations(),"adapter":bundle["adapter"],"preparation_cache":preparation_cache,"ffmpeg":ffmpeg,"diagnostic":diagnostic,"provided_conditioning_and_noise":provided,"width":request.size.width,"height":request.size.height,"frames":request.frames,"prompt":prompt,"bundle":args[0]});
    if prepare_only {
        report["scope"]="input projection/text-refinement boundary versus original CPU model; no body or decode execution".into();
    }
    if encode_only {
        report["scope"]="native 50-layer conditioner tap versus original CPU hidden_states[50]; no denoising or decode".into();
    }
    if let Some(parent) = report_path.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&report_path, serde_json::to_vec_pretty(&report)?)?;
    let started = Instant::now();
    let mut events = Vec::new();
    let mut journal = std::fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(report_path.with_extension("events.jsonl"))?;
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
        println!("{value}");
        use std::io::Write;
        // Persist application checkpoints even if native code aborts later.
        if let Err(error) = writeln!(journal, "{value}") {
            eprintln!("H3 progress journal: {error}");
        }
        events.push(value);
    };
    let result = futures_executor::block_on(async {
        let mut pipeline = H3Pipeline::new(
            assets,
            H3Options {
                diagnostic,
                retain_decoded: true,
                preparation_cache,
            },
        )?;
        if encode_only {
            let actual = pipeline.encode_prompt(&request).await?;
            let reference = tensor(&bundle["oracle"]["text"])
                .map_err(|e| aether::Error::Invalid(e.to_string()))?;
            if actual.shape() != reference.shape() {
                return Err(aether::Error::Invalid(
                    "conditioner output shape differs".into(),
                ));
            }
            let mut error = 0.0;
            let mut norm = 0.0;
            for (a, b) in actual
                .to_vec::<corsairs::BFloat16>()?
                .iter()
                .zip(reference.to_vec::<corsairs::BFloat16>()?)
            {
                let (a, b) = (a.to_f32() as f64, b.to_f32() as f64);
                error += (a - b) * (a - b);
                norm += b * b;
            }
            let relative_l2 = (error / norm.max(1e-30)).sqrt();
            report["conditioner_oracle"] = json!({"relative_l2":relative_l2,"predefined_limit":0.02,"byte_identical":actual.as_bytes()==reference.as_bytes()});
            if !relative_l2.is_finite() || relative_l2 > 0.02 {
                return Err(aether::Error::Invalid(format!(
                    "conditioner relative L2 {relative_l2} exceeds 0.02"
                )));
            }
            pipeline.unload().await?;
            return Ok(None);
        }
        let clip = if provided {
            let oracle = &bundle["oracle"];
            let text =
                tensor(&oracle["text"]).map_err(|e| aether::Error::Invalid(e.to_string()))?;
            let latents = H3InitialLatents {
                video: tensor(&oracle["video"])
                    .map_err(|e| aether::Error::Invalid(e.to_string()))?
                    .to_vec::<f32>()?,
                audio: tensor(&oracle["audio"])
                    .map_err(|e| aether::Error::Invalid(e.to_string()))?
                    .to_vec::<f32>()?,
            };
            if prepare_only {
                let actual = pipeline
                    .prepare_conditioned(&request, text, latents)
                    .await?;
                let reference =
                    tensor(&oracle["hidden"]).map_err(|e| aether::Error::Invalid(e.to_string()))?;
                let a = actual.to_vec::<corsairs::BFloat16>()?;
                let b = reference.to_vec::<corsairs::BFloat16>()?;
                if actual.shape() != reference.shape() {
                    return Err(aether::Error::Invalid(
                        "prepared hidden shape differs".into(),
                    ));
                }
                let (mut error, mut norm) = (0.0, 0.0);
                for (a, b) in a.iter().zip(&b) {
                    let (a, b) = (a.to_f32() as f64, b.to_f32() as f64);
                    error += (a - b) * (a - b);
                    norm += b * b;
                }
                let l2 = (error / norm.max(1e-30)).sqrt();
                report["prepare_oracle"] = json!({"relative_l2":l2,"predefined_limit":0.02,"byte_identical":actual.as_bytes()==reference.as_bytes()});
                println!(
                    "{}",
                    json!({"event":"prepare_oracle","relative_l2":l2,"predefined_limit":0.02,"byte_identical":actual.as_bytes()==reference.as_bytes()})
                );
                if !l2.is_finite() || l2 > 0.02 {
                    return Err(aether::Error::Invalid(format!(
                        "prepared hidden relative L2 {l2} exceeds 0.02"
                    )));
                }
                return Ok(None);
            } else {
                Some(
                    pipeline
                        .generate_conditioned(&request, text, latents, &output, &mut progress)
                        .await?,
                )
            }
        } else {
            Some(
                pipeline
                    .generate_with(&request, &output, &mut progress)
                    .await?,
            )
        };
        pipeline.unload().await?;
        if let (Some(ffmpeg), Some(clip)) = (&ffmpeg, &clip) {
            let mux_started = Instant::now();
            clip.mux_mp4(ffmpeg, clip.directory().join("clip.mp4"))?;
            progress(H3Progress::Timing {
                stage: "mux",
                index: None,
                seconds: mux_started.elapsed().as_secs_f64(),
            });
        }
        Ok::<_, aether::Error>(clip)
    });
    report["events"] = events.into();
    report["elapsed_seconds"] = started.elapsed().as_secs_f64().into();
    match result {
        Ok(clip) => {
            report["passed"] = true.into();
            report["prepare_only"] = prepare_only.into();
            if let Some(clip) = clip {
                report["output"] = json!({"directory":clip.directory(),"frames":clip.frame_count(),"fps":clip.fps(),"sample_rate":clip.sample_rate(),"channels":clip.channels()});
            }
        }
        Err(error) => {
            report["error"] = error.to_string().into();
            std::fs::write(&report_path, serde_json::to_vec_pretty(&report)?)?;
            return Err(error.into());
        }
    }
    std::fs::write(&report_path, serde_json::to_vec_pretty(&report)?)?;
    println!("{}", report_path.display());
    Ok(())
}
