//! External-weight acceptance runner. No downloads, Python, or Swift in inference.
use aether::{
    GeneratedImage, ImageRequest, ImageSize,
    krea::{
        CancellationToken, KreaAssets, KreaOptions, KreaPipeline, LoadPolicy, Progress,
        ShapeSpecialization,
    },
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    path::{Path, PathBuf},
    time::Instant,
};

fn hash(values: &[f32]) -> String {
    let mut h = Sha256::new();
    for value in values {
        h.update(value.to_le_bytes());
    }
    h.finalize().iter().map(|b| format!("{b:02x}")).collect()
}
fn floats(path: impl AsRef<Path>) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let bytes = std::fs::read(path)?;
    if bytes.len() % 4 != 0 {
        return Err("partial float32 fixture".into());
    }
    Ok(bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|v| f32::from_le_bytes(*v))
        .collect())
}
fn read_json(path: impl AsRef<Path>) -> Result<Value, Box<dyn std::error::Error>> {
    Ok(serde_json::from_slice(&std::fs::read(path)?)?)
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<_> = std::env::args().skip(1).collect();
    if !(6..=7).contains(&a.len())
        || !["provided", "encoded"].contains(&a[5].as_str())
        || a.get(6)
            .is_some_and(|v| !["serial", "parallel"].contains(&v.as_str()))
    {
        return Err("usage: parity DYNAMIC_BUNDLE KREA_LAB FIXTURES SHAPE.json OUTPUT_DIRECTORY provided|encoded [serial|parallel]".into());
    }
    let lab = Path::new(&a[1]);
    let fixtures = Path::new(&a[2]);
    let out = Path::new(&a[4]);
    std::fs::create_dir_all(out)?;
    let cases = std::fs::read_to_string(fixtures.join("cases.jsonl"))?;
    let case = cases
        .lines()
        .map(serde_json::from_str::<Value>)
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .find(|v| v["id"] == "tower-768x1344")
        .ok_or("missing tower fixture")?;
    let file = |key: &str| -> Result<PathBuf, Box<dyn std::error::Error>> {
        Ok(fixtures.join(case["files"][key].as_str().ok_or("missing fixture file")?))
    };
    let noise = floats(file("noise")?)?;
    let text = floats(file("text")?)?;
    let id_bytes = std::fs::read(file("ids")?)?;
    if !id_bytes.len().is_multiple_of(4) {
        return Err("partial token ID fixture".into());
    }
    let ids: Vec<i32> = id_bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|v| i32::from_le_bytes(*v))
        .collect();
    let masks = std::fs::read(file("mask")?)?;
    if masks.len() != ids.len() * 4
        || masks
            .as_chunks::<4>()
            .0
            .iter()
            .any(|v| i32::from_le_bytes(*v) != 1)
    {
        return Err("fixture mask is not all ones".into());
    }
    let encoded = a[5] == "encoded";
    let reference_path = lab.join(if encoded {
        "records/round4/fixed-encoded-tower/record.json"
    } else {
        "records/round4/fixed-paired-tower/record.json"
    });
    let reference = read_json(&reference_path)?;
    let row = if encoded {
        &reference["runs"][0]
    } else {
        &reference["pairs"][0]
    };
    let steps = if encoded {
        &row["stepLatentsSHA256"]
    } else {
        &row["stepLatentsSHA256"][1]
    };
    let pixels = if encoded {
        &row["pixelsSHA256"]
    } else {
        &row["pixelsSHA256"][1]
    };
    let expected_steps = steps.as_array().ok_or("missing reference step hashes")?;
    let mut assets = KreaAssets::open(&a[0])?;
    eprintln!("Verifying model streams and offline specialization...");
    assets.verify()?;
    assets.add_specialization(ShapeSpecialization::from_file(&a[3])?)?;
    let load_policy = a.get(6).map(String::as_str).unwrap_or("parallel");
    let options = KreaOptions {
        loads: if load_policy == "serial" {
            LoadPolicy::Serial
        } else {
            LoadPolicy::Parallel
        },
        ..KreaOptions::default()
    };
    let mut pipeline = KreaPipeline::new(assets, options)?;
    let request = ImageRequest::new(
        case["prompt"].as_str().ok_or("missing prompt")?,
        ImageSize::new(768, 1344),
        71344,
    );
    let prompt = pipeline.prepare_prompt(&request.prompt)?;
    if prompt.token_ids() != ids {
        return Err("Rust prompt tokens differ from the saved reference IDs".into());
    }
    let mut events = Vec::new();
    let mut matched = true;
    let mut step_count = 0;
    let started = Instant::now();
    let mut progress = |event: Progress<'_>| {
        let value = match event {
            Progress::TextReady {
                values,
                rows,
                cached,
            } => {
                let actual = hash(values);
                let passed = actual == row["encodedTextSHA256"].as_str().unwrap_or("");
                matched &= passed;
                json!({"event":"text", "rows":rows, "cached":cached, "sha256":actual, "passed":passed})
            }
            Progress::DenoiserReady { specialized } => {
                matched &= specialized;
                json!({"event":"denoiser_ready", "specialized":specialized})
            }
            Progress::Step { index, latents, .. } => {
                step_count += 1;
                let actual = hash(latents);
                let passed =
                    Some(actual.as_str()) == expected_steps.get(index).and_then(Value::as_str);
                matched &= passed;
                json!({"event":"step", "step":index, "sha256":actual, "passed":passed})
            }
            Progress::Decoded => json!({"event":"decoded"}),
        };
        println!("{value}");
        events.push(value);
    };
    let cancel = CancellationToken::default();
    let result: aether::Result<GeneratedImage> = futures_executor::block_on(async {
        if encoded {
            pipeline
                .generate_with(&request, Some(noise), &cancel, &mut progress)
                .await
        } else {
            pipeline
                .generate_conditioned(&request, &text, Some(noise), &cancel, &mut progress)
                .await
        }
    });
    let error = result.as_ref().err().map(ToString::to_string);
    if let Ok(image) = result {
        let actual = hash(image.pixels_chw());
        matched &= Some(actual.as_str()) == pixels.as_str();
        events.push(json!({"event":"pixels", "sha256":actual, "passed":Some(actual.as_str()) == pixels.as_str()}));
        image.save_png(out.join("image.png"))?;
    } else {
        matched = false;
    }
    matched &= step_count == 8;
    let report = json!({"passed":matched, "mode":a[5], "load_policy":load_policy, "seconds":started.elapsed().as_secs_f64(),
        "token_ids_match":true, "model_streams_verified":true, "shape_registration_verified":true,
        "reference":reference_path, "events":events, "error":error});
    std::fs::write(out.join("report.json"), serde_json::to_vec_pretty(&report)?)?;
    if !matched {
        return Err(format!("parity failed; see {}", out.join("report.json").display()).into());
    }
    Ok(())
}
