//! Explicit native adapter gate. Building this example performs no inference.
use corsairs::{
    BFloat16, Compute, DType, LoadOptions, MetalBufferMap, MetalValueMap, Runtime, Tensor,
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    path::{Path, PathBuf},
    time::Instant,
};

fn hash(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}
fn tensor(item: &Value) -> Result<Tensor, Box<dyn std::error::Error>> {
    let dtype = match item["dtype"].as_str() {
        Some("bfloat16") => DType::BFloat16,
        Some("float32") => DType::Float32,
        Some("int32") => DType::Int32,
        _ => return Err("unsupported probe dtype".into()),
    };
    let shape: Vec<usize> = serde_json::from_value(item["shape"].clone())?;
    let bytes = std::fs::read(item["path"].as_str().ok_or("probe path missing")?)?;
    if hash(&bytes) != item["sha256"] {
        return Err("probe tensor hash differs".into());
    }
    Ok(Tensor::from_bytes(dtype, shape, bytes)?)
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 2 {
        return Err("usage: h3_turbo_probe PROBE.json REPORT.json".into());
    }
    let destination = PathBuf::from(&args[1]);
    if destination.exists() {
        return Err("probe report already exists".into());
    }
    let probe: Value = serde_json::from_slice(&std::fs::read(&args[0])?)?;
    let started = Instant::now();
    let mut report = json!({"schema":1,"passed":false,"scope":"dynamic Turbo adapter through production Metal API: small original CPU comparison, then both full-row cardinalities; no full generation/quality claim","predefined_relative_l2_limit":0.02,"cases":[],"corsairs_version":"0.3.0"});
    std::fs::write(&destination, serde_json::to_vec_pretty(&report)?)?;
    let result: Result<(), Box<dyn std::error::Error>> = futures_executor::block_on(async {
        let runtime = Runtime::new()?;
        let mut cases = vec![
            json!({"model":probe["model"],"inputs":probe["inputs"],"reference":probe["reference"]}),
        ];
        cases.extend(
            probe["production_cases"]
                .as_array()
                .ok_or("production cases missing")?
                .iter()
                .cloned(),
        );
        for (index, case) in cases.iter().enumerate() {
            let begin = Instant::now();
            let queue = runtime.metal_queue().await?;
            let mut inputs = MetalValueMap::new();
            let mut shape = None;
            for item in case["inputs"]
                .as_array()
                .ok_or("inputs missing")?
                .iter()
                .chain(probe["parameters"].as_array().ok_or("parameters missing")?)
            {
                let value = tensor(item)?;
                if item["name"] == "hidden" {
                    shape = Some(value.shape().to_vec());
                }
                inputs.insert(
                    item["name"].as_str().ok_or("binding name missing")?,
                    queue.upload(value).await?,
                );
            }
            let model = &case["model"];
            let path = Path::new(model["path"].as_str().ok_or("model path missing")?);
            if hash(&std::fs::read(path.join("main.mlirb"))?) != model["sha256"] {
                return Err("probe model source differs".into());
            }
            let loaded = runtime
                .load_model_with_options(
                    path,
                    LoadOptions::new(Compute::GpuPreferred).with_expect_frequent_reshapes(true),
                )
                .await?;
            let function = loaded
                .load_function(model["function"].as_str().ok_or("function missing")?)
                .await?;
            let output = queue
                .buffer(DType::BFloat16, shape.ok_or("hidden shape missing")?)
                .await?;
            let setup = begin.elapsed().as_secs_f64();
            let mut durations = Vec::new();
            let mut hashes = Vec::new();
            let mut actual = None;
            for _ in 0..2 {
                let begin = Instant::now();
                let result = function
                    .encode(
                        &queue,
                        inputs.clone(),
                        MetalBufferMap::new(),
                        MetalBufferMap::from([("hidden_out", output.clone())]),
                    )
                    .await?;
                if result.outputs.len() != 1 || !result.states.is_empty() {
                    return Err("probe output names differ".into());
                }
                let value = output.download().await?;
                if value
                    .to_vec::<BFloat16>()?
                    .iter()
                    .any(|v| !v.to_f32().is_finite())
                {
                    return Err("nonfinite adapter probe output".into());
                }
                durations.push(begin.elapsed().as_secs_f64());
                hashes.push(hash(value.as_bytes()));
                actual = Some(value);
            }
            let mut observation = json!({"index":index,"setup_seconds":setup,"call_seconds":durations,"sha256":hashes,"finite":true,"repeat_exact":hashes[0]==hashes[1]});
            if index == 0 {
                let actual = actual.as_ref().ok_or("no output")?;
                let expected = tensor(&case["reference"])?;
                if actual.shape() != expected.shape() {
                    return Err("probe reference shape differs".into());
                }
                let (mut error, mut norm) = (0.0f64, 0.0f64);
                for (a, b) in actual
                    .to_vec::<BFloat16>()?
                    .iter()
                    .zip(expected.to_vec::<BFloat16>()?)
                {
                    let (a, b) = (a.to_f32() as f64, b.to_f32() as f64);
                    error += (a - b) * (a - b);
                    norm += b * b;
                }
                let l2 = (error / norm.max(1e-30)).sqrt();
                observation["relative_l2"] = l2.into();
                if !l2.is_finite() || l2 > 0.02 {
                    return Err(format!("adapter relative L2 {l2} exceeds 0.02").into());
                }
            }
            report["cases"].as_array_mut().unwrap().push(observation);
            std::fs::write(&destination, serde_json::to_vec_pretty(&report)?)?;
            if hashes[0] != hashes[1] {
                return Err("adapter probe repeats differ".into());
            }
            println!("completed adapter probe case {index}");
            drop((actual, output, function, loaded, inputs, queue));
            runtime.drain().await?;
        }
        Ok(())
    });
    report["elapsed_seconds"] = started.elapsed().as_secs_f64().into();
    report["passed"] = result.is_ok().into();
    if let Err(error) = &result {
        report["error"] = error.to_string().into();
    }
    std::fs::write(&destination, serde_json::to_vec_pretty(&report)?)?;
    result
}
