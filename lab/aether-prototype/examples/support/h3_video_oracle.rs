use corsairs::{Compute, DType, LoadOptions, MetalBufferMap, MetalValueMap, Runtime, Tensor};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{path::Path, time::Instant};
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
fn hash(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}
fn tensor(item: &Value) -> Result<Tensor> {
    if item["dtype"] != "float32" {
        return Err("original video oracle must be F32".into());
    }
    let bytes = std::fs::read(item["path"].as_str().ok_or("tensor path missing")?)?;
    if hash(&bytes) != item["sha256"] {
        return Err("original video oracle tensor hash differs".into());
    }
    Ok(Tensor::from_bytes(
        DType::Float32,
        serde_json::from_value::<Vec<usize>>(item["shape"].clone())?,
        bytes,
    )?)
}
pub async fn check(runtime: &Runtime, path: &Path) -> Result<Value> {
    let started = Instant::now();
    let probe: Value = serde_json::from_slice(&std::fs::read(path)?)?;
    let queue = runtime.metal_queue().await?;
    let mut inputs = MetalValueMap::new();
    for item in probe["parameters"]
        .as_array()
        .ok_or("VAE parameters missing")?
    {
        inputs.insert(
            item["name"].as_str().ok_or("VAE binding name missing")?,
            queue.upload(tensor(item)?).await?,
        );
    }
    inputs.insert("rows", queue.upload(tensor(&probe["input"])?).await?);
    let descriptor = &probe["model"];
    let path = Path::new(
        descriptor["path"]
            .as_str()
            .ok_or("VAE model path missing")?,
    );
    if hash(&std::fs::read(path.join("main.mlirb"))?) != descriptor["sha256"] {
        return Err("VAE model source hash differs".into());
    }
    let model = runtime
        .load_model_with_options(
            path,
            LoadOptions::new(Compute::GpuPreferred).with_expect_frequent_reshapes(true),
        )
        .await?;
    let function = model.load_function("main").await?;
    let mut result = function
        .encode(&queue, inputs, MetalBufferMap::new(), MetalBufferMap::new())
        .await?;
    if !result.states.is_empty() || result.outputs.len() != 1 {
        return Err("VAE oracle output bindings differ".into());
    }
    let actual = result
        .outputs
        .remove("pixels")
        .ok_or("VAE oracle pixels missing")?
        .download()
        .await?;
    let expected = tensor(&probe["reference"])?;
    if actual.dtype() != DType::Float32 || actual.shape() != expected.shape() {
        return Err("VAE oracle output shape/dtype differs".into());
    }
    let (mut error, mut norm) = (0.0f64, 0.0f64);
    for (a, b) in actual
        .to_vec::<f32>()?
        .into_iter()
        .zip(expected.to_vec::<f32>()?)
    {
        if !a.is_finite() || !b.is_finite() {
            return Err("VAE oracle output is nonfinite".into());
        }
        let (a, b) = (a as f64, b as f64);
        error += (a - b) * (a - b);
        norm += b * b;
    }
    let l2 = (error / norm.max(1e-30)).sqrt();
    if !l2.is_finite() || l2 > 0.02 {
        return Err(format!("VAE relative L2 {l2} exceeds predefined 0.02").into());
    }
    Ok(
        json!({"passed":true,"relative_l2":l2,"predefined_limit":0.02,"finite":true,"shape":actual.shape(),"seconds":started.elapsed().as_secs_f64(),"sha256":hash(actual.as_bytes()),"scope":"original CPU F32 VAE comparison at the exported diagnostic/tile shape; no full-profile numerical or visual-quality claim"}),
    )
}
