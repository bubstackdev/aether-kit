//! Explicit numerical phase gate; building this executable performs no inference.
use corsairs::{
    BFloat16, Compute, DType, LoadOptions, MetalBufferMap, MetalValueMap, Runtime, Tensor,
    TensorMap,
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{path::Path, time::Instant};

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
        _ => return Err("unsupported phase probe dtype".into()),
    };
    let bytes = std::fs::read(item["path"].as_str().ok_or("tensor path missing")?)?;
    if hash(&bytes) != item["sha256"] {
        return Err("phase tensor hash differs".into());
    }
    Ok(Tensor::from_bytes(
        dtype,
        serde_json::from_value::<Vec<usize>>(item["shape"].clone())?,
        bytes,
    )?)
}
fn values(tensor: &Tensor) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    match tensor.dtype() {
        DType::Float32 => Ok(tensor.to_vec::<f32>()?.into_iter().map(f64::from).collect()),
        DType::BFloat16 => Ok(tensor
            .to_vec::<BFloat16>()?
            .into_iter()
            .map(|v| f64::from(v.to_f32()))
            .collect()),
        _ => Err("unsupported phase output dtype".into()),
    }
}
fn selected(tensor: &Tensor, spec: &Value) -> Result<Tensor, Box<dyn std::error::Error>> {
    let Some(rows) = spec["row_indices"].as_array() else {
        return Ok(tensor.clone());
    };
    let native: Vec<usize> = serde_json::from_value(spec["native_shape"].clone())?;
    if tensor.shape() != native || native.len() != 3 || native[0] != 1 {
        return Err("sampled phase native shape differs".into());
    }
    let width = match tensor.dtype() {
        DType::BFloat16 => 2,
        DType::Float32 => 4,
        _ => return Err("unsupported sampled phase dtype".into()),
    };
    let stride = native[2] * width;
    let mut bytes = Vec::with_capacity(rows.len() * stride);
    for row in rows {
        let row = row.as_u64().ok_or("invalid sampled row")? as usize;
        if row >= native[1] {
            return Err("sampled row exceeds native shape".into());
        }
        bytes.extend_from_slice(&tensor.as_bytes()[row * stride..(row + 1) * stride]);
    }
    Ok(Tensor::from_bytes(
        tensor.dtype(),
        [1, rows.len(), native[2]],
        bytes,
    )?)
}
fn finite(tensor: &Tensor) -> bool {
    match tensor.dtype() {
        DType::BFloat16 => tensor
            .as_bytes()
            .as_chunks::<2>()
            .0
            .iter()
            .all(|c| u16::from_le_bytes([c[0], c[1]]) & 0x7f80 != 0x7f80),
        DType::Float32 => tensor
            .as_bytes()
            .as_chunks::<4>()
            .0
            .iter()
            .all(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]) & 0x7f800000 != 0x7f800000),
        _ => false,
    }
}
fn compare(
    actual: &Tensor,
    expected: &Tensor,
    limit: f64,
) -> Result<Value, Box<dyn std::error::Error>> {
    if actual.shape() != expected.shape() || actual.dtype() != expected.dtype() {
        return Err("phase output dtype/shape differs".into());
    }
    let (mut error, mut norm, mut maximum) = (0.0, 0.0, 0.0f64);
    let mut finite = true;
    for (a, b) in values(actual)?.into_iter().zip(values(expected)?) {
        finite &= a.is_finite() && b.is_finite();
        error += (a - b) * (a - b);
        norm += b * b;
        maximum = maximum.max((a - b).abs());
    }
    let l2 = (error / norm.max(1e-30)).sqrt();
    Ok(
        json!({"passed":finite && l2.is_finite() && l2 <= limit,"finite":finite,
        "relative_l2":l2,"max_absolute_error":maximum,"sha256":hash(actual.as_bytes()),
        "reference_sha256":hash(expected.as_bytes()),"byte_identical":actual.as_bytes()==expected.as_bytes(),
        "shape":actual.shape()}),
    )
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 2 {
        return Err("usage: h3_phase_probe PROBE.json REPORT.json".into());
    }
    let destination = Path::new(&args[1]);
    if destination.exists() {
        return Err("phase report already exists".into());
    }
    let probe: Value = serde_json::from_slice(&std::fs::read(&args[0])?)?;
    let limit = probe["predefined_relative_l2_limit"]
        .as_f64()
        .filter(|v| *v == 0.02)
        .ok_or("phase probe requires the predefined 0.02 limit")?;
    let started = Instant::now();
    let mut report = json!({"schema":1,"passed":false,"scope":probe["scope"],
        "corsairs_version":"0.3.0","predefined_relative_l2_limit":limit,"function_reuse":"same model hash/entry point within each route; fresh functions between routes","cases":[]});
    std::fs::write(destination, serde_json::to_vec_pretty(&report)?)?;
    let result: Result<bool, Box<dyn std::error::Error>> = futures_executor::block_on(async {
        let runtime = Runtime::new()?;
        let mut passed = true;
        for route in ["owned", "metal_automatic", "metal_bound"] {
            let mut functions = std::collections::BTreeMap::new();
            for case in probe["cases"].as_array().ok_or("phase cases missing")? {
                let inputs: TensorMap = case["inputs"]
                    .as_array()
                    .ok_or("inputs missing")?
                    .iter()
                    .map(|v| Ok((v["name"].as_str().ok_or("input name missing")?, tensor(v)?)))
                    .collect::<Result<_, Box<dyn std::error::Error>>>()?;
                let expected: TensorMap = case["outputs"]
                    .as_array()
                    .ok_or("outputs missing")?
                    .iter()
                    .map(|v| Ok((v["name"].as_str().ok_or("output name missing")?, tensor(v)?)))
                    .collect::<Result<_, Box<dyn std::error::Error>>>()?;
                let model = &case["model"];
                let path = Path::new(model["path"].as_str().ok_or("model path missing")?);
                if hash(&std::fs::read(path.join("main.mlirb"))?) != model["sha256"] {
                    return Err("phase model hash differs".into());
                }
                let identity = format!("{}:{}", model["sha256"], model["function"]);
                if !functions.contains_key(&identity) {
                    let loaded = runtime
                        .load_model_with_options(
                            path,
                            LoadOptions::new(Compute::GpuPreferred)
                                .with_expect_frequent_reshapes(true),
                        )
                        .await?;
                    functions.insert(
                        identity.clone(),
                        loaded
                            .load_function(model["function"].as_str().ok_or("entry point missing")?)
                            .await?,
                    );
                }
                let function = functions.get(&identity).expect("function inserted");
                if route == "metal_bound"
                    && function
                        .descriptor()
                        .outputs
                        .iter()
                        .any(|v| v.shape.iter().any(Option::is_none))
                {
                    report["cases"].as_array_mut().unwrap().push(json!({"name":case["name"],"model_sha256":model["sha256"],
                        "routes":[{"route":route,"skipped":true,"reason":"public API explicitly requires fixed output descriptors; production uses automatic outputs for dynamic functions"}]}));
                    std::fs::write(destination, serde_json::to_vec_pretty(&report)?)?;
                    continue;
                }
                let queue = runtime.metal_queue().await?;
                let mut metal_inputs = MetalValueMap::new();
                for (name, value) in inputs.iter() {
                    metal_inputs.insert(name, queue.upload(value.clone()).await?);
                }
                let mut observation =
                    json!({"name":case["name"],"model_sha256":model["sha256"],"routes":[]});
                let begin = Instant::now();
                let actual = if route == "owned" {
                    let result = function.run(inputs.clone(), TensorMap::new()).await?;
                    if !result.states.is_empty() {
                        return Err("phase returned unexpected states".into());
                    }
                    result.outputs
                } else {
                    let mut outputs = MetalBufferMap::new();
                    if route == "metal_bound" {
                        for (name, value) in expected.iter() {
                            let spec = case["outputs"]
                                .as_array()
                                .unwrap()
                                .iter()
                                .find(|v| v["name"] == name)
                                .ok_or("output spec missing")?;
                            let shape: Vec<usize> =
                                serde_json::from_value(if spec["native_shape"].is_null() {
                                    spec["shape"].clone()
                                } else {
                                    spec["native_shape"].clone()
                                })?;
                            outputs.insert(name, queue.buffer(value.dtype(), shape).await?);
                        }
                    }
                    let result = function
                        .encode(&queue, metal_inputs.clone(), MetalBufferMap::new(), outputs)
                        .await?;
                    if !result.states.is_empty() {
                        return Err("phase returned unexpected Metal states".into());
                    }
                    let mut actual = TensorMap::new();
                    for (name, value) in result.outputs.iter() {
                        actual.insert(name, value.download().await?);
                    }
                    actual
                };
                if actual.len() != expected.len() {
                    return Err("phase output count differs".into());
                }
                let seconds = begin.elapsed().as_secs_f64();
                let mut checks = serde_json::Map::new();
                for (name, reference) in expected.iter() {
                    let spec = case["outputs"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .find(|v| v["name"] == name)
                        .ok_or("output spec missing")?;
                    let full = actual.get(name).ok_or("named output missing")?;
                    let selected = selected(full, spec)?;
                    let mut check = compare(&selected, reference, limit)?;
                    check["full_output_finite"] = finite(full).into();
                    check["native_shape"] = json!(full.shape());
                    check["sampled_row_indices"] = spec["row_indices"].clone();
                    if !finite(full) {
                        check["passed"] = false.into();
                    }
                    passed &= check["passed"] == true;
                    checks.insert(name.into(), check);
                }
                observation["routes"]
                    .as_array_mut()
                    .unwrap()
                    .push(json!({"route":route,"seconds":seconds,"outputs":checks}));
                println!("phase {} {route} completed", case["name"]);
                report["cases"].as_array_mut().unwrap().push(observation);
                std::fs::write(destination, serde_json::to_vec_pretty(&report)?)?;
                drop((expected, inputs, metal_inputs, queue));
            }
            drop(functions);
            runtime.drain().await?;
        }
        Ok(passed)
    });
    report["passed"] = matches!(result, Ok(true)).into();
    report["elapsed_seconds"] = started.elapsed().as_secs_f64().into();
    if let Err(error) = &result {
        report["error"] = error.to_string().into();
    }
    std::fs::write(destination, serde_json::to_vec_pretty(&report)?)?;
    match result {
        Ok(true) => Ok(()),
        Ok(false) => Err(
            "one or more original CPU phase comparisons failed; all completed routes retained"
                .into(),
        ),
        Err(error) => Err(error),
    }
}
