use super::{DIFFUSERS_COMMIT, H3Schedule, SNAPSHOT};
use crate::{Error, Result, error::require};
use corsairs::{
    Compute, Function, LoadOptions, Runtime, Tensor, TensorDescriptor, TensorHandleMap, TensorMap,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    io::Read,
    path::{Component, Path, PathBuf},
};

mod backbone;
pub use backbone::{H3BackboneQualification, H3BackboneSegment};

#[derive(Clone, Deserialize)]
struct Fixture {
    name: String,
    shape: Vec<usize>,
    dtype: String,
    file: PathBuf,
    sha256: String,
}
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct Oracle {
    schema_version: usize,
    snapshot: String,
    rows: usize,
    block: usize,
    diffusers_commit: String,
    #[serde(rename = "sourceTensorSHA256")]
    source_tensor_sha256: String,
    source_dtype_preserved: bool,
    inputs: Vec<Fixture>,
    reference: Option<Fixture>,
    cached_temb: Fixture,
}
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct SwiftRecord {
    #[serde(rename = "cacheSHA256")]
    cache_sha256: String,
    live_cached_byte_identical: bool,
    all_sample_hashes_identical: bool,
    samples: Vec<SwiftSample>,
}
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct SwiftSample {
    #[serde(rename = "outputSHA256")]
    output_sha256: String,
}

#[derive(Deserialize)]
struct ScheduleFixtures {
    items: Vec<ScheduleFixture>,
}
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct ScheduleFixture {
    step: usize,
    video_timestep: f32,
    audio_timestep: f32,
    conditioning_timestep: f32,
    tensor: Fixture,
}
#[derive(Deserialize)]
struct ScheduleReference {
    samples: Vec<ScheduleExpected>,
}
#[derive(Deserialize)]
struct ScheduleExpected {
    step: usize,
    #[serde(rename = "liveSHA256")]
    live: String,
    #[serde(rename = "cachedSHA256")]
    cached: String,
    #[serde(rename = "tableSHA256")]
    table: String,
}

#[derive(Debug, Serialize)]
pub struct H3ScheduleSample {
    pub step: usize,
    pub passed: bool,
    pub table_sha256: String,
    pub live_sha256: String,
    pub cached_sha256: String,
}
#[derive(Debug, Serialize)]
pub struct H3ScheduleQualification {
    pub passed: bool,
    pub scope: String,
    pub samples: Vec<H3ScheduleSample>,
    pub native_calls: usize,
}

struct Payload {
    dtype: String,
    shape: Vec<usize>,
    bytes: Vec<u8>,
}
impl Payload {
    fn lower_indices(self) -> Result<Self> {
        if self.dtype != "int64" {
            return Ok(self);
        }
        let mut bytes = Vec::with_capacity(self.bytes.len() / 2);
        for value in self.bytes.as_chunks::<8>().0 {
            let value = i32::try_from(i64::from_le_bytes(*value))
                .map_err(|_| Error::Invalid("H3 index does not fit Core AI int32".into()))?;
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        Ok(Self {
            dtype: "int32".into(),
            shape: self.shape,
            bytes,
        })
    }
    fn to_native(&self, descriptor: &TensorDescriptor) -> Result<Tensor> {
        // Match the source encoding against the concrete export descriptor.
        // Never substitute Float16 just because both encodings occupy 2 bytes.
        require(
            descriptor.dtype.name() == self.dtype,
            format!(
                "H3 input {} requires {}, descriptor reports {}; BF16-capable Corsairs is required",
                descriptor.name, self.dtype, descriptor.dtype
            ),
        )?;
        require(
            descriptor.shape.len() == self.shape.len()
                && descriptor
                    .shape
                    .iter()
                    .zip(&self.shape)
                    .all(|(d, n)| d.is_none_or(|v| v == *n)),
            format!("H3 input {} has incompatible shape", descriptor.name),
        )?;
        Ok(Tensor::from_bytes(
            descriptor.dtype,
            self.shape.clone(),
            self.bytes.clone(),
        )?)
    }
}

/// Read-only adapter to the lab's existing block/backbone evidence. This is a bounded
/// qualification workload, not an H3 clip-generation implementation.
pub struct H3Lab {
    root: PathBuf,
}

#[derive(Debug, Serialize)]
pub struct H3Qualification {
    pub passed: bool,
    pub scope: String,
    pub rows: usize,
    pub source_tensor_sha256: String,
    pub modulation_sha256: String,
    pub live_sha256: String,
    pub cached_sha256: String,
    pub expected_output_sha256: String,
    pub byte_identical_to_swift: bool,
    pub finite: bool,
    /// CPU comparisons have a separate tolerance; they cannot relax Swift parity.
    pub cpu_relative_l2: Option<f64>,
}

impl H3Lab {
    pub fn open(root: impl AsRef<Path>) -> Result<Self> {
        let root = std::fs::canonicalize(root)?;
        let environment: serde_json::Value =
            serde_json::from_slice(&std::fs::read(root.join("records/environment.json"))?)?;
        require(
            environment["unmodifiedDependencies"] == true
                && environment["diffusers"]["commit"] == DIFFUSERS_COMMIT,
            "H3 lab dependency provenance differs from the pinned baseline",
        )?;
        Ok(Self { root })
    }
    pub fn base_schedule(&self) -> Result<H3Schedule> {
        H3Schedule::from_lab(self.root.join("records/scheduler-50-forward.json"))
    }
    fn oracle(&self, rows: usize) -> Result<Oracle> {
        require(
            [128, 2048, 8192, 19350, 37838].contains(&rows),
            "no retained H3 oracle for this row count",
        )?;
        let oracle: Oracle = serde_json::from_slice(&std::fs::read(
            self.root.join(format!("records/block-oracle-{rows}.json")),
        )?)?;
        require(
            oracle.schema_version == 1
                && oracle.snapshot == SNAPSHOT
                && oracle.diffusers_commit == DIFFUSERS_COMMIT
                && oracle.rows == rows
                && oracle.block == 0
                && oracle.source_dtype_preserved,
            "H3 block oracle identity mismatch",
        )?;
        require(
            oracle.source_tensor_sha256.len() == 64,
            "invalid H3 source identity",
        )?;
        Ok(oracle)
    }
    /// Hash all recorded inputs and check exact integer lowering without
    /// acquiring a GPU or a native model. Missing assets are reported explicitly.
    pub fn inspect_inputs(&self, rows: usize) -> Result<serde_json::Value> {
        let oracle = self.oracle(rows)?;
        let mut bytes = 0;
        for fixture in oracle
            .inputs
            .iter()
            .chain([&oracle.cached_temb])
            .chain(oracle.reference.iter())
        {
            let payload = self.read(fixture)?;
            bytes += payload.bytes.len();
            payload.lower_indices()?;
        }
        let steps: Vec<_> = self.base_schedule()?.steps().collect();
        let assets: [String; 3] = ["live", "cached", "modulation"]
            .map(|arm| format!("forge/assets/block-{arm}-{rows}.aimodel"));
        let missing: Vec<_> = assets
            .iter()
            .filter(|path| !self.root.join(path).join("main.mlirb").is_file())
            .collect();
        Ok(
            serde_json::json!({"scope":"verified synthetic block inputs; not native inference or clip generation",
            "rows":rows,"input_bytes_verified":bytes,"source_tensor_sha256":oracle.source_tensor_sha256,
            "snapshot":SNAPSHOT,"diffusers_commit":DIFFUSERS_COMMIT,"joint_base_evaluations":steps.len(),
            "required_corsairs_addition":"bfloat16 descriptors, tensors, native handles and Metal bindings",
            "missing_block_assets":missing,"clip_generation_implemented":false}),
        )
    }
    fn read(&self, fixture: &Fixture) -> Result<Payload> {
        require(
            fixture
                .file
                .components()
                .all(|part| matches!(part, Component::Normal(_))),
            "H3 fixture path must stay relative to the lab",
        )?;
        let width: usize = match fixture.dtype.as_str() {
            "bfloat16" => 2,
            "float32" => 4,
            "int64" => 8,
            other => return Err(Error::Unsupported(format!("H3 fixture dtype {other}"))),
        };
        let count = fixture
            .shape
            .iter()
            .try_fold(width, |n, d| n.checked_mul(*d))
            .ok_or_else(|| Error::Invalid("H3 fixture shape overflow".into()))?;
        require(
            count <= corsairs::MAX_TENSOR_BYTES,
            "H3 fixture exceeds current tensor byte limit",
        )?;
        let path = self.root.join(&fixture.file);
        require(
            std::fs::metadata(&path)?.len() == count as u64,
            format!("H3 fixture byte count differs: {}", fixture.name),
        )?;
        let bytes = std::fs::read(path)?;
        require(
            hash(&bytes) == fixture.sha256,
            format!("H3 fixture SHA-256 differs: {}", fixture.name),
        )?;
        Ok(Payload {
            dtype: fixture.dtype.clone(),
            shape: fixture.shape.clone(),
            bytes,
        })
    }
    /// Exercise the original modulation, live block, and cached block serially.
    /// Model source streams and fixture bytes are hashed before execution.
    /// Requires Corsairs BF16 support and a permitted native environment.
    pub async fn qualify_block(&self, runtime: &Runtime, rows: usize) -> Result<H3Qualification> {
        let oracle = self.oracle(rows)?;
        self.inspect_inputs(rows)?;
        let swift: SwiftRecord = serde_json::from_slice(&std::fs::read(
            self.root.join(format!("records/block-gpu-{rows}.json")),
        )?)?;
        require(
            swift.live_cached_byte_identical
                && swift.all_sample_hashes_identical
                && !swift.samples.is_empty(),
            "Swift block reference did not pass its exactness gate",
        )?;
        let expected = swift.samples[0].output_sha256.clone();
        require(
            swift.samples.iter().all(|s| s.output_sha256 == expected),
            "inconsistent Swift block reference hashes",
        )?;
        let mut inputs = BTreeMap::new();
        for fixture in &oracle.inputs {
            require(
                inputs
                    .insert(fixture.name.clone(), self.read(fixture)?.lower_indices()?)
                    .is_none(),
                "duplicate H3 input name",
            )?;
        }
        for arm in ["modulation", "live", "cached"] {
            verify_asset(&self.asset(arm, rows))?;
        }

        let modulation = self.load(runtime, "modulation", rows).await?;
        let temb = inputs
            .get("temb")
            .ok_or_else(|| Error::Invalid("H3 fixture omitted temb".into()))?;
        let mut output = modulation
            .run(
                bind(&modulation, &BTreeMap::from([("temb".to_string(), temb)]))?,
                TensorMap::new(),
            )
            .await?;
        require(output.states.is_empty(), "unexpected H3 modulation state")?;
        let table = output
            .outputs
            .remove("modulations")
            .ok_or_else(|| Error::Invalid("H3 modulation output missing".into()))?;
        require(
            table.dtype().name() == "bfloat16" && table.shape() == [6, 9, 5376],
            "H3 GPU modulation dtype or shape differs",
        )?;
        let modulation_sha256 = hash(table.as_bytes());
        require(
            modulation_sha256 == swift.cache_sha256,
            "GPU modulation table differs from Swift",
        )?;
        drop(modulation);
        drain(runtime).await?;

        let live = self.load(runtime, "live", rows).await?;
        let input_refs = inputs.iter().map(|(k, v)| (k.clone(), v)).collect();
        let mut output = live
            .run(bind(&live, &input_refs)?, TensorMap::new())
            .await?;
        require(output.states.is_empty(), "unexpected H3 live block state")?;
        let live_output = output
            .outputs
            .remove("hidden_out")
            .ok_or_else(|| Error::Invalid("H3 hidden_out missing".into()))?;
        check_output(&live_output, rows)?;
        drop(live);
        drain(runtime).await?;

        inputs.insert(
            "temb".into(),
            Payload {
                dtype: "bfloat16".into(),
                shape: table.shape().to_vec(),
                bytes: table.as_bytes().to_vec(),
            },
        );
        let cached = self.load(runtime, "cached", rows).await?;
        let input_refs = inputs.iter().map(|(k, v)| (k.clone(), v)).collect();
        let mut output = cached
            .run(bind(&cached, &input_refs)?, TensorMap::new())
            .await?;
        require(output.states.is_empty(), "unexpected H3 cached block state")?;
        let cached_output = output
            .outputs
            .remove("hidden_out")
            .ok_or_else(|| Error::Invalid("H3 cached hidden_out missing".into()))?;
        check_output(&cached_output, rows)?;
        drop(cached);
        drain(runtime).await?;
        let live_sha256 = hash(live_output.as_bytes());
        let cached_sha256 = hash(cached_output.as_bytes());
        let values = bf16_floats(live_output.as_bytes());
        let finite = values.iter().all(|v| v.is_finite())
            && bf16_floats(cached_output.as_bytes())
                .iter()
                .all(|v| v.is_finite());
        let cpu_relative_l2 = if let Some(reference) = &oracle.reference {
            let reference = self.read(reference)?;
            require(
                reference.dtype == "bfloat16" && reference.shape == [1, rows, 5376],
                "invalid CPU reference signature",
            )?;
            Some(relative_l2(&bf16_floats(&reference.bytes), &values))
        } else {
            None
        };
        let byte_identical_to_swift = live_sha256 == expected && cached_sha256 == expected;
        Ok(H3Qualification { passed:byte_identical_to_swift && finite && cpu_relative_l2.is_none_or(|v|v.is_finite() && v<=0.02),
            scope:"original-weight synthetic block 0; excludes packing, conditioning, other blocks, scheduler updates and decode".into(),
            rows,source_tensor_sha256:oracle.source_tensor_sha256,modulation_sha256,live_sha256,cached_sha256,
            expected_output_sha256:expected,byte_identical_to_swift,finite,cpu_relative_l2 })
    }
    /// Reproduce block 0 at all 50 captured base timestep pairs. The modulation
    /// output feeds the cached block as the same native BF16 handle. Every
    /// table/live/cached output is compared with the independent Swift record.
    /// These are independent synthetic block calls, not a denoising trajectory.
    pub async fn qualify_schedule(&self, runtime: &Runtime) -> Result<H3ScheduleQualification> {
        let oracle = self.oracle(128)?;
        self.inspect_inputs(128)?;
        let fixtures: ScheduleFixtures = serde_json::from_slice(&std::fs::read(
            self.root.join("records/schedule-inputs.json"),
        )?)?;
        let reference: ScheduleReference = serde_json::from_slice(&std::fs::read(
            self.root.join("records/schedule-cache-parity.json"),
        )?)?;
        let steps: Vec<_> = self.base_schedule()?.steps().collect();
        require(
            fixtures.items.len() == 50 && reference.samples.len() == 50,
            "incomplete H3 schedule fixtures",
        )?;
        for (index, ((item, expected), step)) in fixtures
            .items
            .iter()
            .zip(&reference.samples)
            .zip(&steps)
            .enumerate()
        {
            require(
                item.step == index
                    && expected.step == index
                    && item.video_timestep.to_bits() == step.video_timestep.to_bits()
                    && item.audio_timestep.to_bits() == step.audio_timestep.to_bits()
                    && item.conditioning_timestep == 0.0,
                "H3 schedule fixture identity differs; nonzero keyframe noise is not qualified",
            )?;
            self.read(&item.tensor)?;
        }
        for arm in ["modulation", "live", "cached"] {
            verify_asset(&self.asset(arm, 128))?;
        }
        let modulation = self.load(runtime, "modulation", 128).await?;
        let live = self.load(runtime, "live", 128).await?;
        let cached = self.load(runtime, "cached", 128).await?;
        let descriptor = modulation
            .descriptor()
            .inputs
            .iter()
            .find(|d| d.name == "temb")
            .ok_or_else(|| Error::Invalid("H3 modulation temb descriptor missing".into()))?;
        let mut base = TensorHandleMap::new();
        for fixture in &oracle.inputs {
            if fixture.name == "temb" {
                continue;
            }
            let payload = self.read(fixture)?.lower_indices()?;
            let desc = live
                .descriptor()
                .inputs
                .iter()
                .find(|d| d.name == fixture.name)
                .ok_or_else(|| {
                    Error::Invalid(format!("H3 live descriptor omitted {}", fixture.name))
                })?;
            let tensor = runtime.upload(payload.to_native(desc)?).await?;
            base.insert(&fixture.name, tensor);
        }
        let mut samples = Vec::new();
        for (item, expected) in fixtures.items.iter().zip(&reference.samples) {
            let temb = runtime
                .upload(self.read(&item.tensor)?.to_native(descriptor)?)
                .await?;
            let mut result = modulation
                .run_native(
                    TensorHandleMap::from([("temb", temb.clone())]),
                    TensorHandleMap::new(),
                )
                .await?;
            require(result.states.is_empty(), "unexpected H3 modulation state")?;
            let table = result
                .outputs
                .remove("modulations")
                .ok_or_else(|| Error::Invalid("H3 native table missing".into()))?;
            require(
                table.dtype().name() == "bfloat16" && table.shape() == [6, 9, 5376],
                "native H3 table signature differs",
            )?;
            let table_sha256 = hash(table.download().await?.as_bytes());
            let mut live_inputs = base.clone();
            live_inputs.insert("temb", temb);
            let mut cached_inputs = base.clone();
            cached_inputs.insert("temb", table);
            let mut live_result = live.run_native(live_inputs, TensorHandleMap::new()).await?;
            let mut cached_result = cached
                .run_native(cached_inputs, TensorHandleMap::new())
                .await?;
            require(
                live_result.states.is_empty() && cached_result.states.is_empty(),
                "unexpected H3 native block state",
            )?;
            let live_output = live_result
                .outputs
                .remove("hidden_out")
                .ok_or_else(|| Error::Invalid("H3 native live output missing".into()))?
                .download()
                .await?;
            let cached_output = cached_result
                .outputs
                .remove("hidden_out")
                .ok_or_else(|| Error::Invalid("H3 native cached output missing".into()))?
                .download()
                .await?;
            check_output(&live_output, 128)?;
            check_output(&cached_output, 128)?;
            let live_sha256 = hash(live_output.as_bytes());
            let cached_sha256 = hash(cached_output.as_bytes());
            let live_values = bf16_floats(live_output.as_bytes());
            let cached_values = bf16_floats(cached_output.as_bytes());
            let finite = live_values
                .iter()
                .chain(&cached_values)
                .all(|v| v.is_finite());
            let passed = finite
                && live_sha256 == expected.live
                && cached_sha256 == expected.cached
                && table_sha256 == expected.table;
            samples.push(H3ScheduleSample {
                step: item.step,
                passed,
                table_sha256,
                live_sha256,
                cached_sha256,
            });
            if !passed {
                break;
            }
        }
        drop((modulation, live, cached, base));
        drain(runtime).await?;
        Ok(H3ScheduleQualification {passed:samples.len()==50 && samples.iter().all(|s|s.passed),
            scope:"block 0 only; all 50 base video/audio timestep pairs; zero conditioning noise; synthetic hidden states, not a generation trajectory".into(),
            native_calls:samples.len()*3,samples})
    }
    fn asset(&self, arm: &str, rows: usize) -> PathBuf {
        self.root
            .join(format!("forge/assets/block-{arm}-{rows}.aimodel"))
    }
    async fn load(&self, runtime: &Runtime, arm: &str, rows: usize) -> Result<Function> {
        let options = LoadOptions::new(Compute::GpuPreferred).with_expect_frequent_reshapes(true);
        let model = runtime
            .load_model_with_options(self.asset(arm, rows), options)
            .await
            .map_err(|source| Error::Stage {
                stage: format!("H3 {arm} model load"),
                source,
            })?;
        model
            .load_function(if arm == "modulation" {
                "modulation"
            } else {
                "block"
            })
            .await
            .map_err(|source| Error::Stage {
                stage: format!("H3 {arm} descriptor load (requires BF16-capable Corsairs)"),
                source,
            })
    }
}
fn bind(function: &Function, inputs: &BTreeMap<String, &Payload>) -> Result<TensorMap> {
    require(
        function.descriptor().states.is_empty(),
        "stateful H3 block export is unsupported",
    )?;
    require(
        function.descriptor().inputs.len() == inputs.len(),
        "H3 input names differ from export",
    )?;
    function
        .descriptor()
        .inputs
        .iter()
        .map(|desc| {
            let payload = inputs
                .get(&desc.name)
                .ok_or_else(|| Error::Invalid(format!("H3 missing input {}", desc.name)))?;
            Ok((desc.name.clone(), payload.to_native(desc)?))
        })
        .collect()
}
fn check_output(output: &Tensor, rows: usize) -> Result<()> {
    require(
        output.dtype().name() == "bfloat16" && output.shape() == [1, rows, 5376],
        "H3 hidden output must preserve BF16 [1,rows,5376]",
    )
}
async fn drain(runtime: &Runtime) -> Result<()> {
    runtime.drain().await?;
    Ok(())
}
fn hash(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}
fn verify_asset(path: &Path) -> Result<String> {
    let expected = std::fs::read(path.join("main.hash"))?;
    require(expected.len() == 32, "invalid H3 asset hash")?;
    let mut source = std::fs::File::open(path.join("main.mlirb"))?;
    let mut buffer = vec![0; 4 << 20];
    let mut digest = Sha256::new();
    loop {
        let count = source.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    let digest = digest.finalize();
    require(
        digest.as_slice() == expected,
        "H3 asset source stream hash mismatch",
    )?;
    Ok(digest.iter().map(|b| format!("{b:02x}")).collect())
}
fn bf16_floats(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|b| f32::from_bits((u16::from_le_bytes(*b) as u32) << 16))
        .collect()
}
fn relative_l2(reference: &[f32], candidate: &[f32]) -> f64 {
    let (mut error, mut norm) = (0.0, 0.0);
    for (a, b) in reference.iter().zip(candidate) {
        let (a, b) = (*a as f64, *b as f64);
        error += (a - b) * (a - b);
        norm += a * a;
    }
    (error / norm.max(1e-30)).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn index_lowering_is_exact_and_rejects_out_of_range() {
        let payload = |v: i64| Payload {
            dtype: "int64".into(),
            shape: vec![1],
            bytes: v.to_le_bytes().to_vec(),
        };
        let converted = payload(i32::MIN as i64).lower_indices().unwrap();
        assert_eq!(converted.bytes, i32::MIN.to_le_bytes());
        assert!(payload(i32::MAX as i64 + 1).lower_indices().is_err());
    }
    #[test]
    fn equal_width_fp16_is_never_accepted_for_bf16() {
        let payload = Payload {
            dtype: "bfloat16".into(),
            shape: vec![1],
            bytes: 0x3f80u16.to_le_bytes().to_vec(),
        };
        let descriptor = TensorDescriptor {
            name: "hidden".into(),
            dtype: corsairs::DType::Float16,
            shape: vec![Some(1)],
        };
        assert!(payload.to_native(&descriptor).is_err());
        assert_eq!(bf16_floats(&payload.bytes), [1.0]);
    }
}
