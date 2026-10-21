use super::{DIFFUSERS_COMMIT, H3Geometry, H3Recipe, H3Schedule, SNAPSHOT};
use crate::{AudioVideoRequest, Error, ImageSize, Result, error::require};
use corsairs::{Compute, DType, Function, LoadOptions, Runtime, Tensor};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeSet,
    io::Read,
    path::{Path, PathBuf},
};

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct ModelAsset {
    pub path: PathBuf,
    pub function: String,
    pub sha256: String,
}
impl ModelAsset {
    pub async fn load(&self, runtime: &Runtime) -> Result<Function> {
        let model = runtime
            .load_model_with_options(
                &self.path,
                LoadOptions::new(Compute::GpuPreferred).with_expect_frequent_reshapes(true),
            )
            .await
            .map_err(|source| Error::Stage {
                stage: format!("H3 load {}", self.path.display()),
                source,
            })?;
        model
            .load_function(&self.function)
            .await
            .map_err(|source| Error::Stage {
                stage: format!("H3 function {} in {}", self.function, self.path.display()),
                source,
            })
    }
    fn verify(&self) -> Result<()> {
        require(
            self.sha256.len() == 64 && hash_file(&self.path.join("main.mlirb"))? == self.sha256,
            "H3 model source hash differs from bundle",
        )?;
        let raw = std::fs::read(self.path.join("main.hash"))?;
        require(
            raw.iter().map(|b| format!("{b:02x}")).collect::<String>() == self.sha256,
            "H3 model main.hash differs from bundle",
        )
    }
}
#[derive(Debug, Clone, Deserialize, Serialize)]
pub(crate) struct TensorFile {
    pub name: String,
    pub path: PathBuf,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub sha256: String,
}
#[derive(Debug, Clone, Deserialize, Serialize)]
pub(crate) struct AdapterIdentity {
    pub repository: String,
    pub revision: String,
    pub sha256: String,
    pub execution: String,
    pub rank: usize,
    pub alpha: f32,
}
impl TensorFile {
    pub fn read(&self) -> Result<Tensor> {
        let dtype = match self.dtype.as_str() {
            "bfloat16" => DType::BFloat16,
            "float32" => DType::Float32,
            other => return Err(Error::Unsupported(format!("H3 tensor dtype {other}"))),
        };
        let width = if dtype == DType::BFloat16 {
            2usize
        } else {
            4usize
        };
        let count = self
            .shape
            .iter()
            .try_fold(width, |n, d| n.checked_mul(*d))
            .ok_or_else(|| Error::Invalid("H3 tensor shape overflow".into()))?;
        require(
            count <= corsairs::MAX_TENSOR_BYTES
                && std::fs::metadata(&self.path)?.len() == count as u64,
            "H3 tensor extent differs or exceeds Corsairs limit",
        )?;
        let bytes = std::fs::read(&self.path)?;
        require(
            hash(&bytes) == self.sha256,
            format!("H3 tensor hash differs: {}", self.name),
        )?;
        Ok(Tensor::from_bytes(dtype, self.shape.clone(), bytes)?)
    }
}
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct ConditionerAssets {
    pub embedding: ModelAsset,
    pub layers: Vec<ModelAsset>,
}
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct Manifest {
    schema: usize,
    complete: bool,
    snapshot: String,
    diffusers_commit: String,
    #[serde(default)]
    pub recipe: H3Recipe,
    #[serde(default)]
    pub transformer_weight_layout: String,
    pub adapter: Option<AdapterIdentity>,
    pub width: usize,
    pub height: usize,
    pub frames: usize,
    pub text_rows: usize,
    pub tokenizer: PathBuf,
    pub schedule: PathBuf,
    pub prepare: ModelAsset,
    pub time: ModelAsset,
    pub heads: ModelAsset,
    pub bodies: Vec<ModelAsset>,
    pub modulations: Vec<ModelAsset>,
    pub parameters: Vec<Vec<TensorFile>>,
    pub rotary_cos: TensorFile,
    pub rotary_sin: TensorFile,
    pub video_decoder: ModelAsset,
    #[serde(default)]
    pub video_parameters: Vec<TensorFile>,
    pub video_spatial_tiles: Option<super::H3SpatialTiling>,
    pub audio_decoder: ModelAsset,
    pub conditioner: Option<ConditionerAssets>,
    pub video_window_latents: Option<usize>,
    pub video_window_frames: Option<usize>,
    pub video_window_stride: Option<usize>,
}

/// Immutable externally exported H3 profile. Model weights stay outside the
/// crate. A profile pins geometry and text-row count; unsupported shapes fail
/// before native work instead of padding or changing attention.
pub struct H3Assets {
    pub(crate) manifest: Manifest,
    pub(crate) schedule: H3Schedule,
}
impl H3Assets {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let manifest: Manifest = serde_json::from_slice(&std::fs::read(path)?)?;
        require(
            manifest.transformer_weight_layout == "raw_per_head_qkv_to_contiguous_qkv_v1",
            "H3 bundle lacks canonical raw-shard QKV conversion; re-export with the current Aether forge",
        )?;
        require(
            match (&manifest.recipe, &manifest.adapter) {
                (H3Recipe::Base50, None) => true,
                (H3Recipe::Turbo8, Some(a)) => {
                    a.repository == "lightx2v/Minimax-h3-Turbo"
                        && a.revision == "3ec17a324ced54151364f24f8b5fb6bf7e26414f"
                        && a.sha256
                            == "e16ac20824d6e6649b193806f8fb095639bd9946c97b1bb84b4248eab1cc807f"
                        && a.execution == "dynamic_lora"
                        && a.rank == 128
                        && a.alpha == 8.0
                }
                _ => false,
            },
            "H3 recipe lacks its supported adapter identity or execution contract",
        )?;
        let bindings = match manifest.recipe {
            H3Recipe::Base50 => 10,
            H3Recipe::Turbo8 => 22,
        };
        require(
            manifest.schema == 1
                && manifest.complete
                && manifest.snapshot == SNAPSHOT
                && manifest.diffusers_commit == DIFFUSERS_COMMIT,
            "H3 bundle is incomplete or has different provenance",
        )?;
        require(
            (1..=4096).contains(&manifest.text_rows)
                && manifest.bodies.len() == 2
                && manifest.modulations.len() == 50
                && manifest.parameters.len() == 50
                && manifest.parameters.iter().all(|p| {
                    p.len() == bindings
                        && p.iter().enumerate().all(|(i, t)| t.name == format!("w{i}"))
                }),
            "H3 bundle must contain both actual T2VA cardinalities and all 50 blocks",
        )?;
        require(
            manifest
                .conditioner
                .as_ref()
                .is_none_or(|c| c.layers.len() == 50),
            "H3 conditioner must preserve the unnormalized 50-layer tap",
        )?;
        let request = AudioVideoRequest::new(
            "",
            ImageSize::new(manifest.width, manifest.height),
            manifest.frames,
            0,
        );
        require(
            (manifest.video_parameters.is_empty() || manifest.video_parameters.len() == 703)
                && manifest
                    .video_parameters
                    .iter()
                    .enumerate()
                    .all(|(i, t)| t.name == format!("w{i}") && t.dtype == "float32"),
            "H3 functional video decoder bindings differ",
        )?;
        let geometry = H3Geometry::for_request(&request, true)?;
        if let Some(tiling) = &manifest.video_spatial_tiles {
            tiling.validate(request.size)?;
            require(
                !manifest.video_parameters.is_empty(),
                "H3 spatial export requires explicit decoder weights",
            )?;
        }
        require(
            geometry.frames == manifest.frames,
            "H3 bundle frame grid is not aligned",
        )?;
        require(
            geometry.latent_frames >= 7
                && ((manifest.frames == 22 && manifest.video_window_latents.is_none())
                    || (manifest.video_window_latents == Some(7)
                        && manifest.video_window_frames == Some(22)
                        && manifest.video_window_stride == Some(5))),
            "H3 bundle lacks the source temporal decoder-window contract",
        )?;
        let schedule =
            H3Schedule::from_json_for_recipe(&std::fs::read(&manifest.schedule)?, manifest.recipe)?;
        Ok(Self { manifest, schedule })
    }
    /// Full source-stream integrity before first use. Keep the bundle and all
    /// referenced files immutable after verification and while inference runs.
    pub fn verify(&self) -> Result<()> {
        let m = &self.manifest;
        let mut seen = BTreeSet::new();
        let models = [
            &m.prepare,
            &m.time,
            &m.heads,
            &m.video_decoder,
            &m.audio_decoder,
        ]
        .into_iter()
        .chain(&m.bodies)
        .chain(&m.modulations)
        .chain(
            m.conditioner
                .iter()
                .flat_map(|c| std::iter::once(&c.embedding).chain(&c.layers)),
        );
        for model in models {
            if seen.insert((&model.path, &model.sha256)) {
                model.verify()?;
            }
        }
        Ok(())
    }
    pub fn geometry(&self) -> Result<H3Geometry> {
        H3Geometry::for_request(
            &AudioVideoRequest::new(
                "",
                ImageSize::new(self.manifest.width, self.manifest.height),
                self.manifest.frames,
                0,
            ),
            true,
        )
    }
    pub fn recipe(&self) -> H3Recipe {
        self.manifest.recipe
    }
}
pub(crate) fn hash(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}
fn hash_file(path: &Path) -> Result<String> {
    let mut file = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0; 8 << 20];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    Ok(hasher
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect())
}
