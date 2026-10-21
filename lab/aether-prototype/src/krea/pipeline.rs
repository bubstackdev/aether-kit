use super::{
    KreaAssets, KreaPrompt, KreaTokenizer,
    math::{self, FlowSchedule},
};
use crate::{Error, GeneratedImage, ImagePipeline, ImageRequest, Result, error::require};
use corsairs::{
    Compute, DType, Float16, Function, LoadOptions, ModelLoadRequest, Runtime, Tensor,
    TensorHandleMap, TensorMap,
};
use std::{
    collections::VecDeque,
    path::PathBuf,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Residency {
    /// Retain the fp16 text encoder, one denoiser set, and decoder for warm use.
    #[default]
    Resident,
    /// Release the text encoder after each cache miss. Higher preparation cost.
    Balanced,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LoadPolicy {
    Serial,
    #[default]
    Parallel,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct KreaOptions {
    pub residency: Residency,
    pub loads: LoadPolicy,
}

/// Cooperative cancellation checked between model calls. Native calls already
/// submitted finish under Corsairs ownership; cancellation cannot interrupt them.
#[derive(Debug, Clone, Default)]
pub struct CancellationToken(Arc<AtomicBool>);
impl CancellationToken {
    pub fn cancel(&self) {
        self.0.store(true, Ordering::Relaxed);
    }
    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Relaxed)
    }
    fn check(&self) -> Result<()> {
        if self.is_cancelled() {
            Err(Error::Cancelled)
        } else {
            Ok(())
        }
    }
}

/// Callbacks borrow checkpoint data only for the duration of the callback.
pub enum Progress<'a> {
    TextReady {
        values: &'a [f32],
        rows: usize,
        cached: bool,
    },
    DenoiserReady {
        specialized: bool,
    },
    Step {
        index: usize,
        total: usize,
        latents: &'a [f32],
    },
    Decoded,
}

type TextCache = VecDeque<(Vec<i32>, Tensor)>;

/// A long-lived, serialized generator. `&mut self` prevents overlapping renders
/// on this pipeline. Inject one Runtime to share Corsairs ownership if needed.
/// Pending native jobs retain their inputs even if a generation future drops.
pub struct KreaPipeline {
    runtime: Runtime,
    assets: KreaAssets,
    options: KreaOptions,
    tokenizer: KreaTokenizer,
    encoder: Option<Function>,
    decoder: Option<Function>,
    denoiser: Option<([PathBuf; 4], Vec<Function>)>,
    text_cache: TextCache,
}

impl KreaPipeline {
    pub fn new(assets: KreaAssets, options: KreaOptions) -> Result<Self> {
        Self::with_runtime(Runtime::new()?, assets, options)
    }
    pub fn with_runtime(
        runtime: Runtime,
        assets: KreaAssets,
        options: KreaOptions,
    ) -> Result<Self> {
        let tokenizer = KreaTokenizer::open(&assets.tokenizer)?;
        Ok(Self {
            runtime,
            assets,
            options,
            tokenizer,
            encoder: None,
            decoder: None,
            denoiser: None,
            text_cache: VecDeque::new(),
        })
    }
    pub fn prepare_prompt(&self, prompt: &str) -> Result<KreaPrompt> {
        self.tokenizer.prepare(prompt)
    }

    /// Standard prompt generation with optional reference noise and checkpoints.
    pub async fn generate_with(
        &mut self,
        request: &ImageRequest,
        noise: Option<Vec<f32>>,
        cancel: &CancellationToken,
        progress: &mut (impl FnMut(Progress<'_>) + Send),
    ) -> Result<GeneratedImage> {
        math::grid(request.size)?;
        FlowSchedule::turbo(request.steps.unwrap_or(8))?;
        cancel.check()?;
        let prompt = self.prepare_prompt(&request.prompt)?;
        self.generate_prepared(request, &prompt, noise, cancel, progress)
            .await
    }

    /// Replay exact rendered token IDs while using the real encoder. Useful for
    /// independent parity checks and callers with their own prompt frontend.
    pub async fn generate_prepared(
        &mut self,
        request: &ImageRequest,
        prompt: &KreaPrompt,
        noise: Option<Vec<f32>>,
        cancel: &CancellationToken,
        progress: &mut (impl FnMut(Progress<'_>) + Send),
    ) -> Result<GeneratedImage> {
        let (rows, cols) = math::grid(request.size)?;
        FlowSchedule::turbo(request.steps.unwrap_or(8))?;
        validate_noise(noise.as_deref(), rows * cols * 64)?;
        cancel.check()?;
        let cached = self
            .text_cache
            .iter()
            .position(|(ids, _)| ids == prompt.token_ids());
        let text = if let Some(index) = cached {
            let entry = self.text_cache.remove(index).expect("cache index exists");
            let result = entry.1.clone();
            self.text_cache.push_front(entry);
            result
        } else {
            let text = self.encode(prompt).await?;
            self.text_cache
                .push_front((prompt.token_ids().to_vec(), text.clone()));
            self.text_cache.truncate(2);
            text
        };
        cancel.check()?;
        let values: Vec<_> = text
            .to_vec::<Float16>()?
            .into_iter()
            .map(Float16::to_f32)
            .collect();
        progress(Progress::TextReady {
            values: &values,
            rows: prompt.text_rows(),
            cached: cached.is_some(),
        });
        self.render(request, text, noise, cancel, progress).await
    }

    /// Bypass encoding with supplied float32 text rows `[n,30720]`. They are
    /// narrowed to fp16 exactly as on the lab's provided-text acceptance path.
    pub async fn generate_conditioned(
        &mut self,
        request: &ImageRequest,
        text: &[f32],
        noise: Option<Vec<f32>>,
        cancel: &CancellationToken,
        progress: &mut (impl FnMut(Progress<'_>) + Send),
    ) -> Result<GeneratedImage> {
        let (rows, cols) = math::grid(request.size)?;
        FlowSchedule::turbo(request.steps.unwrap_or(8))?;
        validate_noise(noise.as_deref(), rows * cols * 64)?;
        require(
            text.len().is_multiple_of(30720) && (1..=512).contains(&(text.len() / 30720)),
            "text must be [1..=512,30720]",
        )?;
        let text = halves(text, [text.len() / 30720, 30720])?;
        self.render(request, text, noise, cancel, progress).await
    }

    async fn encode(&mut self, prompt: &KreaPrompt) -> Result<Tensor> {
        let encoder = if self.options.residency == Residency::Balanced {
            // Local ownership also releases the encoder if this future drops
            // during inference. The pending Corsairs job retains its own clone.
            load(&self.runtime, &self.assets.encoder, "encode", true).await?
        } else {
            if self.encoder.is_none() {
                self.encoder =
                    Some(load(&self.runtime, &self.assets.encoder, "encode", true).await?);
            }
            self.encoder.as_ref().expect("encoder loaded").clone()
        };
        let n = prompt.token_ids().len();
        let inputs = TensorMap::from([
            (
                "input_ids",
                Tensor::from_vec([1, n], prompt.token_ids().to_vec())?,
            ),
            (
                "position_ids",
                Tensor::from_vec([1, n], (0..n as i32).collect())?,
            ),
            ("attention_mask", Tensor::from_vec([1, n], vec![1i32; n])?),
        ]);
        // Keep balanced cleanup on both successful and failed model calls.
        let result = encoder.run(inputs, TensorMap::new()).await;
        drop(encoder);
        if self.options.residency == Residency::Balanced {
            self.runtime.drain().await?;
        }
        let mut result = result?;
        require(result.states.is_empty(), "unexpected Krea encoder state")?;
        let text = result
            .outputs
            .remove("text_embeds")
            .ok_or_else(|| Error::Invalid("encoder omitted text_embeds".into()))?;
        verify_tensor(&text, DType::Float16, &[n, 30720], "encoded text")?;
        let bytes = text.as_bytes()[34 * 30720 * 2..].to_vec();
        let selected = Tensor::from_bytes(DType::Float16, [n - 34, 30720], bytes)?;
        require(
            selected
                .to_vec::<Float16>()?
                .iter()
                .all(|v| v.to_f32().is_finite()),
            "nonfinite text embedding",
        )?;
        Ok(selected)
    }

    async fn prepare_denoiser(&mut self, tokens: usize, text_rows: usize) -> Result<bool> {
        let (paths, specialized) = self.assets.select(tokens, text_rows);
        let paths = paths.clone();
        if self
            .denoiser
            .as_ref()
            .is_some_and(|(loaded, _)| loaded == &paths)
        {
            return Ok(specialized);
        }
        // Bound residency to one export; dynamic signatures can handle varying
        // dimensions without keeping one model for every request shape.
        self.denoiser = None;
        self.runtime.drain().await?;
        let options = LoadOptions::new(Compute::GpuPreferred).with_expect_frequent_reshapes(true);
        let mut functions = Vec::new();
        match self.options.loads {
            LoadPolicy::Parallel => {
                let requests = paths
                    .iter()
                    .map(|path| ModelLoadRequest {
                        path: path.clone(),
                        options,
                    })
                    .collect();
                // Corsairs joins all four loads and cleans up failures.
                let models = self.runtime.load_models(requests).await?;
                for model in models {
                    functions.push(model.load_function("segment").await?);
                }
            }
            LoadPolicy::Serial => {
                for path in &paths {
                    functions.push(load(&self.runtime, path, "segment", true).await?);
                }
            }
        }
        validate_segments(&functions, tokens, text_rows)?;
        self.denoiser = Some((paths, functions));
        Ok(specialized)
    }

    async fn render(
        &mut self,
        request: &ImageRequest,
        text: Tensor,
        noise: Option<Vec<f32>>,
        cancel: &CancellationToken,
        progress: &mut (impl FnMut(Progress<'_>) + Send),
    ) -> Result<GeneratedImage> {
        let (rows, cols) = math::grid(request.size)?;
        let tokens = rows * cols;
        let schedule = FlowSchedule::turbo(request.steps.unwrap_or(8))?;
        let mut latents = noise.unwrap_or_else(|| math::seeded_noise(request.seed, tokens * 64));
        validate_noise(Some(&latents), tokens * 64)?;
        cancel.check()?;
        let specialized = self.prepare_denoiser(tokens, text.shape()[0]).await?;
        progress(Progress::DenoiserReady { specialized });
        cancel.check()?;
        let text = self.runtime.upload(text).await?;
        let rope = self
            .runtime
            .upload(Tensor::from_vec(
                [tokens, 2, 128],
                math::rotary(rows, cols),
            )?)
            .await?;
        let functions = &self.denoiser.as_ref().expect("denoiser loaded").1;
        for step in 0..schedule.timesteps.len() {
            cancel.check()?;
            let latent_input = self.runtime.upload(halves(&latents, [tokens, 64])?).await?;
            let sigma = self
                .runtime
                .upload(Tensor::from_vec([1], vec![schedule.timesteps[step]])?)
                .await?;
            // Preserve all named carries until the step completes: cond may be
            // consumed by multiple nonadjacent segments. No host carry copies.
            let mut carries = TensorHandleMap::new();
            let mut velocity = None;
            for function in functions {
                cancel.check()?;
                let mut inputs = TensorHandleMap::new();
                for input in &function.descriptor().inputs {
                    let tensor = match input.name.as_str() {
                        "latents" => latent_input.clone(),
                        "text_embeds" => text.clone(),
                        "sigma" => sigma.clone(),
                        "rope_image" => rope.clone(),
                        name => carries
                            .get(name)
                            .cloned()
                            .ok_or_else(|| Error::Invalid(format!("missing carry {name}")))?,
                    };
                    inputs.insert(&input.name, tensor);
                }
                let result = function.run_native(inputs, TensorHandleMap::new()).await?;
                require(result.states.is_empty(), "unexpected Krea denoiser state")?;
                for (name, tensor) in result.outputs.iter() {
                    let (dtype, shape) = binding(name, tokens, text.shape()[0])?;
                    require(
                        tensor.dtype() == dtype && tensor.shape() == shape,
                        format!("invalid carry/velocity {name}"),
                    )?;
                    if name == "velocity" {
                        velocity = Some(tensor.clone());
                    } else {
                        carries.insert(name, tensor.clone());
                    }
                }
            }
            let velocity = velocity
                .ok_or_else(|| Error::Invalid("denoiser omitted velocity".into()))?
                .download()
                .await?;
            verify_tensor(&velocity, DType::Float16, &[tokens, 64], "velocity")?;
            let values: Vec<_> = velocity
                .to_vec::<Float16>()?
                .into_iter()
                .map(Float16::to_f32)
                .collect();
            schedule.advance(step, &mut latents, &values)?;
            progress(Progress::Step {
                index: step,
                total: schedule.timesteps.len(),
                latents: &latents,
            });
        }
        cancel.check()?;
        if self.decoder.is_none() {
            self.decoder = Some(load(&self.runtime, &self.assets.decoder, "decode", false).await?);
        }
        let input = TensorMap::from([("latents", halves(&latents, [rows, cols, 64])?)]);
        let mut result = self
            .decoder
            .as_ref()
            .expect("decoder loaded")
            .run(input, TensorMap::new())
            .await?;
        require(result.states.is_empty(), "unexpected decoder state")?;
        let image = result
            .outputs
            .remove("image")
            .ok_or_else(|| Error::Invalid("decoder omitted image".into()))?;
        verify_tensor(
            &image,
            DType::Float32,
            &[3, request.size.height, request.size.width],
            "pixels",
        )?;
        cancel.check()?;
        let image = GeneratedImage::from_chw(request.size, image.to_vec::<f32>()?)?;
        progress(Progress::Decoded);
        Ok(image)
    }

    /// Explicitly release resident models and prompt cache. In-flight jobs are
    /// drained transitively before this returns; the pipeline remains reusable.
    /// Core AI and the OS may retain caches and mapped pages.
    pub async fn unload(&mut self) -> Result<()> {
        self.encoder = None;
        self.decoder = None;
        self.denoiser = None;
        self.text_cache.clear();
        self.runtime.drain().await?;
        Ok(())
    }
}

impl ImagePipeline for KreaPipeline {
    async fn generate(&mut self, request: &ImageRequest) -> Result<GeneratedImage> {
        self.generate_with(request, None, &CancellationToken::default(), &mut |_| {})
            .await
    }
}

fn validate_noise(noise: Option<&[f32]>, count: usize) -> Result<()> {
    if let Some(noise) = noise {
        require(
            noise.len() == count && noise.iter().all(|v| v.is_finite()),
            "invalid initial noise shape or value",
        )?;
    }
    Ok(())
}
fn halves(values: &[f32], shape: impl Into<Vec<usize>>) -> Result<Tensor> {
    require(
        values
            .iter()
            .all(|v| v.is_finite() && Float16::from_f32(*v).to_f32().is_finite()),
        "value outside finite float16 range",
    )?;
    Ok(Tensor::from_vec(
        shape,
        values.iter().copied().map(Float16::from_f32).collect(),
    )?)
}
async fn load(
    runtime: &Runtime,
    path: &std::path::Path,
    name: &str,
    reshapes: bool,
) -> Result<Function> {
    let options = LoadOptions::new(Compute::GpuPreferred).with_expect_frequent_reshapes(reshapes);
    Ok(runtime
        .load_model_with_options(path, options)
        .await?
        .load_function(name)
        .await?)
}
fn verify_tensor(tensor: &Tensor, dtype: DType, shape: &[usize], name: &str) -> Result<()> {
    require(
        tensor.dtype() == dtype && tensor.shape() == shape,
        format!(
            "{name}: expected {dtype:?} {shape:?}, got {:?} {:?}",
            tensor.dtype(),
            tensor.shape()
        ),
    )
}
fn binding(name: &str, tokens: usize, text: usize) -> Result<(DType, Vec<usize>)> {
    Ok(match name {
        "latents" | "velocity" => (DType::Float16, vec![tokens, 64]),
        "text_embeds" => (DType::Float16, vec![text, 30720]),
        "rope_image" => (DType::Float32, vec![tokens, 2, 128]),
        "sigma" => (DType::Float32, vec![1]),
        "cond" => (DType::Float32, vec![8, 6144]),
        "image_1" | "image_2" | "image_3" => (DType::Float32, vec![tokens, 6144]),
        "text_1" | "text_2" | "text_3" => (DType::Float32, vec![text, 6144]),
        other => return Err(Error::Unsupported(format!("Krea binding {other}"))),
    })
}
fn validate_segments(functions: &[Function], tokens: usize, text: usize) -> Result<()> {
    require(functions.len() == 4, "expected four Krea denoiser segments")?;
    let mut available = vec!["latents", "text_embeds", "sigma", "rope_image"];
    let mut has_velocity = false;
    for (index, function) in functions.iter().enumerate() {
        let descriptor = function.descriptor();
        require(
            descriptor.states.is_empty(),
            "stateful Krea denoiser is unsupported",
        )?;
        for input in &descriptor.inputs {
            require(
                available.contains(&input.name.as_str()),
                format!("segment {index} has missing upstream {}", input.name),
            )?;
        }
        for desc in descriptor.inputs.iter().chain(&descriptor.outputs) {
            let (dtype, shape) = binding(&desc.name, tokens, text)?;
            require(
                dtype == desc.dtype
                    && shape.len() == desc.shape.len()
                    && shape
                        .iter()
                        .zip(&desc.shape)
                        .all(|(actual, declared)| declared.is_none_or(|d| d == *actual)),
                format!(
                    "segment {index} incompatible binding {} for requested shape",
                    desc.name
                ),
            )?;
        }
        for output in &descriptor.outputs {
            if output.name == "velocity" {
                require(index == 3, "velocity must come from final segment")?;
                has_velocity = true;
            }
            available.push(&output.name);
        }
    }
    require(has_velocity, "last segment has no velocity output")
}
