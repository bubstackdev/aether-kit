use super::assets::{ModelAsset, hash};
use super::{H3Assets, H3Geometry, H3Layout, H3VideoStitcher, pack_video};
use crate::{
    AudioVideoPipeline, AudioVideoRequest, AudioVideoWriter, Error, GeneratedAudioVideo, PcmAudio,
    Result, error::require,
};
use corsairs::{
    BFloat16, DType, Function, MetalBufferMap, MetalValueMap, Runtime, Tensor, TensorMap,
};
use std::time::Instant;
use std::{
    collections::VecDeque,
    path::{Path, PathBuf},
};

#[derive(Debug, Clone, Default)]
pub struct H3Options {
    /// Explicitly permit geometry outside the production 5–15 second range.
    pub diagnostic: bool,
    /// Save unquantized decoder outputs beside PNG/WAV for acceptance checks.
    pub retain_decoded: bool,
    /// Trusted application cache for native time embeddings and modulation
    /// tables. Identity includes weights, adapter and exact schedule/plan bits.
    pub preparation_cache: Option<PathBuf>,
}
#[derive(Debug)]
pub enum H3Progress {
    PreparationCache {
        key: String,
        hit: bool,
    },
    /// Completed wall-clock phase, including public API completion/readback.
    Timing {
        stage: &'static str,
        index: Option<usize>,
        seconds: f64,
    },
    Encoded,
    ConditionerLayer {
        index: usize,
    },
    Modulation {
        block: usize,
    },
    WeightsLoaded {
        block: usize,
    },
    Step {
        index: usize,
        video_sha256: String,
        audio_sha256: String,
    },
    AudioDecoded {
        sha256: String,
    },
    VideoDecoded {
        sha256: String,
    },
    VideoWindow {
        index: usize,
    },
}

/// Authoritative F32 latent rows, also usable to replay an external RNG exactly.
pub struct H3InitialLatents {
    pub video: Vec<f32>,
    pub audio: Vec<f32>,
}

/// Full T2VA orchestration over source-precision exported components. Models
/// are acquired by stage; conditioner/body/decoder weights never coexist.
pub struct H3Pipeline {
    assets: H3Assets,
    runtime: Runtime,
    options: H3Options,
    verified: bool,
    text_cache: VecDeque<(Vec<i32>, Tensor)>,
}
impl H3Pipeline {
    pub fn new(assets: H3Assets, options: H3Options) -> Result<Self> {
        Self::with_runtime(assets, options, Runtime::new()?)
    }
    pub fn with_runtime(assets: H3Assets, options: H3Options, runtime: Runtime) -> Result<Self> {
        Ok(Self {
            assets,
            runtime,
            options,
            verified: false,
            text_cache: VecDeque::new(),
        })
    }
    fn check(&self, request: &AudioVideoRequest) -> Result<H3Geometry> {
        require(
            request.first_frame.is_none() && request.last_frame.is_none(),
            "H3 T2VA bundle does not include vision/keyframe encoding",
        )?;
        let geometry = H3Geometry::for_request(request, self.options.diagnostic)?;
        let m = &self.assets.manifest;
        require(
            request.size.width == m.width
                && request.size.height == m.height
                && geometry.frames == m.frames,
            "H3 request geometry has no matching export profile",
        )?;
        Ok(geometry)
    }
    fn check_output(output: &Path) -> Result<()> {
        match std::fs::symlink_metadata(output) {
            Ok(_) => return Err(Error::Invalid("H3 output path already exists".into())),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(error.into()),
        }
        let parent = output
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        require(parent.is_dir(), "H3 output parent directory is missing")
    }
    fn verify(&mut self) -> Result<()> {
        if !self.verified {
            self.assets.verify()?;
            self.verified = true;
        }
        Ok(())
    }
    async fn encode(
        &mut self,
        request: &AudioVideoRequest,
        progress: &mut (impl FnMut(H3Progress) + Send),
    ) -> Result<Tensor> {
        let tokenizer = tokenizers::Tokenizer::from_file(&self.assets.manifest.tokenizer)
            .map_err(|e| Error::Tokenizer(e.to_string()))?;
        let encoded = tokenizer
            .encode(request.prompt.as_str(), false)
            .map_err(|e| Error::Tokenizer(e.to_string()))?;
        let ids: Vec<_> = encoded
            .get_ids()
            .iter()
            .map(|i| {
                i32::try_from(*i).map_err(|_| Error::Invalid("H3 token ID exceeds i32".into()))
            })
            .collect::<Result<_>>()?;
        require(
            ids.len() == self.assets.manifest.text_rows,
            "H3 prompt token count has no matching export profile",
        )?;
        if let Some(index) = self.text_cache.iter().position(|(key, _)| key == &ids) {
            let entry = self.text_cache.remove(index).expect("cache entry");
            let tensor = entry.1.clone();
            self.text_cache.push_front(entry);
            return Ok(tensor);
        }
        let c=self.assets.manifest.conditioner.as_ref().ok_or_else(||Error::Unsupported("H3 bundle has no conditioner; export it or supply original conditioning explicitly".into()))?;
        let function = c.embedding.load(&self.runtime).await?;
        let result = function
            .run(
                TensorMap::from([("ids", Tensor::from_vec([1, ids.len()], ids.clone())?)]),
                TensorMap::new(),
            )
            .await;
        drop(function);
        self.runtime.drain().await?;
        let mut result = result?;
        require(
            result.states.is_empty() && result.outputs.len() == 3,
            "H3 embedding outputs differ",
        )?;
        let mut hidden = take(&mut result.outputs, "hidden")?;
        let cos = take(&mut result.outputs, "cos")?;
        let sin = take(&mut result.outputs, "sin")?;
        for (index, layer) in c.layers.iter().enumerate() {
            let function = layer.load(&self.runtime).await?;
            let result = function
                .run(
                    TensorMap::from([
                        ("hidden", hidden),
                        ("cos", cos.clone()),
                        ("sin", sin.clone()),
                    ]),
                    TensorMap::new(),
                )
                .await;
            drop(function);
            self.runtime.drain().await?;
            let mut result = result?;
            require(
                result.states.is_empty() && result.outputs.len() == 1,
                "H3 conditioner layer outputs differ",
            )?;
            hidden = take(&mut result.outputs, "hidden_out")?;
            check_tensor(
                &hidden,
                DType::BFloat16,
                &[1, ids.len(), 5120],
                "conditioner hidden",
            )?;
            progress(H3Progress::ConditionerLayer { index });
        }
        // No final normalization: hidden_states[50] is the unnormalized tap.
        self.text_cache.push_front((ids, hidden.clone()));
        self.text_cache.truncate(2);
        Ok(hidden)
    }
    pub async fn encode_prompt(&mut self, request: &AudioVideoRequest) -> Result<Tensor> {
        self.check(request)?;
        self.verify()?;
        self.runtime.drain().await?;
        let result = self.encode(request, &mut |_| {}).await;
        self.runtime.drain().await?;
        result
    }
    pub fn initial_latents(&self, request: &AudioVideoRequest) -> Result<H3InitialLatents> {
        let geometry = self.check(request)?;
        let video_count = geometry.target_video_rows * 96;
        let audio_count =
            (geometry.frames as f64 / 24.0 * 40.0).round_ties_even() as usize * 2 * 32;
        // One deterministic Aether RNG stream: contiguous video latent tensor,
        // then channel-major audio. PyTorch replay uses supplied latent rows.
        let noise = crate::krea::seeded_noise(request.seed, video_count + audio_count);
        Ok(H3InitialLatents {
            video: pack_video(&noise[..video_count], geometry)?,
            audio: noise[video_count..].to_vec(),
        })
    }
    pub async fn generate_with(
        &mut self,
        request: &AudioVideoRequest,
        output: &Path,
        progress: &mut (impl FnMut(H3Progress) + Send),
    ) -> Result<GeneratedAudioVideo> {
        self.check(request)?;
        Self::check_output(output)?;
        let started = Instant::now();
        self.verify()?;
        timing(progress, "integrity", None, started);
        self.runtime.drain().await?;
        let result = async {
            let started = Instant::now();
            let text = self.encode(request, progress).await?;
            timing(progress, "conditioning", None, started);
            progress(H3Progress::Encoded);
            let latents = self.initial_latents(request)?;
            self.run_conditioned(
                request,
                text,
                latents,
                Some(output),
                self.assets.recipe().evaluations(),
                progress,
            )
            .await?
            .ok_or_else(|| Error::Invalid("H3 generation produced no media".into()))
        }
        .await;
        self.runtime.drain().await?;
        result
    }
    /// Original supplied text/noise for parity and explicit conditioning reuse.
    /// The caller owns their association with the prompt; no stub is substituted.
    pub async fn generate_conditioned(
        &mut self,
        request: &AudioVideoRequest,
        text: Tensor,
        latents: H3InitialLatents,
        output: &Path,
        progress: &mut (impl FnMut(H3Progress) + Send),
    ) -> Result<GeneratedAudioVideo> {
        self.check(request)?;
        Self::check_output(output)?;
        self.verify()?;
        self.runtime.drain().await?;
        let result = self
            .run_conditioned(
                request,
                text,
                latents,
                Some(output),
                self.assets.recipe().evaluations(),
                progress,
            )
            .await
            .and_then(|clip| {
                clip.ok_or_else(|| Error::Invalid("H3 generation produced no media".into()))
            });
        self.runtime.drain().await?;
        result
    }
    /// Measure a bounded prefix of the bundle's workload, then both full
    /// decoders. This is not a shorter sampling schedule or a finished render.
    /// All recipe modulation plans are prepared, and every measured step runs
    /// all 50 blocks. Decoding includes host readback, stitching and hashes;
    /// PNG/WAV encoding and mux are omitted. No media artifact is returned.
    pub async fn benchmark_with(
        &mut self,
        request: &AudioVideoRequest,
        denoising_steps: usize,
        progress: &mut (impl FnMut(H3Progress) + Send),
    ) -> Result<()> {
        require(
            (1..=self.assets.recipe().evaluations()).contains(&denoising_steps),
            "H3 benchmark prefix exceeds the bundle's recipe",
        )?;
        self.check(request)?;
        let started = Instant::now();
        self.verify()?;
        timing(progress, "integrity", None, started);
        self.runtime.drain().await?;
        let result = async {
            let started = Instant::now();
            let text = self.encode(request, progress).await?;
            timing(progress, "conditioning", None, started);
            progress(H3Progress::Encoded);
            let latents = self.initial_latents(request)?;
            self.run_conditioned(request, text, latents, None, denoising_steps, progress)
                .await?;
            Ok(())
        }
        .await;
        self.runtime.drain().await?;
        result
    }
    /// Run the input projections and two text-refiner layers separately for
    /// comparison with an independently captured original-model boundary.
    pub async fn prepare_conditioned(
        &mut self,
        request: &AudioVideoRequest,
        text: Tensor,
        latents: H3InitialLatents,
    ) -> Result<Tensor> {
        let geometry = self.check(request)?;
        self.verify()?;
        check_tensor(
            &text,
            DType::BFloat16,
            &[1, self.assets.manifest.text_rows, 5120],
            "provided conditioning",
        )?;
        let function = self.assets.manifest.prepare.load(&self.runtime).await?;
        let result = function
            .run(
                TensorMap::from([
                    (
                        "video",
                        Tensor::from_vec([1, geometry.target_video_rows, 96], latents.video)?,
                    ),
                    (
                        "audio",
                        Tensor::from_vec(
                            [
                                1,
                                (geometry.frames as f64 / 24.0 * 40.0).round_ties_even() as usize
                                    * 2,
                                32,
                            ],
                            latents.audio,
                        )?,
                    ),
                    ("text", text),
                ]),
                TensorMap::new(),
            )
            .await;
        drop(function);
        self.runtime.drain().await?;
        let mut result = result?;
        require(
            result.states.is_empty() && result.outputs.len() == 1,
            "H3 prepare outputs differ",
        )?;
        take(&mut result.outputs, "hidden")
    }
    async fn run_conditioned(
        &self,
        request: &AudioVideoRequest,
        text: Tensor,
        mut latents: H3InitialLatents,
        output: Option<&Path>,
        denoising_steps: usize,
        progress: &mut (impl FnMut(H3Progress) + Send),
    ) -> Result<Option<GeneratedAudioVideo>> {
        let geometry = self.check(request)?;
        let m = &self.assets.manifest;
        let text_host = text.clone();
        check_tensor(
            &text,
            DType::BFloat16,
            &[1, m.text_rows, 5120],
            "provided conditioning",
        )?;
        let layout = H3Layout::new(geometry, &vec![1; m.text_rows], &[])?;
        let nv = geometry.target_video_rows;
        let na = layout.audio_indices.len();
        require(
            latents.video.len() == nv * 96
                && latents.audio.len() == na * 32
                && latents
                    .video
                    .iter()
                    .chain(&latents.audio)
                    .all(|v| v.is_finite()),
            "H3 initial latent rows differ or contain nonfinite data",
        )?;
        if let Some(output) = output {
            require(!output.exists(), "H3 output directory already exists")?;
        }
        let plans: Vec<_> = self
            .assets
            .schedule
            .steps()
            .map(|step| layout.timesteps(step))
            .collect();
        let cache_key = super::cache::key(m, &plans)?;
        let cache_started = Instant::now();
        let cached = self
            .options
            .preparation_cache
            .as_ref()
            .map(|root| super::cache::Prepared::load(root, &cache_key, &plans))
            .transpose()?
            .flatten();
        timing(progress, "preparation_cache_load", None, cache_started);
        if self.options.preparation_cache.is_some() {
            progress(H3Progress::PreparationCache {
                key: cache_key.clone(),
                hit: cached.is_some(),
            });
        }
        let super::cache::Prepared { embeddings, tables } = if let Some(prepared) = cached {
            prepared
        } else {
            let started = Instant::now();
            let time = m.time.load(&self.runtime).await?;
            let mut embeddings = Vec::with_capacity(plans.len());
            for plan in &plans {
                let mut result = time
                    .run(
                        TensorMap::from([(
                            "timestep",
                            Tensor::from_vec([plan.values.len()], plan.values.clone())?,
                        )]),
                        TensorMap::new(),
                    )
                    .await?;
                require(
                    result.states.is_empty() && result.outputs.len() == 1,
                    "H3 time embedding outputs differ",
                )?;
                embeddings.push(take(&mut result.outputs, "temb")?);
            }
            drop(time);
            self.runtime.drain().await?;
            timing(progress, "time_embeddings", None, started);
            let started = Instant::now();
            let mut tables = Vec::with_capacity(50);
            for (block, asset) in m.modulations.iter().enumerate() {
                let function = asset.load(&self.runtime).await?;
                let result = async {
                    let mut block_tables = Vec::with_capacity(plans.len());
                    for (embedding, plan) in embeddings.iter().zip(&plans) {
                        let mut result = function
                            .run(
                                TensorMap::from([("temb", embedding.clone())]),
                                TensorMap::new(),
                            )
                            .await?;
                        require(
                            result.states.is_empty() && result.outputs.len() == 1,
                            "H3 modulation outputs differ",
                        )?;
                        let table = take(&mut result.outputs, "modulations")?;
                        check_tensor(
                            &table,
                            DType::BFloat16,
                            &[6, plan.values.len() * 3, 5376],
                            "modulation table",
                        )?;
                        block_tables.push(table);
                    }
                    Ok::<_, Error>(block_tables)
                }
                .await;
                drop(function);
                self.runtime.drain().await?;
                tables.push(result?);
                progress(H3Progress::Modulation { block });
            }
            timing(progress, "modulations", None, started);
            let prepared = super::cache::Prepared { embeddings, tables };
            if let Some(root) = &self.options.preparation_cache {
                let started = Instant::now();
                prepared.save(root, &cache_key, &plans)?;
                timing(progress, "preparation_cache_write", None, started);
            }
            prepared
        };
        // Body parameters go directly into persistent shared Metal storage.
        // No owned NDArray parameter residency or external raw pointers.
        let started = Instant::now();
        let queue = self.runtime.metal_queue().await?;
        let mut parameters = Vec::with_capacity(50);
        for (block, files) in m.parameters.iter().enumerate() {
            let mut values = MetalValueMap::new();
            for file in files {
                values.insert(&file.name, queue.upload(file.read()?).await?);
            }
            parameters.push(values);
            progress(H3Progress::WeightsLoaded { block });
        }
        let cos = queue.upload(m.rotary_cos.read()?).await?;
        let sin = queue.upload(m.rotary_sin.read()?).await?;
        let text = queue.upload(text).await?;
        let video = queue
            .upload(Tensor::from_vec([1, nv, 96], latents.video.clone())?)
            .await?;
        let audio = queue
            .upload(Tensor::from_vec([1, na, 32], latents.audio.clone())?)
            .await?;
        let buffers = [
            queue
                .buffer(DType::BFloat16, [1, layout.token_tags.len(), 5376])
                .await?,
            queue
                .buffer(DType::BFloat16, [1, layout.token_tags.len(), 5376])
                .await?,
        ];
        let prepare = m.prepare.load(&self.runtime).await?;
        let heads = m.heads.load(&self.runtime).await?;
        timing(progress, "body_setup", None, started);
        let mut body: Option<(usize, Function)> = None;
        for (index, plan) in plans.iter().take(denoising_steps).enumerate() {
            let step_started = Instant::now();
            let cardinality = plan.values.len();
            if body.as_ref().is_none_or(|(n, _)| *n != cardinality) {
                let started = Instant::now();
                drop(body.take());
                self.runtime.drain().await?;
                body = Some((
                    cardinality,
                    m.bodies[cardinality - 1].load(&self.runtime).await?,
                ));
                timing(progress, "body_graph_load", Some(index), started);
            }
            let started = Instant::now();
            video
                .write(Tensor::from_vec([1, nv, 96], latents.video.clone())?)
                .await?;
            audio
                .write(Tensor::from_vec([1, na, 32], latents.audio.clone())?)
                .await?;
            let result = prepare
                .encode(
                    &queue,
                    MetalValueMap::from([("video", &video), ("audio", &audio), ("text", &text)]),
                    MetalBufferMap::new(),
                    MetalBufferMap::from([("hidden", buffers[0].clone())]),
                )
                .await?;
            require(
                result.states.is_empty() && result.outputs.len() == 1,
                "H3 prepare outputs differ",
            )?;
            drop(result);
            timing(progress, "step_prepare", Some(index), started);
            let started = Instant::now();
            let tags = queue
                .upload(Tensor::from_vec(
                    [layout.token_tags.len()],
                    plan.adaln_indices.clone(),
                )?)
                .await?;
            let mut slot = 0;
            for (block, weights) in parameters.iter().enumerate() {
                let block_started = Instant::now();
                let table = queue.upload(tables[block][index].clone()).await?;
                let mut inputs = weights.clone();
                inputs.insert("hidden", &buffers[slot]);
                inputs.insert("temb", table);
                inputs.insert("tags", &tags);
                inputs.insert("cos", &cos);
                inputs.insert("sin", &sin);
                let next = 1 - slot;
                let result = body
                    .as_ref()
                    .expect("body acquired")
                    .1
                    .encode(
                        &queue,
                        inputs,
                        MetalBufferMap::new(),
                        MetalBufferMap::from([("hidden_out", buffers[next].clone())]),
                    )
                    .await?;
                require(
                    result.states.is_empty() && result.outputs.len() == 1,
                    "H3 body outputs differ",
                )?;
                drop(result);
                slot = next;
                timing(
                    progress,
                    "body_block",
                    Some(index * 50 + block),
                    block_started,
                );
            }
            timing(progress, "step_body", Some(index), started);
            let started = Instant::now();
            let temb = queue.upload(embeddings[index].clone()).await?;
            let indices = queue
                .upload(Tensor::from_vec(
                    [layout.token_tags.len()],
                    plan.indices.clone(),
                )?)
                .await?;
            let mut result = heads
                .encode(
                    &queue,
                    MetalValueMap::from([
                        ("hidden", &buffers[slot]),
                        ("temb", &temb),
                        ("indices", &indices),
                    ]),
                    MetalBufferMap::new(),
                    MetalBufferMap::new(),
                )
                .await?;
            require(
                result.states.is_empty() && result.outputs.len() == 2,
                "H3 head outputs differ",
            )?;
            let vv = result
                .outputs
                .remove("video")
                .ok_or_else(|| Error::Invalid("H3 video velocity missing".into()))?
                .download()
                .await?;
            let av = result
                .outputs
                .remove("audio")
                .ok_or_else(|| Error::Invalid("H3 audio velocity missing".into()))?
                .download()
                .await?;
            check_tensor(&vv, DType::Float32, &[1, nv, 96], "video velocity")?;
            check_tensor(&av, DType::Float32, &[1, na, 32], "audio velocity")?;
            timing(progress, "step_heads", Some(index), started);
            let started = Instant::now();
            self.assets
                .schedule
                .advance_video(index, &mut latents.video, &vv.to_vec::<f32>()?)?;
            self.assets
                .schedule
                .advance_audio(index, &mut latents.audio, &av.to_vec::<f32>()?)?;
            progress(H3Progress::Step {
                index,
                video_sha256: float_hash(&latents.video),
                audio_sha256: float_hash(&latents.audio),
            });
            timing(progress, "step_update_and_hash", Some(index), started);
            timing(progress, "step_total", Some(index), step_started);
        }
        let started = Instant::now();
        drop((
            body, heads, prepare, buffers, video, audio, text, cos, sin, parameters, queue, tables,
            embeddings,
        ));
        self.runtime.drain().await?;
        timing(progress, "body_release", None, started);
        self.decode_latents(request, latents, output, Some(text_host), progress)
            .await
    }
    /// Decode seeded synthetic latents at the exact profile geometry to measure
    /// decoder feasibility separately. This produces no clip or quality claim.
    pub async fn benchmark_decoders_with(
        &mut self,
        request: &AudioVideoRequest,
        progress: &mut (impl FnMut(H3Progress) + Send),
    ) -> Result<()> {
        let latents = self.initial_latents(request)?;
        let started = Instant::now();
        self.verify()?;
        timing(progress, "integrity", None, started);
        self.runtime.drain().await?;
        let result = self
            .decode_latents(request, latents, None, None, progress)
            .await;
        self.runtime.drain().await?;
        result.map(|_| ())
    }
    async fn decode_latents(
        &self,
        request: &AudioVideoRequest,
        latents: H3InitialLatents,
        output: Option<&Path>,
        text_host: Option<Tensor>,
        progress: &mut (impl FnMut(H3Progress) + Send),
    ) -> Result<Option<GeneratedAudioVideo>> {
        let geometry = self.check(request)?;
        let m = &self.assets.manifest;
        let nv = geometry.target_video_rows;
        let na = (geometry.frames as f64 / 24.0 * 40.0).round_ties_even() as usize * 2;
        let audio_rows = Tensor::from_vec([na, 32], latents.audio)?;
        let video_rows = Tensor::from_vec([nv, 96], latents.video)?;
        let started = Instant::now();
        let audio = run_decode(&self.runtime, &m.audio_decoder, "pcm", audio_rows.clone()).await?;
        timing(progress, "audio_decoder", None, started);
        require(
            audio.dtype() == DType::Float32 && audio.shape().len() == 2 && audio.shape()[0] == 2,
            "H3 PCM must be F32 [2,samples]",
        )?;
        let samples = audio.to_vec::<f32>()?;
        require(
            samples.iter().all(|v| v.is_finite()),
            "nonfinite H3 decoded audio",
        )?;
        progress(H3Progress::AudioDecoded {
            sha256: hash(audio.as_bytes()),
        });
        let n = audio.shape()[1];
        let pcm = PcmAudio::new(
            32000,
            2,
            (0..n).flat_map(|i| [samples[i], samples[n + i]]).collect(),
        )?;
        require(
            (pcm.sample_frames() as f64 / 32000.0 - geometry.frames as f64 / 24.0).abs()
                <= 1.0 / 24.0,
            "H3 decoded audio/video duration differs by more than one frame",
        )?;
        let mut writer = output
            .map(|output| AudioVideoWriter::create(output, request.size, geometry.frames, 24))
            .transpose()?;
        if let Some(output) = output.filter(|_| self.options.retain_decoded) {
            let text_host = text_host.as_ref().ok_or_else(|| {
                Error::Invalid("retained generation conditioning is missing".into())
            })?;
            std::fs::write(output.join("conditioning.bf16"), text_host.as_bytes())?;
            std::fs::write(
                output.join("conditioning.json"),
                serde_json::to_vec_pretty(
                    &serde_json::json!({"dtype":"bfloat16","shape":text_host.shape(),"sha256":hash(text_host.as_bytes()),"tap":50,"final_norm":false}),
                )?,
            )?;
            std::fs::write(output.join("video-latents.f32"), video_rows.as_bytes())?;
            std::fs::write(output.join("audio-latents.f32"), audio_rows.as_bytes())?;
            std::fs::write(output.join("audio.f32"), audio.as_bytes())?;
            std::fs::write(
                output.join("decoded.json"),
                serde_json::to_vec_pretty(
                    &serde_json::json!({"schema":1,"video":{"file":"video.f32","shape":[geometry.frames,3,request.size.height,request.size.width],"layout":"FCHW","dtype":"float32","normalization":"ImageNet RGB, before clipping"},"audio":{"file":"audio.f32","shape":audio.shape(),"dtype":"float32","layout":"channel-major"}}),
                )?,
            )?;
        }
        let mut raw_video = output
            .filter(|_| self.options.retain_decoded)
            .map(|output| std::fs::File::create(output.join("video.f32")))
            .transpose()?;
        use sha2::Digest;
        let mut pixel_hash = sha2::Sha256::new();
        let mut stitcher = H3VideoStitcher::new(request.size, geometry.frames)?;
        let video_started = Instant::now();
        let started = Instant::now();
        let function = m.video_decoder.load(&self.runtime).await?;
        timing(progress, "video_decoder_load", None, started);
        let started = Instant::now();
        let video_queue = if m.video_parameters.is_empty() {
            None
        } else {
            Some(self.runtime.metal_queue().await?)
        };
        let mut video_parameters = MetalValueMap::new();
        if let Some(queue) = &video_queue {
            for file in &m.video_parameters {
                video_parameters.insert(&file.name, queue.upload(file.read()?).await?);
            }
        }
        timing(progress, "video_decoder_parameters", None, started);
        let frame_rows = geometry.latent_height / 2 * (geometry.latent_width / 2);
        let windows = (geometry.frames - 5) / 17;
        for index in 0..windows {
            let started = Instant::now();
            let start = index * 5 * frame_rows * 96 * 4;
            let end = start + 7 * frame_rows * 96 * 4;
            let rows = Tensor::from_bytes(
                DType::Float32,
                [7 * frame_rows, 96],
                video_rows.as_bytes()[start..end].to_vec(),
            )?;
            let pixels = if let Some(tiling) = &m.video_spatial_tiles {
                let queue = video_queue
                    .as_ref()
                    .ok_or_else(|| Error::Invalid("H3 spatial decoder queue missing".into()))?;
                let packed = rows.to_vec::<f32>()?;
                let mut stitcher = super::H3SpatialStitcher::new(tiling.clone(), request.size, 22)?;
                for tile in 0..tiling.tile_count() {
                    let tile_started = Instant::now();
                    let rows = Tensor::from_vec(
                        [7 * (tiling.tile_height / 32) * (tiling.tile_width / 32), 96],
                        tiling.extract(request.size, &packed, tile)?,
                    )?;
                    let pixels =
                        run_functional_video(&function, queue, &video_parameters, rows).await?;
                    check_tensor(
                        &pixels,
                        DType::Float32,
                        &[1, 3, 22, tiling.tile_height, tiling.tile_width],
                        "decoded spatial tile",
                    )?;
                    stitcher.push(pixels.to_vec::<f32>()?)?;
                    timing(
                        progress,
                        "video_decoder_spatial_tile",
                        Some(index * tiling.tile_count() + tile),
                        tile_started,
                    );
                }
                Tensor::from_vec(
                    [1, 3, 22, request.size.height, request.size.width],
                    stitcher.finish()?,
                )?
            } else if let Some(queue) = &video_queue {
                let mut inputs = video_parameters.clone();
                inputs.insert("rows", queue.upload(rows).await?);
                let mut result = function
                    .encode(queue, inputs, MetalBufferMap::new(), MetalBufferMap::new())
                    .await?;
                require(
                    result.states.is_empty() && result.outputs.len() == 1,
                    "H3 functional video outputs differ",
                )?;
                result
                    .outputs
                    .remove("pixels")
                    .ok_or_else(|| Error::Invalid("H3 functional video pixels missing".into()))?
                    .download()
                    .await?
            } else {
                let mut result = function
                    .run(TensorMap::from([("rows", rows)]), TensorMap::new())
                    .await?;
                require(
                    result.states.is_empty() && result.outputs.len() == 1,
                    "H3 video window outputs differ",
                )?;
                take(&mut result.outputs, "pixels")?
            };
            check_tensor(
                &pixels,
                DType::Float32,
                &[1, 3, 22, request.size.height, request.size.width],
                "decoded video window",
            )?;
            timing(progress, "video_decoder_window", Some(index), started);
            let started = Instant::now();
            let frames = stitcher.push(&pixels.to_vec::<f32>()?)?;
            write_frames(
                &frames,
                request,
                &mut writer,
                &mut raw_video,
                &mut pixel_hash,
            )?;
            timing(progress, "video_stitch_and_output", Some(index), started);
            progress(H3Progress::VideoWindow { index });
        }
        write_frames(
            &stitcher.finish()?,
            request,
            &mut writer,
            &mut raw_video,
            &mut pixel_hash,
        )?;
        drop((function, video_parameters, video_queue));
        self.runtime.drain().await?;
        progress(H3Progress::VideoDecoded {
            sha256: pixel_hash
                .finalize()
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect(),
        });
        timing(progress, "video_decoder_total", None, video_started);
        writer.map(|writer| writer.finish(&pcm)).transpose()
    }
    pub async fn unload(&mut self) -> Result<()> {
        self.text_cache.clear();
        self.runtime.drain().await?;
        Ok(())
    }
}
async fn run_functional_video(
    function: &Function,
    queue: &corsairs::MetalQueue,
    parameters: &MetalValueMap,
    rows: Tensor,
) -> Result<Tensor> {
    let mut inputs = parameters.clone();
    inputs.insert("rows", queue.upload(rows).await?);
    let mut result = function
        .encode(queue, inputs, MetalBufferMap::new(), MetalBufferMap::new())
        .await?;
    require(
        result.states.is_empty() && result.outputs.len() == 1,
        "H3 functional video outputs differ",
    )?;
    result
        .outputs
        .remove("pixels")
        .ok_or_else(|| Error::Invalid("H3 functional video pixels missing".into()))?
        .download()
        .await
        .map_err(Into::into)
}
fn write_frames(
    frames: &[f32],
    request: &AudioVideoRequest,
    writer: &mut Option<AudioVideoWriter>,
    raw: &mut Option<std::fs::File>,
    hasher: &mut sha2::Sha256,
) -> Result<()> {
    use sha2::Digest;
    use std::io::Write;
    let area = request.size.width * request.size.height;
    require(
        frames.len().is_multiple_of(area * 3) && frames.iter().all(|v| v.is_finite()),
        "invalid stitched H3 frames",
    )?;
    let bytes: Vec<_> = frames.iter().flat_map(|v| v.to_le_bytes()).collect();
    hasher.update(&bytes);
    if let Some(file) = raw {
        file.write_all(&bytes)?;
    }
    let Some(writer) = writer else { return Ok(()) };
    let mut rgb = vec![0; area * 3];
    for frame in frames.chunks_exact(area * 3) {
        for p in 0..area {
            for c in 0..3 {
                let normalized = (product(frame[c * area + p], [0.229, 0.224, 0.225][c])
                    + [0.485, 0.456, 0.406][c])
                    .clamp(0.0, 1.0);
                rgb[p * 3 + c] = (normalized * 255.0).round_ties_even() as u8;
            }
        }
        writer.write_frame(&rgb)?;
    }
    Ok(())
}
impl AudioVideoPipeline for H3Pipeline {
    async fn generate(
        &mut self,
        request: &AudioVideoRequest,
        output: &Path,
    ) -> Result<GeneratedAudioVideo> {
        self.generate_with(request, output, &mut |_| {}).await
    }
}
async fn run_decode(
    runtime: &Runtime,
    asset: &ModelAsset,
    name: &str,
    rows: Tensor,
) -> Result<Tensor> {
    let function = asset.load(runtime).await?;
    let result = function
        .run(TensorMap::from([("rows", rows)]), TensorMap::new())
        .await;
    drop(function);
    runtime.drain().await?;
    let mut result = result?;
    require(
        result.states.is_empty() && result.outputs.len() == 1,
        "H3 decoder outputs differ",
    )?;
    take(&mut result.outputs, name)
}
fn take(map: &mut TensorMap, name: &str) -> Result<Tensor> {
    map.remove(name)
        .ok_or_else(|| Error::Invalid(format!("H3 output {name} missing")))
}
fn check_tensor(tensor: &Tensor, dtype: DType, shape: &[usize], name: &str) -> Result<()> {
    require(
        tensor.dtype() == dtype && tensor.shape() == shape,
        format!("H3 {name} signature differs"),
    )?;
    let finite = if dtype == DType::BFloat16 {
        tensor
            .to_vec::<BFloat16>()?
            .iter()
            .all(|v| v.to_f32().is_finite())
    } else {
        tensor.to_vec::<f32>()?.iter().all(|v| v.is_finite())
    };
    require(finite, format!("nonfinite H3 {name}"))
}
fn float_hash(values: &[f32]) -> String {
    hash(
        &values
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>(),
    )
}
fn timing(
    progress: &mut (impl FnMut(H3Progress) + Send),
    stage: &'static str,
    index: Option<usize>,
    started: Instant,
) {
    progress(H3Progress::Timing {
        stage,
        index,
        seconds: started.elapsed().as_secs_f64(),
    });
}
#[inline(never)]
fn product(a: f32, b: f32) -> f32 {
    a * b
}
