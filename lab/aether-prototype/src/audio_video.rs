//! Joint audio/video artifacts: lossless RGB8 frames and unquantized float PCM.
use crate::{Error, ImageSize, Result, error::require};
use std::{
    future::Future,
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    process::Command,
};

/// A decoded keyframe file. The selected model owns preprocessing semantics.
#[derive(Debug, Clone)]
pub struct AudioVideoRequest {
    pub prompt: String,
    pub size: ImageSize,
    pub frames: usize,
    pub seed: u64,
    pub first_frame: Option<PathBuf>,
    pub last_frame: Option<PathBuf>,
}
impl AudioVideoRequest {
    pub fn new(prompt: impl Into<String>, size: ImageSize, frames: usize, seed: u64) -> Self {
        Self {
            prompt: prompt.into(),
            size,
            frames,
            seed,
            first_frame: None,
            last_frame: None,
        }
    }
}

/// A separate modality boundary: audio/video models do not inherit image
/// requests, image diffusion schedules, or image-model residency policies.
pub trait AudioVideoPipeline {
    fn generate(
        &mut self,
        request: &AudioVideoRequest,
        output: &Path,
    ) -> impl Future<Output = Result<GeneratedAudioVideo>> + Send;
}

/// Interleaved float32 PCM, preserving decoded samples without clipping.
#[derive(Debug, Clone)]
pub struct PcmAudio {
    sample_rate: u32,
    channels: u16,
    samples: Vec<f32>,
}
impl PcmAudio {
    pub fn new(sample_rate: u32, channels: u16, samples: Vec<f32>) -> Result<Self> {
        require(
            sample_rate > 0 && (1..=8).contains(&channels),
            "invalid PCM sample rate or channel count",
        )?;
        require(
            !samples.is_empty()
                && samples.len().is_multiple_of(channels as usize)
                && samples.iter().all(|v| v.is_finite()),
            "invalid PCM sample shape or value",
        )?;
        Ok(Self {
            sample_rate,
            channels,
            samples,
        })
    }
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }
    pub fn channels(&self) -> u16 {
        self.channels
    }
    pub fn samples(&self) -> &[f32] {
        &self.samples
    }
    pub fn sample_frames(&self) -> usize {
        self.samples.len() / self.channels as usize
    }
    pub fn save_wav(&self, path: impl AsRef<Path>) -> Result<()> {
        let bytes = self
            .samples
            .len()
            .checked_mul(4)
            .and_then(|v| u32::try_from(v).ok())
            .ok_or_else(|| Error::Invalid("PCM exceeds WAV32 capacity".into()))?;
        let riff_size = bytes
            .checked_add(50)
            .ok_or_else(|| Error::Invalid("PCM exceeds RIFF capacity".into()))?;
        let byte_rate = self
            .sample_rate
            .checked_mul(self.channels as u32)
            .and_then(|v| v.checked_mul(4))
            .ok_or_else(|| Error::Invalid("PCM byte rate overflow".into()))?;
        let frames = u32::try_from(self.sample_frames())
            .map_err(|_| Error::Invalid("PCM frame count overflow".into()))?;
        let mut file = BufWriter::new(
            std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(path)?,
        );
        file.write_all(b"RIFF")?;
        file.write_all(&riff_size.to_le_bytes())?;
        file.write_all(b"WAVEfmt ")?;
        file.write_all(&18u32.to_le_bytes())?;
        file.write_all(&3u16.to_le_bytes())?;
        file.write_all(&self.channels.to_le_bytes())?;
        file.write_all(&self.sample_rate.to_le_bytes())?;
        file.write_all(&byte_rate.to_le_bytes())?;
        file.write_all(&(self.channels * 4).to_le_bytes())?;
        file.write_all(&32u16.to_le_bytes())?;
        file.write_all(&0u16.to_le_bytes())?;
        file.write_all(b"fact")?;
        file.write_all(&4u32.to_le_bytes())?;
        file.write_all(&frames.to_le_bytes())?;
        file.write_all(b"data")?;
        file.write_all(&bytes.to_le_bytes())?;
        for sample in &self.samples {
            file.write_all(&sample.to_le_bytes())?;
        }
        file.flush()?;
        Ok(())
    }
}

/// Video is represented by bounded PNG frames and separate source PCM, avoiding
/// one full decoded clip tensor in Rust. Muxing is an explicit lossy operation.
#[derive(Debug, Clone)]
pub struct GeneratedAudioVideo {
    directory: PathBuf,
    size: ImageSize,
    frames: usize,
    fps: u32,
    sample_rate: u32,
    channels: u16,
}
impl GeneratedAudioVideo {
    pub fn directory(&self) -> &Path {
        &self.directory
    }
    pub fn size(&self) -> ImageSize {
        self.size
    }
    pub fn frame_count(&self) -> usize {
        self.frames
    }
    pub fn fps(&self) -> u32 {
        self.fps
    }
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }
    pub fn channels(&self) -> u16 {
        self.channels
    }
    pub fn audio_path(&self) -> PathBuf {
        self.directory.join("audio.wav")
    }
    /// The caller supplies a permitted FFmpeg executable. No shell is involved;
    /// existing output files are never overwritten. Lossless sources remain.
    pub fn mux_mp4(&self, ffmpeg: impl AsRef<Path>, output: impl AsRef<Path>) -> Result<()> {
        let output = if output.as_ref().is_absolute() {
            output.as_ref().to_path_buf()
        } else {
            std::env::current_dir()?.join(output)
        };
        match std::fs::symlink_metadata(&output) {
            Ok(_) => return Err(Error::Invalid("MP4 output already exists".into())),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(error.into()),
        }
        require(
            self.size.width.is_multiple_of(2) && self.size.height.is_multiple_of(2),
            "H.264 yuv420p requires even dimensions",
        )?;
        let result = Command::new(ffmpeg.as_ref())
            .args([
                "-hide_banner",
                "-loglevel",
                "error",
                "-nostdin",
                "-n",
                "-framerate",
            ])
            .arg(self.fps.to_string())
            .arg("-i")
            .arg(self.directory.join("frames/%08d.png"))
            .arg("-i")
            .arg(self.audio_path())
            .args([
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-frames:v",
            ])
            .arg(self.frames.to_string())
            .arg(&output)
            .output()?;
        require(
            result.status.success(),
            format!("FFmpeg failed: {}", String::from_utf8_lossy(&result.stderr)),
        )
    }
}

/// Receives frames as they become available. Model decoders must retain their
/// required context; this sink does not license arbitrary temporal splitting.
pub struct AudioVideoWriter {
    directory: PathBuf,
    size: ImageSize,
    expected_frames: usize,
    written_frames: usize,
    fps: u32,
}
impl AudioVideoWriter {
    /// Create a new output directory; refuse to mix a new clip with old frames.
    pub fn create(
        directory: impl AsRef<Path>,
        size: ImageSize,
        frames: usize,
        fps: u32,
    ) -> Result<Self> {
        require(
            size.width > 0
                && size.height > 0
                && size
                    .width
                    .checked_mul(size.height)
                    .and_then(|n| n.checked_mul(3))
                    .is_some(),
            "invalid video size",
        )?;
        require(
            frames > 0 && frames <= 99_999_999 && fps > 0,
            "invalid video duration",
        )?;
        let directory = directory.as_ref().to_path_buf();
        std::fs::create_dir(&directory)?;
        std::fs::create_dir(directory.join("frames"))?;
        Ok(Self {
            directory,
            size,
            expected_frames: frames,
            written_frames: 0,
            fps,
        })
    }
    /// Write already quantized RGB bytes. Float decoded samples should be
    /// checked/saved by the model acceptance runner before this conversion.
    pub fn write_frame(&mut self, rgb: &[u8]) -> Result<()> {
        require(
            self.written_frames < self.expected_frames
                && rgb.len() == self.size.width * self.size.height * 3,
            "video frame count or RGB shape mismatch",
        )?;
        let width = u32::try_from(self.size.width)
            .map_err(|_| Error::Invalid("frame width exceeds u32".into()))?;
        let height = u32::try_from(self.size.height)
            .map_err(|_| Error::Invalid("frame height exceeds u32".into()))?;
        let file = BufWriter::new(std::fs::File::create(
            self.directory
                .join(format!("frames/{:08}.png", self.written_frames)),
        )?);
        let mut encoder = png::Encoder::new(file, width, height);
        encoder.set_color(png::ColorType::Rgb);
        encoder.set_depth(png::BitDepth::Eight);
        encoder.set_source_srgb(png::SrgbRenderingIntent::Perceptual);
        let mut writer = encoder.write_header()?;
        writer.write_image_data(rgb)?;
        writer.finish()?;
        self.written_frames += 1;
        Ok(())
    }
    pub fn finish(self, audio: &PcmAudio) -> Result<GeneratedAudioVideo> {
        require(
            self.written_frames == self.expected_frames,
            "decoder did not supply every video frame",
        )?;
        let video_duration = self.expected_frames as f64 / self.fps as f64;
        let audio_duration = audio.sample_frames() as f64 / audio.sample_rate as f64;
        require(
            (video_duration - audio_duration).abs() <= 1.0 / self.fps as f64,
            "decoded audio/video duration differs by more than one frame",
        )?;
        audio.save_wav(self.directory.join("audio.wav"))?;
        let metadata = serde_json::json!({"schema":1,"width":self.size.width,"height":self.size.height,
            "frames":self.expected_frames,"fps":self.fps,"sample_rate":audio.sample_rate,"channels":audio.channels,
            "sample_frames":audio.sample_frames(),"frame_pattern":"frames/%08d.png","audio":"audio.wav"});
        std::fs::write(
            self.directory.join("media.json"),
            serde_json::to_vec_pretty(&metadata)?,
        )?;
        Ok(GeneratedAudioVideo {
            directory: self.directory,
            size: self.size,
            frames: self.expected_frames,
            fps: self.fps,
            sample_rate: audio.sample_rate,
            channels: audio.channels,
        })
    }
}
