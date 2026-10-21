use crate::{Result, error::require};
use std::{future::Future, io::BufWriter, path::Path};

/// Pixel dimensions. Each model family applies its own grid and size limits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImageSize {
    pub width: usize,
    pub height: usize,
}

impl ImageSize {
    pub const fn new(width: usize, height: usize) -> Self {
        Self { width, height }
    }
}

#[derive(Debug, Clone)]
pub struct ImageRequest {
    pub prompt: String,
    pub size: ImageSize,
    pub seed: u64,
    /// None uses the model family's default.
    pub steps: Option<usize>,
}

impl ImageRequest {
    pub fn new(prompt: impl Into<String>, size: ImageSize, seed: u64) -> Self {
        Self {
            prompt: prompt.into(),
            size,
            seed,
            steps: None,
        }
    }
}

/// Common image-generation boundary. Other modalities can define their own
/// request/artifact traits without inheriting image or diffusion assumptions.
pub trait ImagePipeline {
    fn generate(
        &mut self,
        request: &ImageRequest,
    ) -> impl Future<Output = Result<GeneratedImage>> + Send;
}

/// Decoded float32 planar RGB, on the model's [-1, 1] scale. Values are kept
/// before clipping so parity and downstream processing can use the real output.
#[derive(Debug, Clone)]
pub struct GeneratedImage {
    size: ImageSize,
    pixels: Vec<f32>,
}

impl GeneratedImage {
    pub fn from_chw(size: ImageSize, pixels: Vec<f32>) -> Result<Self> {
        let count = size
            .width
            .checked_mul(size.height)
            .and_then(|n| n.checked_mul(3));
        require(
            size.width > 0 && size.height > 0 && count == Some(pixels.len()),
            "invalid RGB image dimensions",
        )?;
        require(
            pixels.iter().all(|v| v.is_finite()),
            "nonfinite decoded pixel",
        )?;
        Ok(Self { size, pixels })
    }
    pub fn size(&self) -> ImageSize {
        self.size
    }
    pub fn pixels_chw(&self) -> &[f32] {
        &self.pixels
    }
    /// Matches the Swift lab's float32 arithmetic and ties-to-even rounding.
    pub fn rgb8(&self) -> Vec<u8> {
        let n = self.size.width * self.size.height;
        (0..n)
            .flat_map(|i| {
                (0..3).map(move |c| {
                    (((self.pixels[c * n + i] / 2.0 + 0.5).clamp(0.0, 1.0) * 255.0)
                        .round_ties_even()) as u8
                })
            })
            .collect()
    }
    pub fn save_png(&self, path: impl AsRef<Path>) -> Result<()> {
        let width = u32::try_from(self.size.width)
            .map_err(|_| crate::Error::Invalid("PNG width exceeds u32".into()))?;
        let height = u32::try_from(self.size.height)
            .map_err(|_| crate::Error::Invalid("PNG height exceeds u32".into()))?;
        let file = BufWriter::new(std::fs::File::create(path)?);
        let mut encoder = png::Encoder::new(file, width, height);
        encoder.set_color(png::ColorType::Rgb);
        encoder.set_depth(png::BitDepth::Eight);
        let mut writer = encoder.write_header()?;
        writer.write_image_data(&self.rgb8())?;
        writer.finish()?;
        Ok(())
    }
}
