//! The source VAE's temporal boundaries: seven latent frames per call, stride
//! five, seventeen output frames plus a five-frame overlap. Blend normalized
//! decoder samples before ImageNet denormalization or quantization.
use crate::{ImageSize, Result, error::require};

pub struct H3VideoStitcher {
    area: usize,
    windows: usize,
    expected_windows: usize,
    tail: Option<Vec<f32>>,
}
impl H3VideoStitcher {
    pub fn new(size: ImageSize, frames: usize) -> Result<Self> {
        require(
            frames >= 22 && (frames - 5).is_multiple_of(17),
            "H3 streaming decode requires 17*n+5 frames, n>=1",
        )?;
        let area = size
            .width
            .checked_mul(size.height)
            .filter(|n| *n > 0 && n.checked_mul(66).is_some())
            .ok_or_else(|| crate::Error::Invalid("H3 decoder canvas overflow".into()))?;
        Ok(Self {
            area,
            windows: 0,
            expected_windows: (frames - 5) / 17,
            tail: None,
        })
    }
    /// Input is CHW F32 `[3,22,H,W]`, output is frame-major `[17,3,H,W]`.
    pub fn push(&mut self, window: &[f32]) -> Result<Vec<f32>> {
        require(
            self.windows < self.expected_windows
                && window.len() == self.area * 66
                && window.iter().all(|v| v.is_finite()),
            "H3 decoded window count, shape or values differ",
        )?;
        let mut frames = Vec::with_capacity(self.area * 51);
        for f in 0..17 {
            for c in 0..3 {
                for p in 0..self.area {
                    let value = window[(c * 22 + f) * self.area + p];
                    let value = if f < 5 {
                        if let Some(tail) = &self.tail {
                            let weight = f as f32 / 5.0;
                            product(tail[(c * 5 + f) * self.area + p], 1.0 - weight)
                                + product(value, weight)
                        } else {
                            value
                        }
                    } else {
                        value
                    };
                    frames.push(value);
                }
            }
        }
        let mut tail = Vec::with_capacity(self.area * 15);
        for c in 0..3 {
            tail.extend_from_slice(&window[(c * 22 + 17) * self.area..(c * 22 + 22) * self.area]);
        }
        self.tail = Some(tail);
        self.windows += 1;
        Ok(frames)
    }
    /// The final five frames, frame-major `[5,3,H,W]`.
    pub fn finish(self) -> Result<Vec<f32>> {
        require(
            self.windows == self.expected_windows,
            "H3 decoder did not supply every temporal window",
        )?;
        let tail = self.tail.expect("at least one checked window");
        let mut frames = Vec::with_capacity(tail.len());
        for f in 0..5 {
            for c in 0..3 {
                frames
                    .extend_from_slice(&tail[(c * 5 + f) * self.area..(c * 5 + f + 1) * self.area]);
            }
        }
        Ok(frames)
    }
}
#[inline(never)]
fn product(a: f32, b: f32) -> f32 {
    a * b
}
