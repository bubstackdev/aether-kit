//! Source VAE spatial tiling outside the native graph. Pixel-space boundaries
//! are recorded by the original VAE, including offsets on odd latent rows.
use crate::{ImageSize, Result, error::require};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct H3SpatialTiling {
    pub tile_height: usize,
    pub tile_width: usize,
    pub y_starts: Vec<usize>,
    pub x_starts: Vec<usize>,
    pub y_overlaps: Vec<usize>,
    pub x_overlaps: Vec<usize>,
}
impl H3SpatialTiling {
    pub fn validate(&self, size: ImageSize) -> Result<()> {
        require(
            size.width.is_multiple_of(32)
                && size.height.is_multiple_of(32)
                && size
                    .width
                    .checked_mul(size.height)
                    .is_some_and(|n| n > 0 && n <= 1_032_192),
            "H3 spatial canvas exceeds the profile contract",
        )?;
        require(
            self.tile_height == 256 && self.tile_width == 256,
            "H3 spatial tile export must retain the source 256px tile",
        )?;
        for (extent, tile, starts, overlaps) in [
            (
                size.height,
                self.tile_height,
                &self.y_starts,
                &self.y_overlaps,
            ),
            (
                size.width,
                self.tile_width,
                &self.x_starts,
                &self.x_overlaps,
            ),
        ] {
            require(
                !starts.is_empty()
                    && starts.len() <= 64
                    && starts[0] == 0
                    && starts.len() == overlaps.len() + 1
                    && starts.last().and_then(|v| v.checked_add(tile)) == Some(extent),
                "H3 spatial tile grid does not cover the canvas",
            )?;
            require(
                starts.iter().all(|s| s.is_multiple_of(16))
                    && overlaps
                        .iter()
                        .all(|o| *o > 0 && *o < tile && o.is_multiple_of(16))
                    && starts
                        .windows(2)
                        .zip(overlaps)
                        .all(|(s, o)| s[0].checked_add(tile - o) == Some(s[1])),
                "H3 spatial tile offsets/overlaps differ",
            )?;
        }
        Ok(())
    }
    pub fn tile_count(&self) -> usize {
        self.y_starts.len() * self.x_starts.len()
    }
    /// Extract seven frames from globally packed `[T*H/2*W/2,96]` latents.
    /// Tile starts need not align with the global 2x2 spatial patches.
    pub fn extract(&self, size: ImageSize, packed: &[f32], index: usize) -> Result<Vec<f32>> {
        self.validate(size)?;
        require(
            index < self.tile_count()
                && packed.len() == 7 * (size.height / 32) * (size.width / 32) * 96,
            "H3 spatial packed input differs",
        )?;
        let y0 = self.y_starts[index / self.x_starts.len()] / 16;
        let x0 = self.x_starts[index % self.x_starts.len()] / 16;
        let mut tile =
            Vec::with_capacity(7 * (self.tile_height / 32) * (self.tile_width / 32) * 96);
        for f in 0..7 {
            for y in 0..self.tile_height / 32 {
                for x in 0..self.tile_width / 32 {
                    for c in 0..24 {
                        for dy in 0..2 {
                            for dx in 0..2 {
                                let (gy, gx) = (y0 + y * 2 + dy, x0 + x * 2 + dx);
                                let at = ((f * (size.height / 32) + gy / 2) * (size.width / 32)
                                    + gx / 2)
                                    * 96
                                    + c * 4
                                    + (gy % 2) * 2
                                    + gx % 2;
                                tile.push(packed[at]);
                            }
                        }
                    }
                }
            }
        }
        Ok(tile)
    }
}

pub struct H3SpatialStitcher {
    tiling: H3SpatialTiling,
    size: ImageSize,
    frames: usize,
    next: usize,
    above: Vec<Vec<f32>>,
    output: Vec<f32>,
}
impl H3SpatialStitcher {
    pub fn new(tiling: H3SpatialTiling, size: ImageSize, frames: usize) -> Result<Self> {
        tiling.validate(size)?;
        require(
            frames == 22,
            "H3 spatial window must contain 22 decoded frames",
        )?;
        Ok(Self {
            above: vec![Vec::new(); tiling.x_starts.len()],
            tiling,
            size,
            frames,
            next: 0,
            output: vec![0.0; 3 * frames * size.height * size.width],
        })
    }
    /// Original NCTHW data with singleton N removed. Blend vertically first,
    /// then horizontally, using original neighboring tiles before cropping.
    pub fn push(&mut self, original: Vec<f32>) -> Result<()> {
        let t = &self.tiling;
        let (h, w) = (t.tile_height, t.tile_width);
        require(
            self.next < t.tile_count()
                && original.len() == 3 * self.frames * h * w
                && original.iter().all(|v| v.is_finite()),
            "H3 spatial decoded tile count/shape/values differ",
        )?;
        let (iy, ix) = (self.next / t.x_starts.len(), self.next % t.x_starts.len());
        let mut pixels = original.clone();
        if iy > 0 {
            let overlap = t.y_overlaps[iy - 1];
            for plane in 0..3 * self.frames {
                for y in 0..overlap {
                    for x in 0..w {
                        let at = (plane * h + y) * w + x;
                        let weight = y as f32 / overlap as f32;
                        pixels[at] = product(
                            self.above[ix][(plane * h + h - overlap + y) * w + x],
                            1.0 - weight,
                        ) + product(pixels[at], weight);
                    }
                }
            }
        }
        if ix > 0 {
            let overlap = t.x_overlaps[ix - 1];
            for plane in 0..3 * self.frames {
                for y in 0..h {
                    for x in 0..overlap {
                        let at = (plane * h + y) * w + x;
                        let weight = x as f32 / overlap as f32;
                        pixels[at] = product(
                            self.above[ix - 1][(plane * h + y) * w + w - overlap + x],
                            1.0 - weight,
                        ) + product(pixels[at], weight);
                    }
                }
            }
        }
        let crop_h = h - t.y_overlaps.get(iy).copied().unwrap_or(0);
        let crop_w = w - t.x_overlaps.get(ix).copied().unwrap_or(0);
        for plane in 0..3 * self.frames {
            for y in 0..crop_h {
                let to = (plane * self.size.height + t.y_starts[iy] + y) * self.size.width
                    + t.x_starts[ix];
                let from = (plane * h + y) * w;
                self.output[to..to + crop_w].copy_from_slice(&pixels[from..from + crop_w]);
            }
        }
        self.above[ix] = original;
        self.next += 1;
        Ok(())
    }
    pub fn finish(self) -> Result<Vec<f32>> {
        require(
            self.next == self.tiling.tile_count(),
            "H3 spatial decoder did not supply every tile",
        )?;
        Ok(self.output)
    }
}
#[inline(never)]
fn product(a: f32, b: f32) -> f32 {
    a * b
}
