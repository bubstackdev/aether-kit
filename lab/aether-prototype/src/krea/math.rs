use crate::{ImageSize, Result, error::require};

/// Reference schedule: Double linspace, float32 shift, integer-scale timestep
/// roundtrip, final zero. Preserve these operations and their rounding order.
#[derive(Debug, Clone)]
pub struct FlowSchedule {
    pub sigmas: Vec<f32>,
    pub timesteps: Vec<f32>,
}

impl FlowSchedule {
    pub fn turbo(steps: usize) -> Result<Self> {
        require((1..=50).contains(&steps), "Krea steps must be in 1..=50")?;
        let stop = 1.0 / steps as f64;
        let delta = if steps > 1 {
            (stop - 1.0) / (steps - 1) as f64
        } else {
            0.0
        };
        let e = 1.15f64.exp() as f32;
        let mut sigmas: Vec<_> = (0..steps)
            .map(|i| {
                let base = if i == steps - 1 && steps > 1 {
                    stop
                } else {
                    i as f64 * delta + 1.0
                } as f32;
                e / (e + (1.0 / base - 1.0))
            })
            .collect();
        let timesteps = sigmas.iter().map(|s| *s * 1000.0 / 1000.0).collect();
        sigmas.push(0.0);
        Ok(Self { sigmas, timesteps })
    }
    pub fn advance(&self, index: usize, latents: &mut [f32], velocity: &[f32]) -> Result<()> {
        require(
            self.sigmas.len() == self.timesteps.len() + 1
                && index < self.timesteps.len()
                && latents.len() == velocity.len(),
            "invalid Euler step",
        )?;
        require(velocity.iter().all(|v| v.is_finite()), "nonfinite velocity")?;
        let delta = self.sigmas[index + 1] - self.sigmas[index];
        for (x, v) in latents.iter_mut().zip(velocity) {
            *x += rounded_product(delta, *v);
        }
        require(
            latents.iter().all(|v| v.is_finite()),
            "nonfinite Euler result",
        )
    }
}

// Separate non-inlined multiply prevents fused multiply-add contraction.
#[inline(never)]
fn rounded_product(a: f32, b: f32) -> f32 {
    a * b
}

/// SplitMix64 + Double Box-Muller, in packed token order, matching Aether's
/// Swift seeded noise. Hugging Face/PyTorch seeds use a different generator.
pub fn seeded_noise(seed: u64, count: usize) -> Vec<f32> {
    fn next(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }
    let mut state = seed;
    let mut values = Vec::with_capacity(count);
    while values.len() < count {
        let u1 = ((next(&mut state) >> 11) as f64 + 1.0) / 9007199254740992.0;
        let u2 = (next(&mut state) >> 11) as f64 / 9007199254740992.0;
        let rho = (-2.0 * u1.ln()).sqrt();
        let angle = 2.0 * std::f64::consts::PI * u2;
        values.push((rho * angle.cos()) as f32);
        if values.len() < count {
            values.push((rho * angle.sin()) as f32);
        }
    }
    values
}

pub(crate) fn grid(size: ImageSize) -> Result<(usize, usize)> {
    require(
        size.width > 0
            && size.height > 0
            && size.width.is_multiple_of(16)
            && size.height.is_multiple_of(16),
        "Krea dimensions must be positive multiples of 16",
    )?;
    let rows = size.height / 16;
    let cols = size.width / 16;
    require(
        rows.checked_mul(cols).is_some_and(|n| n <= 4096),
        "Krea asset limit is 4096 image tokens",
    )?;
    Ok((rows, cols))
}

pub(crate) fn rotary(rows: usize, cols: usize) -> Vec<f32> {
    let mut frequencies = Vec::new();
    for (axis, dim) in [32, 48, 48].into_iter().enumerate() {
        for j in (0..dim).step_by(2) {
            frequencies.push((axis, 1.0 / 1000f64.powf(j as f64 / dim as f64)));
        }
    }
    let mut out = vec![0.0; rows * cols * 2 * 128];
    for row in 0..rows {
        for col in 0..cols {
            let base = (row * cols + col) * 256;
            for (i, &(axis, frequency)) in frequencies.iter().enumerate() {
                let angle = [0, row, col][axis] as f64 * frequency;
                let c = angle.cos() as f32;
                let s = angle.sin() as f32;
                out[base + i] = c;
                out[base + i + 64] = c;
                out[base + i + 128] = s;
                out[base + i + 192] = s;
            }
        }
    }
    out
}

/// Float32 image rotary rows `[image_tokens,2,128]`, in the lab's split-halves
/// layout. Exposed for offline input inspection and independent parity checks.
pub fn image_rotary(size: ImageSize) -> Result<Vec<f32>> {
    let (rows, cols) = grid(size)?;
    Ok(rotary(rows, cols))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_rejects_invalid_sizes_before_allocation() {
        for size in [
            ImageSize::new(0, 16),
            ImageSize::new(17, 16),
            ImageSize::new(2048, 2048),
            ImageSize::new(usize::MAX, 16),
        ] {
            assert!(image_rotary(size).is_err());
        }
        assert_eq!(image_rotary(ImageSize::new(16, 16)).unwrap().len(), 256);
    }
}
