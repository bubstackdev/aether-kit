use crate::{AudioVideoRequest, Result, error::require};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// H3's actual geometry, after an explicit upward snap to its VAE frame grid.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct H3Geometry {
    pub frames: usize,
    pub latent_frames: usize,
    pub latent_height: usize,
    pub latent_width: usize,
    pub target_video_rows: usize,
    pub keyframes: usize,
}
impl H3Geometry {
    /// Production duration is 5–15 seconds at 24 fps. `diagnostic` explicitly
    /// allows shorter clips; it does not establish production qualification.
    pub fn for_request(request: &AudioVideoRequest, diagnostic: bool) -> Result<Self> {
        let size = request.size;
        require(
            size.width > 0
                && size.height > 0
                && size.width.is_multiple_of(32)
                && size.height.is_multiple_of(32),
            "H3 dimensions must be positive multiples of 32",
        )?;
        require(
            size.width
                .checked_mul(size.height)
                .is_some_and(|n| n <= 1_032_192),
            "H3 canvas exceeds the pinned baseline pixel budget",
        )?;
        require(
            request.frames > 0 && request.frames <= 360,
            "H3 requested frame count must be in 1..=360",
        )?;
        let n = request.frames.saturating_sub(5).div_ceil(17);
        let frames = 17 * n + 5;
        require(
            frames <= 360 && (diagnostic || frames >= 120),
            "H3 snapped duration must be 5–15 seconds; shorter runs need diagnostic=true",
        )?;
        let latent_frames = 5 * n + 2;
        Ok(Self {
            frames,
            latent_frames,
            latent_height: size.height / 16,
            latent_width: size.width / 16,
            target_video_rows: latent_frames * (size.height / 32) * (size.width / 32),
            keyframes: usize::from(request.first_frame.is_some())
                + usize::from(request.last_frame.is_some()),
        })
    }
}

#[derive(Debug, Clone, Deserialize)]
struct ScheduleArm {
    shift: f32,
    timesteps: Vec<f32>,
    sigmas: Vec<f32>,
}
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ScheduleRecord {
    grid_points: usize,
    expected_forwards: usize,
    schedules: Vec<ScheduleArm>,
}

/// A weight/sampling recipe, never a request to truncate another schedule.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum H3Recipe {
    #[default]
    Base50,
    Turbo8,
}
impl H3Recipe {
    pub fn evaluations(self) -> usize {
        match self {
            Self::Base50 => 50,
            Self::Turbo8 => 8,
        }
    }
}

/// The captured base schedule, not a re-created linspace with different rounding.
/// Custom schedules require a new oracle record; schedule-specific cache reuse
/// is not implicit in this type.
#[derive(Debug, Clone)]
pub struct H3Schedule {
    video: ScheduleArm,
    audio: ScheduleArm,
}
#[derive(Debug, Clone, Copy)]
pub struct H3Step {
    pub index: usize,
    pub video_timestep: f32,
    pub audio_timestep: f32,
    pub video_delta: f32,
    pub audio_delta: f32,
}
impl H3Schedule {
    pub fn from_lab(path: impl AsRef<Path>) -> Result<Self> {
        Self::from_json(&std::fs::read(path)?)
    }
    pub fn from_json(bytes: &[u8]) -> Result<Self> {
        Self::from_json_for_recipe(bytes, H3Recipe::Base50)
    }
    pub fn from_json_for_recipe(bytes: &[u8], recipe: H3Recipe) -> Result<Self> {
        let mut record: ScheduleRecord = serde_json::from_slice(bytes)?;
        let count = recipe.evaluations();
        require(
            record.expected_forwards == count
                && record.grid_points == count + 1
                && record.schedules.len() == 2,
            "H3 schedule count differs from the weight recipe",
        )?;
        if recipe == H3Recipe::Turbo8 {
            let canonical: ScheduleRecord =
                serde_json::from_slice(include_bytes!("recipes/turbo8-schedule.json"))?;
            require(
                record
                    .schedules
                    .iter()
                    .zip(&canonical.schedules)
                    .all(|(a, b)| {
                        a.shift == b.shift
                            && a.timesteps.len() == b.timesteps.len()
                            && a.sigmas.len() == b.sigmas.len()
                            && a.timesteps
                                .iter()
                                .zip(&b.timesteps)
                                .chain(a.sigmas.iter().zip(&b.sigmas))
                                .all(|(a, b)| a.to_bits() == b.to_bits())
                    }),
                "H3 Turbo8 requires the pinned eight-evaluation 12/3 sigma grid",
            )?;
        }
        let audio = record.schedules.pop().expect("two arms checked");
        let video = record.schedules.pop().expect("two arms checked");
        require(
            video.shift == 12.0 && audio.shift == 3.0,
            "H3 video/audio schedule shifts must be 12/3",
        )?;
        for arm in [&video, &audio] {
            require(
                arm.timesteps.len() == count && arm.sigmas.len() == count + 1,
                "invalid H3 schedule lengths",
            )?;
            require(
                arm.sigmas.first() == Some(&1.0)
                    && arm.sigmas.last() == Some(&0.0)
                    && arm
                        .sigmas
                        .iter()
                        .all(|s| s.is_finite() && (0.0..=1.0).contains(s))
                    && arm.sigmas.windows(2).all(|s| s[1] < s[0]),
                "invalid H3 sigma grid",
            )?;
            require(
                arm.timesteps
                    .iter()
                    .zip(&arm.sigmas)
                    .all(|(t, s)| t.is_finite() && t.to_bits() == (1.0 - *s).to_bits()),
                "H3 model timesteps must match the captured inverted sigma grid",
            )?;
        }
        Ok(Self { video, audio })
    }
    pub fn steps(&self) -> impl ExactSizeIterator<Item = H3Step> + '_ {
        (0..self.video.timesteps.len()).map(|index| H3Step {
            index,
            video_timestep: self.video.timesteps[index],
            audio_timestep: self.audio.timesteps[index],
            video_delta: self.video.sigmas[index + 1] - self.video.sigmas[index],
            audio_delta: self.audio.sigmas[index + 1] - self.audio.sigmas[index],
        })
    }
    /// Apply the source scheduler's data-ward velocity update. Its x0 blend
    /// preserves float32 rounding and is not Krea's signed Euler delta.
    pub fn advance_video(&self, index: usize, sample: &mut [f32], velocity: &[f32]) -> Result<()> {
        advance(&self.video, index, sample, velocity)
    }
    pub fn advance_audio(&self, index: usize, sample: &mut [f32], velocity: &[f32]) -> Result<()> {
        advance(&self.audio, index, sample, velocity)
    }
}

#[inline(never)]
fn product(a: f32, b: f32) -> f32 {
    a * b
}
fn advance(arm: &ScheduleArm, index: usize, sample: &mut [f32], velocity: &[f32]) -> Result<()> {
    require(
        index < arm.timesteps.len() && sample.len() == velocity.len(),
        "invalid H3 scheduler step",
    )?;
    require(
        sample.iter().chain(velocity).all(|v| v.is_finite()),
        "nonfinite H3 scheduler input",
    )?;
    let sigma_from_timestep = 1.0 - arm.timesteps[index];
    let ratio = arm.sigmas[index + 1] / arm.sigmas[index];
    for (value, velocity) in sample.iter_mut().zip(velocity) {
        let denoised = *value + product(sigma_from_timestep, *velocity);
        *value = product(ratio, *value) + product(1.0 - ratio, denoised);
    }
    require(
        sample.iter().all(|v| v.is_finite()),
        "nonfinite H3 scheduler output",
    )
}
