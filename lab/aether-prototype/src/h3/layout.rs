//! Packed FL2VA row geometry, following the pinned Diffusers behavior contract.
use super::{H3Geometry, H3Step};
use crate::{Result, error::require};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum H3Anchor {
    First,
    Last,
}

#[derive(Debug, Clone)]
pub struct H3Layout {
    /// Source float64 coordinates narrowed once, as the transformer's rope does.
    pub positions: Vec<f32>,
    pub token_tags: Vec<i32>,
    pub video_indices: Vec<i32>,
    pub audio_indices: Vec<i32>,
    pub text_indices: Vec<i32>,
    pub condition_video_rows: usize,
    pub audio_latents_per_channel: usize,
}
#[derive(Debug, Clone)]
pub struct H3RowTimesteps {
    pub values: Vec<f32>,
    pub indices: Vec<i32>,
    pub adaln_indices: Vec<i32>,
}
impl H3Layout {
    pub fn new(geometry: H3Geometry, text_tags: &[i32], anchors: &[H3Anchor]) -> Result<Self> {
        require(
            !text_tags.is_empty()
                && text_tags.len() <= 4096
                && text_tags.iter().all(|t| [0, 1].contains(t)),
            "invalid H3 text row tags",
        )?;
        require(
            anchors.len() == geometry.keyframes && anchors.len() <= 2,
            "H3 keyframe layout differs from request",
        )?;
        require(
            anchors.len() < 2 || anchors == [H3Anchor::First, H3Anchor::Last],
            "H3 keyframes must be first then last",
        )?;
        let n = text_tags.len();
        let h = geometry.latent_height;
        let w = geometry.latent_width;
        require(
            h > 0
                && w > 0
                && h.is_multiple_of(2)
                && w.is_multiple_of(2)
                && geometry.latent_frames > 0
                && h.checked_mul(w)
                    .and_then(|n| n.checked_mul(geometry.latent_frames))
                    .is_some_and(|n| n / 4 == geometry.target_video_rows)
                && geometry.frames > 0
                && geometry.frames <= 360,
            "invalid H3 latent geometry",
        )?;
        let frame_rows = h / 2 * (w / 2);
        let condition_video_rows = frame_rows * anchors.len();
        let audio_latents_per_channel =
            (geometry.frames as f64 / 24.0 * 40.0).round_ties_even() as usize;
        let audio_start = n + condition_video_rows;
        let video_start = audio_start + audio_latents_per_channel * 2;
        let rows = video_start + geometry.target_video_rows;
        require(rows <= i32::MAX as usize, "H3 packed indices exceed i32")?;
        let mut positions = vec![0.0; rows * 3];
        for i in 0..n {
            positions[i * 3] = i as f32;
        }
        let sqrt_area = ((h * w) as f64).sqrt();
        let grid = |dim: usize| {
            let ratio = dim as f64 / sqrt_area;
            let left = (1.0 - ratio) / 2.0;
            let step = ((left + ratio) - left) / (dim / 2) as f64;
            (0..dim / 2)
                .map(|i| (left + i as f64 * step) * 32.0)
                .collect::<Vec<_>>()
        };
        let hg = grid(h);
        let wg = grid(w);
        let spans: Vec<_> = (0..geometry.latent_frames)
            .map(|i| (5.0 / 3.0) * if i % 5 == 0 { 1.0 } else { 4.0 })
            .collect();
        let mut temporal = Vec::with_capacity(spans.len());
        let mut cumulative = 0.0;
        for span in &spans {
            temporal.push(n as f64 + cumulative);
            cumulative += span;
        }
        let mut frame = |start: usize, time: f64| {
            for (y, hy) in hg.iter().enumerate() {
                for (x, wx) in wg.iter().enumerate() {
                    let at = (start + y * wg.len() + x) * 3;
                    positions[at..at + 3].copy_from_slice(&[time as f32, *hy as f32, *wx as f32]);
                }
            }
        };
        for (i, anchor) in anchors.iter().enumerate() {
            let t = if *anchor == H3Anchor::First {
                n as f64
            } else {
                n as f64 + spans.iter().sum::<f64>() - 5.0 / 3.0
            };
            frame(n + i * frame_rows, t);
        }
        for (i, t) in temporal.iter().enumerate() {
            frame(video_start + i * frame_rows, *t);
        }
        for channel in 0..2 {
            for t in 0..audio_latents_per_channel {
                let at = (audio_start + channel * audio_latents_per_channel + t) * 3;
                positions[at] = (n + t) as f32;
                positions[at + 2] = if channel == 0 {
                    wg[0]
                } else {
                    wg[wg.len() - 1]
                } as f32;
            }
        }
        let mut token_tags = vec![0; rows];
        token_tags[..n].copy_from_slice(text_tags);
        token_tags[audio_start..video_start].fill(2);
        Ok(Self {
            positions,
            token_tags,
            video_indices: (n..audio_start)
                .chain(video_start..rows)
                .map(|i| i as i32)
                .collect(),
            audio_indices: (audio_start..video_start).map(|i| i as i32).collect(),
            text_indices: (0..n).map(|i| i as i32).collect(),
            condition_video_rows,
            audio_latents_per_channel,
        })
    }
    /// Deduplicate actual row noise levels in sorted order. Text inherits video
    /// time, and keyframe rows remain at max(video time, 0.999).
    pub fn timesteps(&self, step: H3Step) -> H3RowTimesteps {
        let mut per_row = vec![step.video_timestep; self.token_tags.len()];
        for &i in &self.audio_indices {
            per_row[i as usize] = step.audio_timestep;
        }
        for &i in &self.video_indices[..self.condition_video_rows] {
            per_row[i as usize] = step.video_timestep.max(0.999);
        }
        let mut values = per_row.clone();
        values.sort_by(f32::total_cmp);
        values.dedup();
        let indices: Vec<_> = per_row
            .iter()
            .map(|v| {
                values
                    .iter()
                    .position(|x| x == v)
                    .expect("timestep present") as i32
            })
            .collect();
        let adaln_indices = indices
            .iter()
            .zip(&self.token_tags)
            .map(|(i, t)| i * 3 + t)
            .collect();
        H3RowTimesteps {
            values,
            indices,
            adaln_indices,
        }
    }
}

/// Convert contiguous `[24,T,H,W]` source noise into `[T*H/2*W/2,96]` rows.
pub fn pack_video(latents: &[f32], geometry: H3Geometry) -> Result<Vec<f32>> {
    let (t, h, w) = (
        geometry.latent_frames,
        geometry.latent_height,
        geometry.latent_width,
    );
    require(
        t > 0
            && h > 0
            && w > 0
            && h.is_multiple_of(2)
            && w.is_multiple_of(2)
            && t.checked_mul(h)
                .and_then(|n| n.checked_mul(w))
                .and_then(|n| n.checked_mul(24))
                .is_some(),
        "invalid H3 patch geometry",
    )?;
    require(
        latents.len() == 24 * t * h * w && latents.iter().all(|v| v.is_finite()),
        "invalid H3 video latents",
    )?;
    let mut rows = Vec::with_capacity(latents.len());
    for f in 0..t {
        for y in (0..h).step_by(2) {
            for x in (0..w).step_by(2) {
                for c in 0..24 {
                    for dy in 0..2 {
                        for dx in 0..2 {
                            rows.push(latents[((c * t + f) * h + y + dy) * w + x + dx]);
                        }
                    }
                }
            }
        }
    }
    Ok(rows)
}
